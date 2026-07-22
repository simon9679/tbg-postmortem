#!/usr/bin/env python3
"""
Golden-снимок поведения SemanticDecisionLayer (ТЗ E0).

РОЛЬ ЭТАЛОНА
-----------
Снимок фиксирует ДЕТЕРМИНИРОВАННЫЙ выход _sdl.resolve(...) на наборе фикстур.
resolve не зовёт LLM — он использует только эмбеддер MiniLM (all-MiniLM-L6-v2)
через _embed_labels/_get_embed_model().encode, который детерминирован. Значит
при неизменном коде и неизменном эмбеддере выход побайтово воспроизводим.

ЗАЧЕМ:
  Снимок снят на ТЕКУЩЕМ эмбеддере (MiniLM) ПЕРЕД чисткой экстрактора (E2/E3/E4).
  - E2 (рефактор / выпил мёртвого кода): поведение ОБЯЗАНО совпасть → этот снимок
    строго охраняет его. Любое расхождение в --check на E2 = регрессия.
  - E3 (смена эмбеддера) и E4 (Define): НАМЕРЕННО изменят merge/flag-решения
    (они зависят от косинуса). Там снимок ПЕРЕСНИМАЕТСЯ (re-bless) — это ожидаемо,
    а не баг. После re-bless снова должно быть детерминированно.
  Главная ценность снимка: ловить НЕПРЕДНАМЕРЕННЫЕ изменения в логике, не
  зависящей от косинуса — opposition / performative / reversal / type-gating /
  обработке рёбер. Эти ветки эмбеддер-смена менять не должна.

ЧТО СЕРИАЛИЗУЕТСЯ (детерминированно, json sort_keys=True):
  add_nodes             [label, category, concept_id, node_type]
  add_edges             [source_label, relation, target_label]
  reinforce_ids         [плейсхолдер]
  contradict_ids        [плейсхолдер]
  strong_contradict_ids [плейсхолдер]
UUID узлов НЕ пишутся как есть: они случайны на каждом прогоне. Вместо id
подставляются стабильные плейсхолдеры n0, n1, ... в порядке появления
(сначала исходные узлы фикстуры в порядке объявления, затем add_nodes).
Конфиденсы (float) НЕ сериализуются намеренно — они эмбеддер-зависимы и шумны;
ветка решения (структура) уже отражает результат.

ИСПОЛЬЗОВАНИЕ:
  py -3 tools/snapshot_sdl.py            # сгенерировать tests/golden/sdl_snapshot.json
  py -3 tools/snapshot_sdl.py --check    # сравнить с эталоном; exit 1 при расхождении
"""
import os
import sys
import json

# Этот тул — read-only по отношению к проду: ничего в tbg_extractor/engine/адаптере
# не меняем. Импортируем singleton _sdl и схему как есть.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tbg_schema import UserTBG, BeliefNode          # noqa: E402
from tbg_extractor import _sdl                       # noqa: E402

SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "golden", "sdl_snapshot.json",
)


# ---------------------------------------------------------------------------
# Построение исходного UserTBG в памяти (БД не нужна).
# nodes_spec — список dict: label, category, confidence, [node_type], [concept_id].
# Порядок списка == порядок появления плейсхолдеров n0, n1, ...
# ---------------------------------------------------------------------------
def build_tbg(nodes_spec):
    tbg = UserTBG(user_id="snapshot")
    ordered_ids = []
    for spec in nodes_spec:
        node = BeliefNode(
            label=spec["label"],
            category=spec["category"],
            confidence=spec.get("confidence", 0.5),
            source=spec.get("source", "inferred"),
            node_type=spec.get("node_type", ""),
            concept_id=spec.get("concept_id"),
        )
        tbg.set_node(node)
        ordered_ids.append(node.id)
    return tbg, ordered_ids


# ---------------------------------------------------------------------------
# Детерминированная сериализация выхода resolve.
# ---------------------------------------------------------------------------
def serialize_delta(delta, tbg, ordered_existing_ids):
    # Плейсхолдеры: сначала исходные узлы (в порядке объявления фикстуры),
    # затем новые узлы из add_nodes (в порядке их создания resolve).
    placeholder = {}
    counter = 0
    for nid in ordered_existing_ids:
        placeholder[nid] = f"n{counter}"
        counter += 1
    for node in delta.add_nodes:
        if node.id not in placeholder:
            placeholder[node.id] = f"n{counter}"
            counter += 1

    # id -> label для рёбер (исходные узлы + новые узлы).
    id_to_label = {nid: tbg.nodes[nid].label for nid in tbg.nodes}
    for node in delta.add_nodes:
        id_to_label[node.id] = node.label

    def ph(nid):
        # Любой неожиданный id всё равно получает стабильный плейсхолдер.
        if nid not in placeholder:
            placeholder[nid] = f"x{len(placeholder)}"
        return placeholder[nid]

    add_nodes = [
        [n.label, n.category, n.concept_id, n.node_type]
        for n in delta.add_nodes
    ]
    add_edges = [
        [id_to_label.get(e.source_id, "?"), e.relation, id_to_label.get(e.target_id, "?")]
        for e in delta.add_edges
    ]

    return {
        "add_nodes": add_nodes,
        "add_edges": add_edges,
        "reinforce_ids": [ph(i) for i in delta.reinforce_ids],
        "contradict_ids": [ph(i) for i in delta.contradict_ids],
        "strong_contradict_ids": [ph(i) for i in delta.strong_contradict_ids],
    }


# ---------------------------------------------------------------------------
# ФИКСТУРЫ
# Косинусы (MiniLM all-MiniLM-L6-v2, замерены при написании ТЗ E0):
#   MERGE  (sim >= 0.82 глобально, либо >= 0.72 в той же категории) -> reinforce/merge
#   FLAG   (0.72 <= sim < 0.82, РАЗНЫЕ категории)                   -> oscillation/contradict
#   NONE   (sim < 0.72)                                             -> новый узел
# Polarity-слова (_NEGATIVE_POLARITY / _POSITIVE_POLARITY) управляют opposition:
#   stability/security/trust/satisfaction/happiness/autonomy/confidence -> +1
#   instability/insecurity/distrust/dissatisfaction/unhappiness/doubt   -> -1
# ---------------------------------------------------------------------------
FIXTURES = [
    # 1. NONE — новый узел в пустом графе.
    {
        "name": "new_node_empty_graph",
        "desc": "Пустой TBG, один факт -> чистый add_node (ветка _SIM_NONE).",
        "user_text": "I value my personal freedom above all.",
        "nodes": [],
        "raw_facts": [{"label": "values freedom", "category": "values", "confidence": 0.7, "source": "explicit"}],
        "raw_edges": [],
    },

    # 2. NONE + same-category opposition edge (строки 671-693).
    #    'self confidence'(+) vs 'self doubt'(-), sim 0.525 (<0.72 -> NONE),
    #    одна категория -> создаётся conflicts_with к сильнейшему оппоненту.
    {
        "name": "none_same_category_opposition_edge",
        "desc": "NONE-ветка: новый узел + conflicts_with к оппоненту той же категории.",
        "user_text": "Lately I just feel a lot of self doubt.",
        "nodes": [{"label": "self confidence", "category": "mood", "confidence": 0.6}],
        "raw_facts": [{"label": "self doubt", "category": "mood", "confidence": 0.6, "source": "explicit"}],
        "raw_edges": [],
    },

    # 3. MERGE-парафраз -> reinforce (без reversal).
    #    Идентичный label в той же категории, sim 1.0.
    {
        "name": "merge_paraphrase_reinforce",
        "desc": "MERGE: идентичный концепт -> reinforce существующего узла.",
        "user_text": "I really value freedom.",
        "nodes": [{"label": "values freedom", "category": "values", "confidence": 0.6}],
        "raw_facts": [{"label": "values freedom", "category": "values", "confidence": 0.7, "source": "explicit"}],
        "raw_edges": [],
    },

    # 4. MERGE + reversal + opposition -> strong_contradict (existing conf >= 0.75).
    #    'career stability'(+) vs 'career instability'(-), sim 0.889, conf 0.8.
    {
        "name": "merge_reversal_opposition_strong",
        "desc": "MERGE+reversal+opposition, conf>=0.75 -> strong_contradict + uncertain node.",
        "user_text": "Actually, everything feels unstable in my career now.",
        "nodes": [{"label": "career stability", "category": "career", "confidence": 0.8}],
        "raw_facts": [{"label": "career instability", "category": "career", "confidence": 0.6, "source": "explicit"}],
        "raw_edges": [],
    },

    # 5. MERGE + reversal + opposition -> contradict (existing conf < 0.75).
    {
        "name": "merge_reversal_opposition_weak",
        "desc": "MERGE+reversal+opposition, conf<0.75 -> contradict (не strong).",
        "user_text": "Actually my career feels unstable lately.",
        "nodes": [{"label": "career stability", "category": "career", "confidence": 0.6}],
        "raw_facts": [{"label": "career instability", "category": "career", "confidence": 0.6, "source": "explicit"}],
        "raw_edges": [],
    },

    # 6. FLAG oscillation (без reversal).
    #    'values autonomy'(+, values) vs 'seeks autonomy'(+, goals) cross-cat, sim 0.761.
    #    Одинаковая полярность -> не opposition -> oscillation pair + conflicts_with.
    {
        "name": "flag_oscillation_pair",
        "desc": "FLAG, разные категории, без opposition -> новый узел + conflicts_with.",
        "user_text": "I really seek autonomy in everything I do.",
        "nodes": [{"label": "values autonomy", "category": "values", "confidence": 0.6}],
        "raw_facts": [{"label": "seeks autonomy", "category": "goals", "confidence": 0.6, "source": "explicit"}],
        "raw_edges": [],
    },

    # 7. FLAG + reversal + opposition -> contradict (existing conf < 0.75).
    #    'financial security'(+, finances) vs 'financial insecurity'(-, fears) cross-cat, sim 0.794.
    {
        "name": "flag_reversal_opposition_weak",
        "desc": "FLAG+reversal+opposition, conf<0.75 -> contradict + новый узел.",
        "user_text": "Actually I now feel deep financial insecurity.",
        "nodes": [{"label": "financial security", "category": "finances", "confidence": 0.6}],
        "raw_facts": [{"label": "financial insecurity", "category": "fears", "confidence": 0.6, "source": "explicit"}],
        "raw_edges": [],
    },

    # 8. FLAG + reversal + opposition -> strong_contradict (existing conf >= 0.75).
    {
        "name": "flag_reversal_opposition_strong",
        "desc": "FLAG+reversal+opposition, conf>=0.75 -> strong_contradict.",
        "user_text": "Actually I now feel deep financial insecurity.",
        "nodes": [{"label": "financial security", "category": "finances", "confidence": 0.8}],
        "raw_facts": [{"label": "financial insecurity", "category": "fears", "confidence": 0.6, "source": "explicit"}],
        "raw_edges": [],
    },

    # 9. reversal БЕЗ opposition -> reinforce (reversal сам по себе не контрит).
    {
        "name": "reversal_without_opposition_reinforce",
        "desc": "Reversal-язык есть, но концепт тот же и полярность та же -> reinforce.",
        "user_text": "Actually, I still value freedom as much as ever.",
        "nodes": [{"label": "values freedom", "category": "values", "confidence": 0.6}],
        "raw_facts": [{"label": "values freedom", "category": "values", "confidence": 0.7, "source": "explicit"}],
        "raw_edges": [],
    },

    # 10. type-gate БЛОКИРУЕТ opposition в MERGE-ветке.
    #     Та же пара что #4 (sim 0.889, reversal, opposite polarity), но типы
    #     fact <-> value -> _types_can_conflict=False -> opposition подавлена ->
    #     reinforce вместо strong_contradict. Контраст с #4 доказывает работу гейта.
    {
        "name": "type_gate_blocks_merge_opposition",
        "desc": "MERGE+reversal+opposition, но типы fact/value -> гейт блокирует -> reinforce.",
        "user_text": "Actually, everything feels unstable in my career now.",
        "nodes": [{"label": "career stability", "category": "career", "confidence": 0.8, "node_type": "value"}],
        "raw_facts": [{"label": "career instability", "category": "career", "confidence": 0.6,
                       "source": "explicit", "type": "fact"}],
        "raw_edges": [],
    },

    # 11. performative (quit job) -> contradict career-домена.
    {
        "name": "performative_quit_job",
        "desc": "Performative 'quit my job' -> contradict узлов в затронутых доменах.",
        "user_text": "I quit my job today.",
        "nodes": [
            {"label": "corporate lawyer identity", "category": "career", "confidence": 0.7},
            {"label": "values long term security", "category": "values", "confidence": 0.6},
        ],
        "raw_facts": [],
        "raw_edges": [],
    },

    # 12. performative + новый факт.
    {
        "name": "performative_with_new_fact",
        "desc": "Performative + новый факт: новый узел добавлен, старый career-узел контрится.",
        "user_text": "I got divorced last month.",
        "nodes": [
            {"label": "happy marriage", "category": "relationships", "confidence": 0.7},
        ],
        "raw_facts": [{"label": "feeling free", "category": "mood", "confidence": 0.5, "source": "inferred"}],
        "raw_edges": [],
    },

    # 13. пустые raw_facts, не-performative текст -> пустая дельта.
    {
        "name": "empty_raw_facts_no_perf",
        "desc": "raw_facts=[] и нет performative -> дельта пустая, граф не тронут.",
        "user_text": "The weather is nice today.",
        "nodes": [{"label": "values freedom", "category": "values", "confidence": 0.6}],
        "raw_facts": [],
        "raw_edges": [],
    },

    # 14. raw_edges: contradicts между двумя новыми узлами.
    {
        "name": "edges_contradicts_new_new",
        "desc": "Два новых факта + ребро contradicts между ними.",
        "user_text": "I want to start a business but I'm terrified of going broke.",
        "nodes": [],
        "raw_facts": [
            {"label": "wants to start business", "category": "goals", "confidence": 0.7, "source": "explicit"},
            {"label": "fear of financial ruin", "category": "fears", "confidence": 0.7, "source": "explicit"},
        ],
        "raw_edges": [{"source": "wants to start business", "relation": "contradicts",
                       "target": "fear of financial ruin", "confidence": 0.6}],
    },

    # 15. raw_edges: blocks между новым и существующим узлом.
    {
        "name": "edges_blocks_new_existing",
        "desc": "Новый факт + ребро blocks к существующему узлу (резолв по label).",
        "user_text": "My procrastination is blocking my career growth.",
        "nodes": [{"label": "career growth", "category": "career", "confidence": 0.6}],
        "raw_facts": [{"label": "chronic procrastination", "category": "fears", "confidence": 0.6, "source": "explicit"}],
        "raw_edges": [{"source": "chronic procrastination", "relation": "blocks",
                       "target": "career growth", "confidence": 0.6}],
    },

    # 16. raw_edges: conflicts_with + невалидное ребро + ребро в несуществующий узел.
    #     Должно остаться только валидное conflicts_with.
    {
        "name": "edges_conflicts_plus_invalid",
        "desc": "conflicts_with валидно; relation вне VALID_RELATIONS и ребро в неизвестный label отброшены.",
        "user_text": "I want freedom but also crave stability.",
        "nodes": [],
        "raw_facts": [
            {"label": "desire for freedom", "category": "values", "confidence": 0.7, "source": "explicit"},
            {"label": "desire for stability", "category": "values", "confidence": 0.7, "source": "explicit"},
        ],
        "raw_edges": [
            {"source": "desire for freedom", "relation": "conflicts_with", "target": "desire for stability"},
            {"source": "desire for freedom", "relation": "enables", "target": "desire for stability"},
            {"source": "desire for freedom", "relation": "causes", "target": "nonexistent node"},
        ],
    },

    # 17. невалидная категория отфильтрована, валидный факт остаётся.
    {
        "name": "invalid_category_filtered",
        "desc": "Факт с категорией вне VALID_CATEGORIES пропускается; валидный создаётся.",
        "user_text": "I love hiking and I value adventure.",
        "nodes": [],
        "raw_facts": [
            {"label": "loves hiking", "category": "hobbies", "confidence": 0.7, "source": "explicit"},
            {"label": "values adventure", "category": "values", "confidence": 0.7, "source": "explicit"},
        ],
        "raw_edges": [],
    },

    # 18. несколько фактов в одном ходу: MERGE + NONE одновременно.
    {
        "name": "mixed_merge_and_new",
        "desc": "Два факта: один merge'ится в существующий, второй — новый.",
        "user_text": "I still value freedom, and now I also want career growth.",
        "nodes": [{"label": "values freedom", "category": "values", "confidence": 0.6}],
        "raw_facts": [
            {"label": "values freedom", "category": "values", "confidence": 0.7, "source": "explicit"},
            {"label": "wants career growth", "category": "goals", "confidence": 0.7, "source": "explicit"},
        ],
        "raw_edges": [],
    },

    # 19. MERGE same-category на пороге 0.72 (нижний порог внутри категории).
    #     'career change'(career) vs 'career transition'(career), sim 0.834 -> merge внутри
    #     категории -> reinforce. (Демонстрирует pass-1 same-category merge.)
    {
        "name": "same_category_merge_lower_threshold",
        "desc": "Та же категория: пара 0.834 -> merge через pass-1 -> reinforce.",
        "user_text": "I'm going through a career transition.",
        "nodes": [{"label": "career change", "category": "career", "confidence": 0.6}],
        "raw_facts": [{"label": "career transition", "category": "career", "confidence": 0.7, "source": "explicit"}],
        "raw_edges": [],
    },

    # 20. performative + reversal вместе на одном тексте.
    #     'quit' (performative career-домен) + 'Actually' (reversal) + opposition в MERGE.
    {
        "name": "performative_and_reversal_combined",
        "desc": "Текст триггерит и performative, и reversal; MERGE-оппозиция -> contradict + perf-sweep.",
        "user_text": "Actually, I quit my job — my career stability is gone.",
        "nodes": [
            {"label": "career stability", "category": "career", "confidence": 0.6},
            {"label": "values prestige", "category": "values", "confidence": 0.6},
        ],
        "raw_facts": [{"label": "career instability", "category": "career", "confidence": 0.6, "source": "explicit"}],
        "raw_edges": [],
    },
]


def generate():
    out = {}
    for fx in FIXTURES:
        tbg, ordered_ids = build_tbg(fx["nodes"])
        delta = _sdl.resolve(
            fx["raw_facts"],
            fx["user_text"],
            tbg,
            fx["raw_edges"],
        )
        out[fx["name"]] = {
            "desc": fx["desc"],
            "result": serialize_delta(delta, tbg, ordered_ids),
        }
    return out


def _dump(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)


def main():
    check = "--check" in sys.argv[1:]
    current = generate()

    if check:
        if not os.path.exists(SNAPSHOT_PATH):
            print(f"[FAIL] Эталон не найден: {SNAPSHOT_PATH}")
            print("       Сначала сгенерируй: py -3 tools/snapshot_sdl.py")
            return 1
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            golden = json.load(f)

        cur_s = _dump(current)
        gold_s = _dump(golden)
        if cur_s == gold_s:
            print(f"[OK] Снимок совпадает с эталоном ({len(current)} фикстур). exit 0")
            return 0

        # Печатаем per-fixture diff.
        print("[FAIL] Снимок РАСХОДИТСЯ с эталоном:\n")
        names = sorted(set(current) | set(golden))
        for name in names:
            c = current.get(name)
            g = golden.get(name)
            if c is None:
                print(f"  - {name}: отсутствует в текущем прогоне (была в эталоне)")
                continue
            if g is None:
                print(f"  + {name}: новая фикстура (нет в эталоне)")
                continue
            cr = _dump(c.get("result"))
            gr = _dump(g.get("result"))
            if cr != gr:
                print(f"  ~ {name}: расхождение")
                print(f"      эталон : {_dump(g.get('result'))}".replace("\n", "\n      "))
                print(f"      текущий: {_dump(c.get('result'))}".replace("\n", "\n      "))
        return 1

    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        f.write(_dump(current))
        f.write("\n")
    print(f"[OK] Эталон записан: {SNAPSHOT_PATH} ({len(current)} фикстур)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
