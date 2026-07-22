# Отчёт — Ballast · op/ref + two-sided anchor-sanity

Дата: 2026-06-22. Провайдер: `cerebras / gpt-oss-120b`, `temp=0`, `LLM_MAX_TOKENS=5000`, MockDB.
Прогоны под `PYTHONIOENCODING=utf-8` (консоль Windows). Весь новый код — English.

## Что сделано
- **Часть A (op/ref):** правки только в `tbg_extractor.py` + новый промпт-шаблон.
- **Часть B:** новый `anchor_sanity.py` (English).
- **Движок `tbg_engine.py` НЕ изменён** (только читался для проверки механики reinforce).

## Приёмка
| Критерий | Статус |
|---|---|
| `tbg_realtest.py` зелёный при `TBG_OPREF=0` (байт-идентично, тест не тронут) | ✅ EXIT=0, дрейф 3·4·6·8·12 (как Шаг 2d) |
| `tbg_realtest.py` зелёный при `TBG_OPREF=1` (извлечение не сломано) | ✅ EXIT=0, 3·5·8·11·15 |
| `anchor_sanity.py` отрабатывает, печатает B(a)+B(b)+сравнение ON/OFF | ✅ EXIT=0 |
| Движок не изменён | ✅ diff только `tbg_extractor.py` + `anchor_sanity.py` |

## Часть A — реализация op/ref (флаг `TBG_OPREF`, default OFF)
- Флаг читается **динамически** (`_opref_enabled()` из env), а не как модульная константа —
  чтобы `anchor_sanity` сравнивал ON/OFF в одном процессе. При OFF — байт-идентично.
- **A1.** Отдельный шаблон `EXTRACTION_PROMPT_OPREF` (тот же 1 LLM-вызов). Кандидаты:
  до 50 `tbg.active_nodes(0.3)` по убыванию confidence, формат `- "<label>" [<concept_id>]`
  (`_build_opref_block`). Инструкция: тег `"ref"` = дословный label из списка, либо отсутствует;
  выдумывать ref вне списка запрещено.
- **A2.** В `resolve()` перед `_lookup_or_register`: при ON строится `opref_index`
  (`label.lower().strip() → concept_id`) **только из показанных кандидатов**. Если у факта есть
  `ref` и он точно (lower+strip) совпал с кандидатом → берём concept_id существующего концепта →
  `_make_node` создаёт узел с этим concept_id → downstream `_update_node` находит существующий
  по concept_id и копит якорь. Не совпал / нет ref / OFF → текущий путь (mint slug). Без fuzzy.
- **A3.** При OFF: `_opref_enabled()` False → OPREF-промпт не строится, `opref_index` пуст,
  `ref` не читается (`ref = raw.get("ref") if opref_index else None`) → байт-идентично (доказано realtest).

## Часть B — anchor_sanity (структурные инварианты, без хардкода)

### B(a) UNIFY + ACCUMULATE — PASS
Один концепт «values stability» 5 парафразами (разные surface-лейблы), вперемешку с 2 шумовыми
ходами + явный контр-ход в конце.
- ON-якорь (по макс. `len(confidence_history)`): **`values stability`**, `history_len=4` —
  **4 из 5 парафразов схлопнулись в ОДИН концепт** через op/ref;
- confidence по ходам: `[0.85×7, 0.377]` — **упал на контр-ходе** (0.85 → 0.377);
- пережил шумовые ходы (не исчез).

### B(b) NO FALSE MERGE — PASS (важнейшее)
Два похожих, но разных убеждения (financial security vs career security), по 2 хода.
- Остались **4 раздельных** активных concept-узла (financial и career НЕ слиты). Ложного merge нет.

### Сравнение ON/OFF (INFO, в выводе теста)
- ON: 6 узлов, anchor_hist=4. OFF: 2 узла, anchor_hist=5.

## Ключевая находка (честно, для решения по policy)
1. **`pos_evidence` и `confidence` НЕ растут при reinforce** в текущем (post-SLIM) пайплайне:
   `EVIDENCE_WEIGHTS["neutral"]=0.0` (engine:82-88), а экстрактор не выставляет `evidence_type`
   → вес 0 → reinforce двигает только `confidence_history`. Это свойство движка (не трогаем).
   Поэтому единственный валидный сигнал накопления — **длина `confidence_history`**; на
   `pos_evidence`/«рост confidence» **не грейдим** (это мерило неиспользуемого поля, а не op/ref).
   confidence движется только вниз — через contradict (контр-ход это и показал).
2. **OFF тоже юнифицирует** — стандартный `EXTRACTION_PROMPT` содержит инструкцию label-echo
   (EXISTING BELIEFS → копировать существующий label), и LLM схлопнул парафразы в «stability»
   (hist=5). Поэтому **node-count ON<OFF — не чистый дискриминатор** op/ref.
   Выигрыш op/ref не в числе узлов, а в **детерминизме на уровне concept_id** (закрытый словарь,
   reuse по точному ref даже при дрейфе лейбла) + **анти-merge гарантии** (B(b)).

## Вывод
**ЯКОРЬ ПРИГОДЕН** — (a) формируется и накапливается, И (b) ложного merge нет.
Движок+экстракция дают пригодный якорь → идём в **policy + гейт (Фаза 1)**.

## Файлы прогонов
`opref_off.log`, `opref_on.log`, `anchor_sanity.log`.

## Оговорка на будущее (не блокер)
Если для policy понадобится, чтобы `confidence`/`pos_evidence` действительно росли на повторных
подтверждениях (а не только история), это требует, чтобы экстракция выставляла `evidence_type`
(`medium_pos`/`strong_pos`) — это правка экстрактора + промпта, отдельный шаг (движок уже умеет).
