# Отчёт — Ballast · Фаза 0 · Шаг 2: чистка (strip + delete)

Дата: 2026-06-22. Провайдер прогонов: `cerebras / gpt-oss-120b`, `temp=0`, `max_tokens=5000`.
Тест `tbg_realtest.py` **не модифицирован**. Запуск под `PYTHONIOENCODING=utf-8`
(консоль Windows cp1251 не печатает рамки `─`; это окружение, не код теста).

## Итог: все 4 под-шага зелёные (realtest EXIT=0)

| Под-шаг | Узлы (msg 1→5) | Конфликты (msg 1→5) | avg_conf | EXIT |
|---|---|---|---|---|
| baseline | 3·4·6·8·12 | 1·1·1·1·1 | 69→83% | 0 |
| 2a axis_state | 3·4·6·8·12 | 1·1·1·1·1 | 69→83% | 0 |
| 2b SLIM extractor | 3·4·7·9·13 | 1·1·1·2·2 | 69→78% | 0 |
| 2c delete files | 3·4·7·9·12 | 1·1·1·1·1 | 69→83% | 0 |
| 2d translate | 3·4·6·8·12 | 1·1·1·2·2 | 69→77% | 0 |

Дрейф убеждений сохранён на каждом шаге (узлы растут, конфликты есть, insight
непустой, decay работает). Вариация цифр между 2b–2d — недетерминизм reasoning-модели
gpt-oss при temp=0 (LLM, не код); приёмочный критерий качественный.

## 2a — выпил axis_state

`tbg_engine.py`:
- удалён `from tbg_axes import get_belief_axes`;
- `load()` — убраны распаковка `__axis_state__` и реконструкция `UserAxisState` (`from user_state import …`);
- `save()` — убрана упаковка `__axis_state__`;
- `apply_delta()` — убран вызов `self._update_axis_state(tbg)`;
- удалён метод `_update_axis_state` (~62 строки);
- `_update_node()` — убран блок `node.axis_projection = get_belief_axes().project(...)`;
- удалён осиротевший `import statistics` (использовался только в `_update_axis_state`).

`tbg_schema.py`:
- удалён `from user_state import UserAxisState` и поле `axis_state` в `UserTBG`.

**Сохранено** (по ТЗ): `amf_filter` и `_amf_state` в движке — не тронуты.
2a дал **байт-в-байт** ту же динамику, что baseline (поведенчески нейтрально).

## 2b — SLIM extractor (`tbg_extractor.py`, 74 КБ → 38 КБ)

Удалено:
- `import her_resolver` + HER-routing / opposition-gate блок в `resolve()`;
- `from fact_engine import get_embed_model` + `_embed_labels` + `_check_semantic_similarity` (cosine-dedup);
- cosine-ветка в `_lookup_or_register` (оставлен детерминированный slug-mint + alias-кэш);
- cosine-«already covered» в `_ensure_identity_fallback` → заменён на точный label-guard;
- SDL-оппозиция: `_is_opposition` (polarity+EPA), `_get_polarity`, `_detect_reversal`,
  oscillation-рёбра, MERGE/FLAG-ветки `resolve()`;
- `import tbg_nli` + NLI-синглтон + NLI-скан в `resolve()`;
- EPA-проекция: вызовы `from tbg_axes import get_belief_axes`, `_eval_prod`;
- `DeterministicExtractor` (EPA-путь извлечения) + флаг `_USE_DETERMINISTIC`/`_det_extractor`
  + детерминированная ветка в `extract_tbg_delta`;
- `_embed_for_polarity`, `_node_polarity_vs_action`;
- мёртвые константы/флаги: `_NEGATIVE/_POSITIVE_POLARITY`, `_REVERSAL_PATTERNS`,
  `_EPA_AXES/_EPA_OPP_THRESHOLD`, `_STRONG_CONTRADICT_THRESHOLD`, `DEDUP_*`,
  `CONCEPT_REGISTRY_THRESHOLD`, `_SIM_*`, `_CONFLICT_FORBIDDEN`/`_types_can_conflict`,
  `_LAST_OPP_REASON`, `_FIX_FLAG_CONCEPT_ID`, `_FIX_PERFORMATIVE_POLARITY`,
  `_HER_ROUTING`, `_OPPOSITION_GATE`, `import numpy as np`.

`tbg_engine.py`: удалена `_USE_DET`-ветка `update_tbg_background` (импортировала `_det_extractor`).

Оставлено (то, что потребляет `apply_delta`): LLM-извлечение belief-дельт →
`SemanticDecisionLayer.resolve` (создание узлов, concept_id через registry/slug,
performative-sweep на regex, `_resolve_edges` для LLM-рёбер), `_ensure_identity_fallback`
(regex), `validate_delta`, `_canonicalize_labels` (чистый LLM, флаг off),
`query_graph_semantic` (чистый LLM, без embed-модели), JSON-хелперы, промпты.

### Поведенческая заметка (важно для приёмки)
ТЗ называет `fact_engine` «извлечением фактов», но в экстракторе он импортировался
только ради embed-модели (`get_embed_model`), питавшей **cosine-dedup**. Удаление
`fact_engine` (по цели 2c — ноль ссылок) убирает семантический merge/flag. Следствие:
- exact-label повторы по-прежнему схлопываются — это делает движок (`_update_node`
  дедупит по label/concept_id), поэтому `reinforce` происходит на уровне движка;
- теряются только парафраз-merge (разный label, один концепт) → иногда на 1 узел больше
  (13 vs 12 в 2b). Для критерия «узлы растут, конфликты есть» — приемлемо;
- `delta.reinforce_ids` теперь всегда 0 (SDL больше не эмитит reinforce); поле в схеме
  сохранено.

## 2c — удаление dead-файлов

Удалены (предварительно подтверждён **ноль** ссылок из остающихся файлов):
`user_state.py`, `tbg_axes.py`, `tbg_nli.py`, `fact_engine.py`, `her_resolver.py`
(+ их `__pycache__`). После удаления `grep` по `ballast` не находит импортов этих модулей.

## 2d — перевод комментов в English

Переведены кириллические **комментарии** (логика не менялась):
- `tbg_schema.py`: «Шаг 4 …» → «Step 4 …»;
- `tbg_engine.py`: два коммента «Шаг 4 closed-vocab cache»;
- `tbg_extractor.py`: «Шаг 4 …», блок «Шаг 2 / Шаг 4 / Stage 1 / Stage 2 / База …».

Заодно поправлены устаревшие англоязычные комментарии в `tbg_schema.py`, ссылавшиеся
на удалённый код (`signal_span`, `axis_projection`, `domain` — помечены legacy).

### Функциональные кириллические строки (на решение) — НЕ менялись
- **`tbg_realtest.py`** (по правилу «тест не модифицировать»): диалог `CONVERSATION`
  (русские сообщения — это контент теста), все UI-строки `print` («Статистика:»,
  «КОНФЛИКТЫ:», «ГРАФ после сообщения», «ИТОГ — динамика графа», лейблы DP/risk и т.п.),
  и сравнения вида `'действию' in traj` / `'открыто' in window` (логика интерпретатора
  завязана на русские подстроки). Перевод сломал бы тест и/или его логику — оставлено как есть.
- В **остальных 8 файлах** функциональных кириллических строк нет (только комменты, переведены).

## Итоговый состав `ballast` (ядро)

`amf_filter.py`, `dissonance_engine.py`, `intervention_engine.py`, `llm_client.py`,
`mode_engine.py`, `tbg_engine.py` (1095 стр.), `tbg_extractor.py` (921 стр.),
`tbg_schema.py` (270 стр.), `tbg_realtest.py`, `requirements.txt`.

Вне ядра / вне scope Шага 2 (появились в папке во время сессии, вероятно через OneDrive;
удалённых модулей **не** импортируют): `ballast_live_benchmark.py`, `ballast_pressure_test.py`.

Лог-файлы прогонов: `baseline.log`, `step2a.log`, `step2b.log`, `step2c.log`, `step2d.log`.

## Стоп после Шага 2
Реорганизация в `core/` + `governance/` — отдельный шаг. Дальше — проектирование Фазы 1 (policy + гейт).
