# Отчёт — Ballast · evidence_type (последний фикс перед policy)

Дата: 2026-06-22. Провайдер: `cerebras / gpt-oss-120b`, `temp=0`, `LLM_MAX_TOKENS=5000`, MockDB.
Прогоны под `PYTHONIOENCODING=utf-8`. Весь новый код — English.

## Что сделано
- **Часть A:** `evidence_type` в экстракторе под флагом `TBG_EVIDENCE_TYPE` (default OFF) —
  правки только в `tbg_extractor.py` + инъекция clause в промпт.
- **Часть B:** обновлён `anchor_sanity.py` (вернул валидный инвариант + детерминированный A3).
- **Движок `tbg_engine.py` НЕ изменён** (diff=0; только читался — он уже потребляет `evidence_type`
  через `EVIDENCE_WEIGHTS` в `_update_node`).

## Приёмка
| Критерий | Статус |
|---|---|
| `tbg_realtest.py` при `TBG_EVIDENCE_TYPE=0` — байт-идентично | ✅ зелёный + детерминированное доказательство (см. ниже) |
| `tbg_realtest.py` при `=1` — извлечение не сломано | ✅ EXIT=0 |
| `anchor_sanity.py` (OPREF=1, EVIDENCE_TYPE=1): B(a) растёт+копит+падает, OFF→ON плоский→растущий; A3; B(b) | ✅ все PASS, EXIT=0 |
| Движок не изменён | ✅ diff только `tbg_extractor.py` + промпт + `anchor_sanity.py` |

## Часть A — реализация
- **Флаг динамический** `_evidence_type_enabled()` (env `TBG_EVIDENCE_TYPE`) — чтобы один процесс
  сравнивал ON/OFF.
- **A1 (промпт).** При ON на уже построенный промпт (любой из базовых шаблонов, включая op/ref)
  инъектится `EVIDENCE_TYPE_CLAUSE` + поле `"evidence_type"` в JSON-схему facts —
  `_inject_evidence_type_clause()` через replace по стабильным якорям
  (`"source": "explicit|inferred",` и `Return ONLY valid JSON…`). Closed-set
  {strong_pos, medium_pos, medium_neg, strong_neg}, дефолт medium_pos, strong_* — редко.
- **A2 (парс в `resolve()`).** При ON: `et = raw.get("evidence_type")`; валид против
  `VALID_EVIDENCE_TYPES`; невалид/нет → `medium_pos`; передаётся в `_make_node(evidence_type=…)`.
  При OFF → `None` (текущее).
- **A3 (анти-двойной-учёт).** После performative-sweep: для add-узла, чей `concept_id` совпал с
  концептом, попавшим в `contradict_ids`, и чей `evidence_type` негативный (`*_neg`) — обнуляем
  `evidence_type`. Одно сомнение не идёт двумя каналами; контр-канал (contradict) оставляем.
  Позитивный `evidence_type` на контрадикченном концепте НЕ трогаем (движок мирит через ambivalence).

### Доказательство OFF-байт-идентичности (детерминированно, без LLM-шума)
`resolve()` с фиксированным фактом:
- OFF (`TBG_EVIDENCE_TYPE` unset) → `node.evidence_type is None`;
- ON → `strong_pos` (как в факте); невалид → `medium_pos`.
Промпт-инъекция при OFF не вызывается (`'evidence_type' not in prompt`). Т.е. при OFF путь
идентичен прежнему. Численные строки realtest варьируются **между прогонами** из-за
недетерминизма cerebras (reasoning при temp=0), а не из-за кода — поэтому идентичность доказана
на уровне кода, а не сравнением выводов realtest.

## Часть B — anchor_sanity (структурно, без хардкода)

### B(a) UNIFY + ACCUMULATE — PASS
Один концепт «values stability» 5 парафразами + 2 шума + контр-ход.
| | ON (OPREF=1, ET=1) | OFF (ET=0) |
|---|---|---|
| anchor confidence/ход | `0.85, 0.85, 0.92, 0.92, 0.92, 0.92, 0.92, 0.494` | `0.85×7, 0.401` |
| anchor pos_evidence/ход | `0, 0, 2.2, 3.05, 3.05, 3.05, 3.05, 3.05` | `0×8` |
- ON: confidence РОСЛА (0.85→0.92, ambiv-cap), pos_evidence НАКОПИЛАСЬ (→3.05), упала на контр-ходе;
- OFF: confidence ПЛОСКАЯ 0.85, pos_evidence 0 — доказывает, что именно фикс усиливает якорь.

### A3 NO DOUBLE COUNT — PASS (детерминированно)
Performative-событие («I quit my job») сметает существующий career-концепт в `contradict_ids`;
факт того же концепта с `strong_neg` → его `evidence_type` обнулён. Двойного бития нет.
Контроль величины на контр-ходе B(a): 0.92→0.494 ≈ один `strong_neg` (оценка ~0.56), не удвоено.

### B(b) NO FALSE MERGE — PASS
financial security vs career security → 4 раздельных активных concept-узла. Ложного merge нет.

## Вывод
**ЯКОРЬ НАСТОЯЩИЙ** — набирает силу на подтверждениях (confidence↑, pos_evidence↑), падает на
сомнении, дубли не плодятся (A3), разное не сливается (B(b)). → Идём в **policy + pressure-гейт (Фаза 1)**.

## Файлы прогонов
`et_off.log`, `et_on.log`, `anchor_sanity.log`.
