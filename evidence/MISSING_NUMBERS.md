# MISSING_NUMBERS.md
# Провенанс-приложение к BALLAST_FULL_STORY.md
# Числа, у которых нет on-disk артефакта — только из архива чатов.
# Структура: claim / number / source / on-disk artifact / how measured
# Статус: chat-sourced, не перепроверены на диске. Пометить в публикации.

---

## Кластер 1 — E3: антонимная стена (концепт-тождество)

**1.1 Катастрофическая пара**
- claim: cosine-сходство смежных концептов выше, чем реальных синонимов
- number: `career security` vs `financial security` — косинус **0.574**
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: sentence embedding (all-MiniLM или аналог), cosine similarity между двумя строками

**1.2 Нижняя граница синонимов**
- claim: реальный синоним в том же тесте давал ниже, чем смежная пара
- number: слабейший синоним в тесте — косинус **0.146**
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: тот же тест, та же embedding-модель; контраст с 0.574 показывает: нет порога

**1.3 NLI false-contradiction на смежных парах**
- claim: NLI как полярный гейт даёт false-contradiction на adjacent-парах
- number: **8/8** (все 8 смежных пар → false-contradiction)
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: cross-encoder NLI на 8 предзафиксированных adjacent-парах; планки зафиксированы до прогона, точные threshold-значения в архиве чата

**1.4 Перепробованные методы**
- claim: все методы провалили pre-committed bars
- number: n/a (перечень методов, не число)
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: лучшие эмбеддеры, mean-centering, whitening, ColBERT/MaxSim, NLI cross-encoders, STS cross-encoders — каждый с заранее зафиксированной планкой

---

## Кластер 2 — Scrooge: «весь интеллект арендован у LLM»

**2.1 Sonnet-прогон: belief-trajectory**
- claim: на claude-sonnet-4-6 движок поймал всю дугу Скруджа
- number: `Christmas is a fraud` **92%→54%→35%**; `social responsibility is not my concern` **91%→6%**; `commits to year-round joy` **→92%**; turning point на ходе 7; **19 связей** (vs 5 на gpt-oss)
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none (на диске только headline «85% edges from LLM, 0 from Python»)
- how measured: прогон test_scrooge_drift.py на sonnet-4-6; confidence по ходам из вывода движка

**2.2 gpt-oss: два противоречивых прогона (живой анекдот про n=1)**
- claim: тот же провайдер дал противоположные результаты в разных прогонах
- number: прогон 1 — `Christmas is a fraud` **осталась 92%** (движок «слеп»); прогон 2 — та же реплика уронила до **46%**
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21) — оба прогона там
- on-disk artifact: none
- how measured: test_scrooge_drift.py на gpt-oss-120b (cerebras), два независимых запуска без заморозки экстракции

**2.3 Контроль на узнавание (doppelganger)**
- claim: движок воспроизвёл арк на non-Dickens двойнике → не пересказ
- number: идентичная дуга (качественно, без точных confidence-чисел в памяти)
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: disguised-twin variant: тот же нарратив, другая лексика (не из Диккенса); движок поймал арк → подтверждает генерализацию, не меморизацию

**2.4 Attribution: 85% drops от LLM-рёбер**
- claim: вся семантика — от LLM, Python-машинерия дала 0 рёбер
- number: **85%** belief-drops traced to LLM-emitted contradiction edges; SDL/NLI/HER opposition machinery — **0 edges** (оба варианта теста)
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21); summary подтверждён в «ласт тбг»
- on-disk artifact: headline «85/0» на диске есть (tbg_post_ru_full.md §11.2), сырые числа — только чат
- how measured: frozen-extraction provenance-анализ: attribution каждого belief-drop к источнику (LLM-edge vs Python-machinery); de Finetti нормализация — несущий механизм

---

## Кластер 3 — Closed-set стабильность

**3.1 HER domain-routing within-run**
- claim: LLM-классификация в закрытый набор доменов стабильна внутри прогона
- number: **98.9%** within-run consistency
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21) + «ласт тбг»
- on-disk artifact: none
- how measured: HER domain-routing на N концептов, frozen extraction replay; доля одинаковых domain-assignments между прогонами

**3.2 HER domain-routing cross-provider**
- claim: та же стабильность держится при смене провайдера
- number: **95.8%** cross-provider (между двумя разными LLM-провайдерами)
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: сравнение domain-assignments на одном входе: провайдер A vs провайдер B

**3.3 Language-invariance**
- claim: domain-routing работает одинаково на разных языках
- number: «духовный кризис» и «spiritual crisis» → один домен, при косинусе между ними **0.005**
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: ручная проверка пары на одной конфигурации

**3.4 HER как гейт убеждений — OCS регрессия**
- claim: стабильный routing как belief-tracking гейт вредит когнитивным метрикам
- number: OCS **−0.128** при включённом HER как гейте (frozen extraction)
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: frozen-extraction A/B: HER-гейт ON vs OFF; разница в OCS = −0.128

**3.5 «Классификация>>генерация» при temp=0 — ничья**
- claim: широкое утверждение «классификация стабильнее генерации» ложно при temp=0
- number: **ничья** (tie) — нет измеримой разницы между stability of classification vs free generation при temperature=0
- source: чат «Изучение структуры и кода ТБГ» (2026-06-21)
- on-disk artifact: none
- how measured: прямой head-to-head: попросить LLM (a) выбрать из списка, (b) сгенерировать free-form — сравнить consistency при temp=0

---

## Кластер 4 — DriftBench хронология (метрик-разработка)
# ВАЖНО: это отдельный эксперимент от Кластера TBG vs bare LLM.
# Модель: gemini-3-flash-preview. Вопрос: куда двигались метрики системы.
# НЕ смешивать с head-to-head OCS 0.17 vs 0.04 (sonnet-4-6, другой вопрос).

**4.1 Стартовые метрики**
- claim: начальные числа до улучшений
- number: BDA=0.37, CDR=0.52, ISS=0.62, OCS=0.00, TPS=0.61, Overall=**0.496**
- source: чат «Temporal belief graph проект» (2026-06-17)
- on-disk artifact: none (только итоговые в чате «ласт тбг»)
- how measured: DriftBench evaluate.py на gemini-3-flash-preview, 1 прогон

**4.2 NLI-серия OCS (4 прогона)**
- claim: OCS бимодальна и нестабильна
- number: 0.000 / 0.464 / 0.000 / 0.056 → mean **0.173**, std≈0.25; ниже цели 0.40
- source: чат «Temporal belief graph проект» (2026-06-17) + «ласт тбг»
- on-disk artifact: none
- how measured: 4 независимых прогона evaluate.py, TBG_NLI_ENABLED=1; биомодальность = Gemini extraction stochasticity

**4.3 NLI-диагноз: apply_delta переукорачивает**
- claim: корень OCS-нестабильности — не NLI, а apply_delta
- number: нода найдена с sim=**0.58**, confidence упала 0.50→**0.13** (переукорочено)
- source: чат «Temporal belief graph проект» (2026-06-17)
- on-disk artifact: none
- how measured: per-node diagnostic в NLI-прогоне; P4 выдал traceable ноду

**4.4 Финал v1 (3-run mean)**
- claim: итоговые числа после всех улучшений v1
- number: BDA=0.421±0.008, CDR=0.657±0.029, ISS=0.597±0.105, OCS=**0.549±0.323** (бимодален), TPS=0.745±0.019, Overall=**0.558** (+12% от старта)
- source: чат «Temporal belief graph проект» (2026-06-17) + «ласт тбг»
- on-disk artifact: none (только «ласт тбг» содержит эту таблицу из чата)
- how measured: 3 прогона evaluate.py, gemini-3-flash-preview, NLI=1

**4.5 Двухпромптовый экстрактор**
- claim: разделение на Prompt A (узлы) + Prompt B (конфликты) поднимает CDR, но роняет BDA
- number: CDR **0.462→0.733** (+58%), AS **0.518→0.980**; BDA **0.562→0.460** (регресс −18%); конфликтных рёбер 39–60% от всех
- source: чат «Ошибки дрифтбенча с новым экстрактором» (2026-06-26)
- on-disk artifact: none
- how measured: async two-prompt architecture (asyncio.gather); Prompt B overgenerated без threshold; диагноз: recall без precision

**4.6 Self-graded числа — отозваны**
- claim: BDA=1.0 на кастомном DriftBench не переносится на публичный
- number: кастомный: BDA=**1.0**; публичный run_drift_official: CER=**0**, BDA≈**0.2**
- source: чат «ласт тбг» + чат «Изучение структуры и кода ТБГ»
- on-disk artifact: run_drift_official выходы частично на диске
- how measured: кастомный — concept_id в бенче совпадали с concept_id TBG (утечка); публичный — без утечки, честный baseline

---

## Разделительная таблица DriftBench-двойни
# Для редактора черновика: эти два эксперимента нельзя смешивать.

| параметр | (a) TBG vs bare LLM | (b) метрик-разработка |
|---|---|---|
| модель | claude-sonnet-4-6 | gemini-3-flash-preview |
| вопрос | делает ли ядро что-то поверх голой модели? | куда двигались метрики системы? |
| OCS TBG | **0.17** | **0.549±0.323** (финал) |
| OCS baseline | bare LLM **0.04** | стартовый **0.00** |
| интерпретация | единственная зона превосходства на коротких диалогах | эволюция метрики, не доказательство продукта |
| on-disk | EVALUATION_REPORT §7 | none (только чат) |
| source-чат | «Изучение структуры и кода ТБГ» | «Temporal belief graph» + «Ошибки дрифтбенча» |
