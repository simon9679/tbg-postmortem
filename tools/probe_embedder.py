#!/usr/bin/env python3
"""
E3a — оффлайн-проба эмбеддера: MiniLM vs multilingual-e5-small.

ЦЕЛЬ
----
До свапа эмбеддера проверить, ЛУЧШЕ ли e5-small разделяет классы концептов
(same / related / different), и вывести пороги MERGE/FLAG под e5.
Решение «свапать или нет» принимается по этой пробе. Прод не трогаем.

ЧТО СЧИТАЕМ
----------
Для каждой пары — косинус на ОБЕИХ моделях (эмбеддинги normalize=True, косинус=dot).
e5 — симметричная STS-задача: ОБА лейбла префиксятся "query: " (так требует e5).
MiniLM — без префикса.

Классы (размечены по смыслу в calib_pairs.json):
  same      -> должны быть ВЫШЕ порога MERGE (один концепт)
  related   -> между FLAG и MERGE (тот же домен, разные концепты)
  different -> НИЖЕ порога FLAG (несвязанные концепты)

Разделимость:
  margin1 = min(same)    - max(related)    # запас на границе MERGE
  margin2 = min(related) - max(different)  # запас на границе FLAG
Положительный margin = класс чисто отделяется одним порогом. Больше = лучше.

Пороги под e5:
  MERGE_e5 = середина между max(related) и min(same)
  FLAG_e5  = середина между max(different) и min(related)
(если margin отрицательный — порог попадает в зону перекрытия, печатаем предупреждение)

ЗАПУСК:
  py -3 tools/probe_embedder.py
"""
import os
import sys
import json

import numpy as np

PAIRS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_pairs.json")

MINILM_NAME = "all-MiniLM-L6-v2"
E5_NAME = "intfloat/multilingual-e5-small"


def load_pairs():
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {cls: data[cls] for cls in ("same", "related", "different")}


def cosines_for_model(model, pairs_by_class, e5_prefix=False):
    """Возвращает {класс: [косинусы пар]} для одной модели."""

    def emb(text):
        t = f"query: {text}" if e5_prefix else text
        return model.encode(t, normalize_embeddings=True, show_progress_bar=False)

    out = {}
    for cls, pairs in pairs_by_class.items():
        sims = []
        for a, b in pairs:
            ea, eb = emb(a), emb(b)
            sims.append(float(np.dot(ea, eb)))
        out[cls] = sims
    return out


def stats(sims):
    arr = np.array(sims, dtype=float)
    return float(arr.min()), float(arr.mean()), float(arr.max())


def print_model_table(title, cos_by_class):
    print(f"\n=== {title} ===")
    print(f"{'класс':<10} {'n':>3} {'min':>7} {'mean':>7} {'max':>7}")
    print("-" * 40)
    s = {}
    for cls in ("same", "related", "different"):
        mn, me, mx = stats(cos_by_class[cls])
        s[cls] = (mn, me, mx)
        print(f"{cls:<10} {len(cos_by_class[cls]):>3} {mn:>7.3f} {me:>7.3f} {mx:>7.3f}")

    margin1 = s["same"][0] - s["related"][2]      # min(same) - max(related)
    margin2 = s["related"][0] - s["different"][2]  # min(related) - max(different)
    print(f"\nmargin1 (граница MERGE, min(same)-max(related))    = {margin1:+.3f}"
          f"  {'OK разделяет' if margin1 > 0 else 'ПЕРЕКРЫТИЕ классов'}")
    print(f"margin2 (граница FLAG,  min(related)-max(different)) = {margin2:+.3f}"
          f"  {'OK разделяет' if margin2 > 0 else 'ПЕРЕКРЫТИЕ классов'}")
    return s, margin1, margin2


def print_pairs_detail(pairs_by_class, mini, e5):
    """Построчно: пара, косинус MiniLM, косинус e5 — для ручной проверки."""
    print("\n=== ПОПАРНО (MiniLM | e5) ===")
    print(f"{'класс':<10} {'mini':>6} {'e5':>6}  пара")
    print("-" * 70)
    for cls in ("same", "related", "different"):
        for i, (a, b) in enumerate(pairs_by_class[cls]):
            print(f"{cls:<10} {mini[cls][i]:>6.3f} {e5[cls][i]:>6.3f}  {a!r} <-> {b!r}")


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Нет sentence-transformers. Установи: pip install sentence-transformers")
        return 1

    pairs = load_pairs()
    n_total = sum(len(v) for v in pairs.values())
    print(f"Загружено пар: {n_total} "
          f"(same={len(pairs['same'])}, related={len(pairs['related'])}, different={len(pairs['different'])})")

    print(f"\nЗагрузка модели 1/2: {MINILM_NAME} ...")
    m_mini = SentenceTransformer(MINILM_NAME)
    print(f"Загрузка модели 2/2: {E5_NAME} (может скачиваться с HF ~120МБ) ...")
    m_e5 = SentenceTransformer(E5_NAME)

    mini = cosines_for_model(m_mini, pairs, e5_prefix=False)
    e5 = cosines_for_model(m_e5, pairs, e5_prefix=True)

    print_pairs_detail(pairs, mini, e5)

    print_model_table(f"MiniLM ({MINILM_NAME})", mini)
    s_e5, m1_e5, m2_e5 = print_model_table(f"e5-small ({E5_NAME}, prefix='query: ')", e5)

    # Пороги под e5: середины зон между соседними классами.
    merge_e5 = (s_e5["related"][2] + s_e5["same"][0]) / 2.0     # между max(related) и min(same)
    flag_e5 = (s_e5["different"][2] + s_e5["related"][0]) / 2.0  # между max(different) и min(related)

    print("\n=== ПРЕДЛОЖЕННЫЕ ПОРОГИ ДЛЯ e5 ===")
    print(f"MERGE_e5 = {merge_e5:.3f}   (середина между max(related)={s_e5['related'][2]:.3f} "
          f"и min(same)={s_e5['same'][0]:.3f})")
    print(f"FLAG_e5  = {flag_e5:.3f}   (середина между max(different)={s_e5['different'][2]:.3f} "
          f"и min(related)={s_e5['related'][0]:.3f})")
    if m1_e5 <= 0:
        print("  ВНИМАНИЕ: margin1<=0 — same и related перекрываются, MERGE_e5 не чистый.")
    if m2_e5 <= 0:
        print("  ВНИМАНИЕ: margin2<=0 — related и different перекрываются, FLAG_e5 не чистый.")

    print("\n(Текущие пороги MiniLM в проде: MERGE=0.82, FLAG=0.72.)")
    print("Прод-код не трогался. Это только оффлайн-замер.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
