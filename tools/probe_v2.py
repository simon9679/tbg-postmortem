#!/usr/bin/env python3
"""
E3a-2 — расширенная проба эмбеддера + ре-центрирование (анти-анизотропия).
БЕЗ правок прода. Только оффлайн-замер.

ИДЕЯ
----
v1 показал: e5 чинит recall у MERGE (парафразы перестают проваливаться), но
сжимает всю шкалу в [0.76, 0.96] — даже unrelated сидят на ~0.78. Это анизотропия
e5, и она ломает FLAG-полосу при АБСОЛЮТНЫХ порогах.

Разметка теперь ПО ТОПИКУ (calib_pairs_v2.json), 4 класса:
  paraphrase        — один концепт
  opposite_polarity — тот же топик, обратная полярность (для ЭМБЕДДЕРА = тот же
                      топик, должен скорить высоко; полярность ловит _is_opposition)
  adjacent          — смежный топик/ось, разные концепты
  unrelated         — несвязанные темы

Цель разделимости ТОПИЧЕСКАЯ:
  sep_MERGE = min(paraphrase ∪ opposite_polarity) − max(adjacent)
  sep_FLAG  = min(adjacent) − max(unrelated)

РЕ-ЦЕНТРИРОВАНИЕ:
  floor = mean(unrelated)  (порог выведен из «несвязанных», НЕ из бенча)
  sim'  = (cos − floor) / (1 − floor)
  Пересчитать sep на sim' и сравнить с raw. Вопрос: возвращает ли это e5 чистые
  (положительные) sep_MERGE и sep_FLAG?

ЗАПУСК:
  py -3 tools/probe_v2.py
"""
import os
import sys
import json

import numpy as np

PAIRS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_pairs_v2.json")

MINILM_NAME = "all-MiniLM-L6-v2"
E5_NAME = "intfloat/multilingual-e5-small"

CLASSES = ("paraphrase", "opposite_polarity", "adjacent", "unrelated")


def load_pairs():
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {cls: data[cls] for cls in CLASSES}


def cosines_for_model(model, pairs_by_class, e5_prefix=False):
    def emb(text):
        t = f"query: {text}" if e5_prefix else text
        return model.encode(t, normalize_embeddings=True, show_progress_bar=False)

    out = {}
    for cls, pairs in pairs_by_class.items():
        out[cls] = [float(np.dot(emb(a), emb(b))) for a, b in pairs]
    return out


def recenter(cos_by_class):
    """sim' = (cos - floor)/(1 - floor), floor = mean(unrelated)."""
    floor = float(np.mean(cos_by_class["unrelated"]))
    denom = 1.0 - floor
    out = {}
    for cls, sims in cos_by_class.items():
        out[cls] = [(c - floor) / denom for c in sims]
    return out, floor


def stats(sims):
    a = np.array(sims, dtype=float)
    return float(a.min()), float(a.mean()), float(a.max())


def seps(cos_by_class):
    """Топическая разделимость."""
    merge_lo = min(min(cos_by_class["paraphrase"]), min(cos_by_class["opposite_polarity"]))
    sep_merge = merge_lo - max(cos_by_class["adjacent"])
    sep_flag = min(cos_by_class["adjacent"]) - max(cos_by_class["unrelated"])
    return sep_merge, sep_flag, merge_lo


def thresholds(cos_by_class, merge_lo):
    """MERGE = середина между max(adjacent) и min(merge-группы);
       FLAG  = середина между max(unrelated) и min(adjacent)."""
    merge_t = (max(cos_by_class["adjacent"]) + merge_lo) / 2.0
    flag_t = (max(cos_by_class["unrelated"]) + min(cos_by_class["adjacent"])) / 2.0
    return merge_t, flag_t


def print_table(title, cos_by_class):
    print(f"\n=== {title} ===")
    print(f"{'класс':<18} {'n':>3} {'min':>7} {'mean':>7} {'max':>7}")
    print("-" * 48)
    for cls in CLASSES:
        mn, me, mx = stats(cos_by_class[cls])
        print(f"{cls:<18} {len(cos_by_class[cls]):>3} {mn:>7.3f} {me:>7.3f} {mx:>7.3f}")
    sm, sf, _ = seps(cos_by_class)
    print(f"sep_MERGE = {sm:+.3f}  {'OK' if sm > 0 else 'ПЕРЕКРЫТИЕ'}   "
          f"sep_FLAG = {sf:+.3f}  {'OK' if sf > 0 else 'ПЕРЕКРЫТИЕ'}")


def report_model(name, cos_raw):
    print_table(f"{name} — RAW", cos_raw)
    cos_rc, floor = recenter(cos_raw)
    print_table(f"{name} — RE-CENTERED (floor=mean(unrelated)={floor:.3f})", cos_rc)

    sm_raw, sf_raw, lo_raw = seps(cos_raw)
    sm_rc, sf_rc, lo_rc = seps(cos_rc)
    mt_raw, ft_raw = thresholds(cos_raw, lo_raw)
    mt_rc, ft_rc = thresholds(cos_rc, lo_rc)

    return {
        "floor": floor,
        "raw": {"sep_merge": sm_raw, "sep_flag": sf_raw, "merge_t": mt_raw, "flag_t": ft_raw},
        "rc": {"sep_merge": sm_rc, "sep_flag": sf_rc, "merge_t": mt_rc, "flag_t": ft_rc},
    }


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Нет sentence-transformers. pip install sentence-transformers")
        return 1

    pairs = load_pairs()
    n = sum(len(v) for v in pairs.values())
    print(f"Пар: {n} (" + ", ".join(f"{c}={len(pairs[c])}" for c in CLASSES) + ")")

    print(f"\nЗагрузка {MINILM_NAME} ...")
    m_mini = SentenceTransformer(MINILM_NAME)
    print(f"Загрузка {E5_NAME} ...")
    m_e5 = SentenceTransformer(E5_NAME)

    mini = cosines_for_model(m_mini, pairs, e5_prefix=False)
    e5 = cosines_for_model(m_e5, pairs, e5_prefix=True)

    r_mini = report_model(f"MiniLM ({MINILM_NAME})", mini)
    r_e5 = report_model(f"e5-small ({E5_NAME}, prefix 'query: ')", e5)

    print("\n=== ПРЕДЛОЖЕННЫЕ ПОРОГИ ДЛЯ e5 ===")
    print(f"RAW          : MERGE_e5 = {r_e5['raw']['merge_t']:.3f}  FLAG_e5 = {r_e5['raw']['flag_t']:.3f}"
          f"   (sep_MERGE={r_e5['raw']['sep_merge']:+.3f}, sep_FLAG={r_e5['raw']['sep_flag']:+.3f})")
    print(f"RE-CENTERED  : MERGE_e5'= {r_e5['rc']['merge_t']:.3f}  FLAG_e5'= {r_e5['rc']['flag_t']:.3f}"
          f"   (sep_MERGE={r_e5['rc']['sep_merge']:+.3f}, sep_FLAG={r_e5['rc']['sep_flag']:+.3f})")
    print(f"             floor (e5) = {r_e5['floor']:.3f};  sim' = (cos - floor)/(1 - floor)")

    print("\n=== ВЕРДИКТ ===")
    clean_raw = r_e5["raw"]["sep_merge"] > 0 and r_e5["raw"]["sep_flag"] > 0
    clean_rc = r_e5["rc"]["sep_merge"] > 0 and r_e5["rc"]["sep_flag"] > 0
    print(f"e5 RAW чистые пороги (оба sep>0)?         {'ДА' if clean_raw else 'НЕТ'}")
    print(f"e5 RE-CENTERED чистые пороги (оба sep>0)? {'ДА' if clean_rc else 'НЕТ'}")
    if clean_rc and not clean_raw:
        print("-> Ре-центрирование возвращает e5 чистые пороги. Свап оправдан с sim'.")
    elif clean_raw:
        print("-> e5 разделяет и без ре-центрирования.")
    else:
        print("-> Даже после ре-центрирования FLAG не отделяется: абсолютные пороги")
        print("   несовместимы с анизотропией e5. Варианты: e5 только под MERGE, либо")
        print("   менее анизотропная модель (gte-small).")

    print("\n(Прод-пороги MiniLM: MERGE=0.82, FLAG=0.72.) Прод не трогался, run_drift не запускался.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
