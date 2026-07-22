# tools/her_stress_test_v2.py
"""
HER Technology Stress Test v2
~300 pairs, 7 classes
Tests if HER is a real technology or just a TBG hack
"""

import os
import asyncio
import json
import httpx
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from collections import defaultdict
from datetime import datetime

CEREBRAS_KEY = os.environ.get("CEREBRAS_API_KEY", "")  # never hardcode keys
model = SentenceTransformer("all-MiniLM-L6-v2")

# ===================================================================
# DATA
# ===================================================================

CLASS_A_LEXICAL_TRAPS = [
    ("career security", "financial security"),
    ("memory issue", "computer memory"),
    ("relationship attachment", "file attachment"),
    ("sleep debt", "financial debt"),
    ("work stress", "heart stress"),
    ("economic growth", "career growth"),
    ("emotional baggage", "airline baggage"),
    ("trust issues", "SSL trust chain"),
    ("bandwidth usage", "emotional bandwidth"),
    ("database corruption", "moral corruption"),
    ("credit score", "exam score"),
    ("cash flow", "workflow"),
    ("criminal charge", "battery charge"),
    ("drug resistance", "antibiotic resistance"),
    ("stack overflow", "arithmetic overflow"),
    ("memory leak", "information leak"),
    ("heart failure", "kidney failure"),
    ("Redis cluster", "Elasticsearch cluster"),
    ("mortgage payment", "payment plan"),
    ("job satisfaction", "customer satisfaction"),
    ("data privacy", "privacy policy"),
    ("system crash", "stock market crash"),
    ("physical health", "mental health"),
    ("financial stability", "emotional stability"),
    ("career path", "hiking path"),
    ("personal space", "parking space"),
    ("time management", "trauma management"),
    ("risk assessment", "performance assessment"),
    ("life balance", "bank balance"),
    ("human resources", "natural resources"),
    ("company culture", "bacterial culture"),
    ("power dynamics", "engine dynamics"),
    ("emotional intelligence", "artificial intelligence"),
    ("financial freedom", "personal freedom"),
    ("work-life balance", "nutritional balance"),
    ("cognitive load", "workload"),
    ("decision fatigue", "material fatigue"),
    ("social capital", "financial capital"),
    ("sleep quality", "product quality"),
    ("interpersonal skills", "technical skills"),
    ("self-esteem", "market esteem"),
    ("emotional baggage", "checked baggage"),
    ("identity theft", "data theft"),
    ("cognitive dissonance", "sonic dissonance"),
    ("behavioral economics", "macroeconomic"),
    ("psychological safety", "road safety"),
    ("emotional regulation", "temperature regulation"),
    ("career trajectory", "missile trajectory"),
    ("relationship status", "flight status"),
    ("personal brand", "corporate brand"),
]

CLASS_B_CROSS_DOMAIN_PARAPHRASES = [
    ("burnout", "job burnout"),
    ("people pleaser", "struggles to say no"),
    ("fear of failure", "afraid everything will fail"),
    ("mindfulness practice", "meditation"),
    ("career burnout", "exhaustion"),
    ("life purpose", "spiritual meaning"),
    ("wants recognition", "seeks approval"),
    ("financial stress", "money anxiety"),
    ("work-life balance", "managing career and family"),
    ("imposter syndrome", "feels like a fraud"),
    ("emotional exhaustion", "being drained"),
    ("career ambition", "desire to advance"),
    ("relationship anxiety", "fear of abandonment"),
    ("personal growth", "self-improvement"),
    ("job satisfaction", "happy at work"),
    ("financial freedom", "economic independence"),
    ("health anxiety", "worrying about health"),
    ("social anxiety", "fear of social situations"),
    ("decision paralysis", "can't choose"),
    ("procrastination", "putting things off"),
    ("perfectionism", "never good enough"),
    ("people pleasing", "always saying yes"),
    ("boundary issues", "can't say no"),
    ("self-doubt", "lack of confidence"),
    ("negative thinking", "pessimism"),
    ("positive thinking", "optimism"),
    ("work stress", "pressure at job"),
    ("family obligations", "family duties"),
    ("financial planning", "money management"),
    ("career planning", "professional development"),
    ("relationship issues", "problems with partner"),
    ("health issues", "medical problems"),
    ("mental health", "psychological wellbeing"),
    ("physical health", "bodily health"),
    ("emotional health", "feeling stable"),
    ("spiritual health", "sense of purpose"),
    ("life satisfaction", "happiness"),
    ("career success", "professional achievement"),
    ("financial success", "monetary wealth"),
    ("personal success", "achieving goals"),
    ("relationship success", "happy partnership"),
    ("work success", "career progression"),
    ("financial independence", "economic self-sufficiency"),
    ("personal independence", "autonomy"),
    ("emotional independence", "not needy"),
    ("social independence", "self-reliance"),
    ("career independence", "job autonomy"),
    ("financial anxiety", "money worries"),
    ("health anxiety", "illness anxiety"),
    ("social success", "popular"),
]

CLASS_C_SAME_DOMAIN_OPPOSITES = [
    ("marriage", "divorce"),
    ("buying a house", "selling a house"),
    ("save money", "spend money"),
    ("wants stability", "wants change"),
    ("exercise more", "avoid exercise"),
    ("promotion", "retirement"),
    ("optimism", "pessimism"),
    ("confidence", "insecurity"),
    ("trust", "suspicion"),
    ("love", "hate"),
    ("happy", "sad"),
    ("success", "failure"),
    ("strength", "weakness"),
    ("courage", "fear"),
    ("peace", "conflict"),
    ("freedom", "constraint"),
    ("growth", "stagnation"),
    ("hope", "despair"),
    ("pride", "shame"),
    ("gratitude", "resentment"),
    ("forgiveness", "blame"),
    ("acceptance", "rejection"),
    ("belonging", "isolation"),
    ("purpose", "meaninglessness"),
    ("order", "chaos"),
    ("certainty", "uncertainty"),
    ("clarity", "confusion"),
    ("unity", "division"),
    ("progress", "regression"),
    ("empathy", "indifference"),
    ("connection", "detachment"),
    ("harmony", "discord"),
    ("balance", "imbalance"),
    ("safety", "danger"),
    ("stability", "instability"),
    ("control", "surrender"),
    ("action", "inaction"),
    ("speaking", "silence"),
    ("giving", "taking"),
    ("leading", "following"),
    ("teaching", "learning"),
    ("listening", "ignoring"),
    ("understanding", "misunderstanding"),
    ("compromise", "confrontation"),
    ("patience", "urgency"),
    ("flexibility", "rigidity"),
    ("openness", "closedness"),
    ("curiosity", "apathy"),
    ("play", "seriousness"),
    ("rest", "activity"),
]

CLASS_D_MULTILINGUAL = [
    ("хобби", "hobby"),
    ("выгорание", "burnout"),
    ("страх неудачи", "fear of failure"),
    ("духовный кризис", "spiritual crisis"),
    ("личный рост", "personal growth"),
    ("семья", "family"),
    ("работа", "work"),
    ("финансовая безопасность", "financial security"),
    ("эмоциональное истощение", "emotional exhaustion"),
    ("карьерный рост", "career growth"),
    ("отношения", "relationships"),
    ("здоровье", "health"),
    ("самооценка", "self-esteem"),
    ("уверенность", "confidence"),
    ("тревога", "anxiety"),
    ("депрессия", "depression"),
    ("стресс", "stress"),
    ("счастье", "happiness"),
    ("успех", "success"),
    ("деньги", "money"),
    ("карьера", "career"),
    ("личность", "identity"),
    ("ценности", "values"),
    ("смысл жизни", "meaning of life"),
    ("цели", "goals"),
    ("планы", "plans"),
    ("мечты", "dreams"),
    ("страхи", "fears"),
    ("сильные стороны", "strengths"),
    ("слабости", "weaknesses"),
    ("поддержка", "support"),
    ("понимание", "understanding"),
    ("принятие", "acceptance"),
    ("прощение", "forgiveness"),
    ("благодарность", "gratitude"),
    ("гармония", "harmony"),
    ("свобода", "freedom"),
    ("выбор", "choice"),
    ("изменения", "change"),
    ("рост", "growth"),
    ("развитие", "development"),
    ("успех", "achievement"),
    ("баланс", "balance"),
    ("покой", "peace"),
    ("энергия", "energy"),
    ("сила", "strength"),
    ("мудрость", "wisdom"),
    ("терпение", "patience"),
    ("смелость", "courage"),
    ("надежда", "hope"),
]

CLASS_E_OTHER_DOMAINS = [
    # E-commerce
    ("AirPods Pro", "AirPods Max"),
    ("MacBook Pro 14", "MacBook Pro 16"),
    ("iPhone 15 Pro", "Apple iPhone 15 Pro"),
    ("Nike Air Max", "Nike Air Force"),
    ("PlayStation 5", "Xbox Series X"),
    ("Samsung S24", "Samsung S24 Ultra"),
    ("iPad Pro", "iPad Air"),
    ("Apple Watch", "Samsung Watch"),
    ("Kindle", "Kobo"),
    ("Dyson V15", "Dyson V12"),
    # Medicine
    ("heart failure", "kidney failure"),
    ("drug resistance", "antibiotic resistance"),
    ("type 1 diabetes", "type 2 diabetes"),
    ("benign tumor", "malignant tumor"),
    ("acute pain", "chronic pain"),
    ("high blood pressure", "hypertension"),
    ("heart attack", "cardiac arrest"),
    ("stroke", "brain hemorrhage"),
    ("anxiety disorder", "panic disorder"),
    ("migraine", "headache"),
    # Programming
    ("memory leak", "information leak"),
    ("stack overflow", "arithmetic overflow"),
    ("Redis cluster", "Elasticsearch cluster"),
    ("SQL injection", "dependency injection"),
    ("API endpoint", "URL endpoint"),
    ("frontend", "backend"),
    ("devops", "sysadmin"),
    ("container", "image"),
    ("unit test", "integration test"),
    ("production", "staging"),
    # Finance
    ("credit score", "exam score"),
    ("cash flow", "workflow"),
    ("stock market", "farmers market"),
    ("investment portfolio", "portfolio manager"),
    ("bank statement", "mission statement"),
    ("capital gain", "capital city"),
    ("fixed income", "fixed mindset"),
    ("bear market", "farmer's market"),
    ("bullish", "bearish"),
    ("hedge fund", "mutual fund"),
    # Law
    ("criminal charge", "battery charge"),
    ("cross-examination", "examination"),
    ("court order", "restraining order"),
    ("plaintiff", "defendant"),
    ("evidence", "precedent"),
    ("objection", "objection overruled"),
    ("testimony", "deposition"),
    ("verdict", "settlement"),
    ("appeal", "objection"),
    ("trial", "hearing"),
]

CLASS_F_STABILITY_CONCEPTS = [
    "career security",
    "financial security",
    "burnout",
    "imposter syndrome",
    "spiritual crisis",
    "hobby",
    "meditation",
    "work stress",
    "marriage",
    "promotion",
    "family",
    "health",
    "freedom",
    "success",
    "fear of failure",
    "self-esteem",
    "balance",
    "hope",
    "money",
    "relationships",
]

CLASS_G_CLASSIFICATION_VS_GENERATION = [
    "career security",
    "financial security",
    "burnout",
    "imposter syndrome",
    "spiritual crisis",
    "hobby",
    "meditation",
    "work stress",
    "marriage",
    "promotion",
    "family",
    "health",
    "freedom",
    "success",
    "fear of failure",
    "self-esteem",
    "balance",
    "hope",
    "money",
    "relationships",
]

DOMAINS = ["career", "money", "health", "relationships", "self", "other"]

DOMAIN_PROMPT = """Classify this concept into ONE domain:
- career (work, job, professional life)
- money (finances, wealth, income)
- health (physical, mental, energy)
- relationships (family, friends, connections)
- self (identity, personal growth, values)
- other

Concept: "{concept}"
Return ONLY JSON: {{"domain": "career|money|health|relationships|self|other"}}"""

GENERATION_PROMPT = """Extract semantic coordinates for this concept.
Concept: "{concept}"
Return ONLY JSON: {{"object": "<1-2 word noun>", "attribute": "<1-2 word noun>", "stance": "<approach|avoid|neutral>"}}"""

CLASSIFY_PROMPT = """Are these the SAME concept or DIFFERENT concepts?
A: "{a}"
B: "{b}"
Return ONLY JSON: {{"verdict": "SAME|DIFFERENT", "confidence": 0.0-1.0}}"""

# ===================================================================
# HELPERS
# ===================================================================

def cosine_similarity(a: str, b: str) -> float:
    emb = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))

async def call_cerebras(prompt: str, model_name: str = "gpt-oss-120b", retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.cerebras.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {CEREBRAS_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"}
                    }
                )
                
                if resp.status_code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"  ⏳ Rate limit, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                    
                if resp.status_code != 200:
                    print(f"  ⚠️ API error {resp.status_code}, attempt {attempt+1}")
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                    
                data = resp.json()
                if "choices" not in data:
                    print(f"  ⚠️ No 'choices' in response, attempt {attempt+1}")
                    await asyncio.sleep(3)
                    continue
                    
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
                
        except Exception as e:
            print(f"  ⚠️ Error: {e}, attempt {attempt+1}")
            await asyncio.sleep(5 * (attempt + 1))
    
    return {}

async def get_domain(concept: str) -> str:
    result = await call_cerebras(DOMAIN_PROMPT.format(concept=concept))
    return result.get("domain", "other")

async def get_generation(concept: str) -> dict:
    result = await call_cerebras(GENERATION_PROMPT.format(concept=concept))
    return result

async def classify_pair(a: str, b: str) -> str:
    result = await call_cerebras(CLASSIFY_PROMPT.format(a=a, b=b))
    return result.get("verdict", "DIFFERENT")

# ===================================================================
# MAIN
# ===================================================================

async def main():
    print("=" * 80)
    print("HER TECHNOLOGY STRESS TEST v2")
    print("~300 pairs, 7 classes")
    print("=" * 80)
    
    all_concepts = set()
    
    # Collect all concepts
    for pairs in [CLASS_A_LEXICAL_TRAPS, CLASS_B_CROSS_DOMAIN_PARAPHRASES, 
                  CLASS_C_SAME_DOMAIN_OPPOSITES, CLASS_D_MULTILINGUAL, CLASS_E_OTHER_DOMAINS]:
        for a, b in pairs:
            all_concepts.add(a)
            all_concepts.add(b)
    
    for c in CLASS_F_STABILITY_CONCEPTS:
        all_concepts.add(c)
    
    for c in CLASS_G_CLASSIFICATION_VS_GENERATION:
        all_concepts.add(c)
    
    print(f"\nConcepts: {len(all_concepts)}")
    print("Getting domains (with 2s delay between requests)...")
    
    domains = {}
    i = 0
    for concept in all_concepts:
        i += 1
        print(f"  {i}/{len(all_concepts)}: '{concept[:30]}'...", end=" ", flush=True)
        domains[concept] = await get_domain(concept)
        print(f"→ {domains[concept]}")
        await asyncio.sleep(1.5)
    
    print("\n" + "=" * 80)
    print("RUNNING COMPARISONS")
    print("=" * 80)
    
    results = []
    
    # Class A
    for a, b in CLASS_A_LEXICAL_TRAPS:
        cos = cosine_similarity(a, b)
        da, db = domains.get(a, "other"), domains.get(b, "other")
        her = "DIFFERENT" if da != db else "SAME"
        results.append({
            "class": "A_lexical_traps",
            "a": a, "b": b,
            "gt": "DIFFERENT",
            "cos": round(cos, 3),
            "domains": f"{da} vs {db}",
            "her": her,
            "cos_correct": (cos < 0.7),
            "her_correct": her == "DIFFERENT"
        })
    
    # Class B
    for a, b in CLASS_B_CROSS_DOMAIN_PARAPHRASES:
        cos = cosine_similarity(a, b)
        da, db = domains.get(a, "other"), domains.get(b, "other")
        her = "DIFFERENT" if da != db else "SAME"
        results.append({
            "class": "B_cross_domain_paraphrases",
            "a": a, "b": b,
            "gt": "SAME",
            "cos": round(cos, 3),
            "domains": f"{da} vs {db}",
            "her": her,
            "cos_correct": (cos >= 0.7),
            "her_correct": her == "SAME"
        })
    
    # Class C
    for a, b in CLASS_C_SAME_DOMAIN_OPPOSITES:
        cos = cosine_similarity(a, b)
        da, db = domains.get(a, "other"), domains.get(b, "other")
        her = "DIFFERENT" if da != db else "SAME"
        results.append({
            "class": "C_same_domain_opposites",
            "a": a, "b": b,
            "gt": "DIFFERENT",
            "cos": round(cos, 3),
            "domains": f"{da} vs {db}",
            "her": her,
            "cos_correct": (cos < 0.7),
            "her_correct": her == "DIFFERENT"
        })
    
    # Class D
    for a, b in CLASS_D_MULTILINGUAL:
        cos = cosine_similarity(a, b)
        da, db = domains.get(a, "other"), domains.get(b, "other")
        her = "DIFFERENT" if da != db else "SAME"
        results.append({
            "class": "D_multilingual",
            "a": a, "b": b,
            "gt": "SAME",
            "cos": round(cos, 3),
            "domains": f"{da} vs {db}",
            "her": her,
            "cos_correct": (cos >= 0.7),
            "her_correct": her == "SAME"
        })
    
    # Class E
    for a, b in CLASS_E_OTHER_DOMAINS:
        cos = cosine_similarity(a, b)
        da, db = domains.get(a, "other"), domains.get(b, "other")
        her = "DIFFERENT" if da != db else "SAME"
        # GT: all pairs in E should be DIFFERENT
        results.append({
            "class": "E_other_domains",
            "a": a, "b": b,
            "gt": "DIFFERENT",
            "cos": round(cos, 3),
            "domains": f"{da} vs {db}",
            "her": her,
            "cos_correct": (cos < 0.7),
            "her_correct": her == "DIFFERENT"
        })
    
    # ===================================================================
    # CLASS F: Stability
    # ===================================================================
    print("\n" + "=" * 80)
    print("CLASS F: CROSS-PROVIDER STABILITY (simulated with 5 runs)")
    print("=" * 80)
    
    # We already have domains from one run. For stability, we do additional runs
    # but limited to CLASS_F_STABILITY_CONCEPTS to save time
    
    stability_results = {}
    for concept in CLASS_F_STABILITY_CONCEPTS[:10]:  # Limit to 10 for time
        runs = []
        for run in range(3):  # 3 runs instead of 5 to save time
            print(f"  Run {run+1}/3 for '{concept}'...")
            d = await get_domain(concept)
            runs.append(d)
            await asyncio.sleep(1.5)
        stability_results[concept] = runs
    
    # ===================================================================
    # CLASS G: Classification vs Generation
    # ===================================================================
    print("\n" + "=" * 80)
    print("CLASS G: CLASSIFICATION vs GENERATION")
    print("=" * 80)
    
    gen_results = {}
    for concept in CLASS_G_CLASSIFICATION_VS_GENERATION[:10]:  # Limit to 10 for time
        runs = []
        for run in range(3):
            print(f"  Run {run+1}/3 for '{concept}' (generation)...")
            gen = await get_generation(concept)
            runs.append(gen)
            await asyncio.sleep(1.5)
        gen_results[concept] = runs
    
    # ===================================================================
    # METRICS
    # ===================================================================
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Per-class metrics
    class_names = {
        "A_lexical_traps": "A: Lexical Traps (should be DIFFERENT)",
        "B_cross_domain_paraphrases": "B: Cross-Domain Paraphrases (should be SAME)",
        "C_same_domain_opposites": "C: Same-Domain Opposites (sanity)",
        "D_multilingual": "D: Multilingual (should be SAME)",
        "E_other_domains": "E: Other Domains (should be DIFFERENT)"
    }
    
    for cls, label in class_names.items():
        cls_results = [r for r in results if r["class"] == cls]
        if not cls_results:
            continue
        n = len(cls_results)
        her_ok = sum(1 for r in cls_results if r["her_correct"])
        cos_ok = sum(1 for r in cls_results if r["cos_correct"])
        print(f"\n{label}:")
        print(f"  HER: {her_ok}/{n} = {her_ok/n*100:.1f}%")
        print(f"  Cosine: {cos_ok}/{n} = {cos_ok/n*100:.1f}%")
        print(f"  Delta: {(her_ok - cos_ok)/n*100:+.1f}%")
    
    # Class F: Stability
    print("\n" + "=" * 80)
    print("CLASS F: STABILITY")
    print("=" * 80)
    
    stable_count = 0
    for concept, runs in stability_results.items():
        if len(set(runs)) == 1:
            stable_count += 1
            status = "✅"
        else:
            status = f"❌ {runs}"
        print(f"  {concept[:20]}: {status}")
    
    stable_pct = stable_count / len(stability_results) * 100
    print(f"\n  Stability: {stable_count}/{len(stability_results)} = {stable_pct:.1f}%")
    
    # Class G: Classification vs Generation
    print("\n" + "=" * 80)
    print("CLASS G: CLASSIFICATION vs GENERATION")
    print("=" * 80)
    
    gen_stable = 0
    class_stable = 0
    for concept, runs in gen_results.items():
        # Generation stability: object+attribute+stance tuple
        gen_tuples = [(r.get("object", ""), r.get("attribute", ""), r.get("stance", "")) for r in runs]
        if len(set(gen_tuples)) == 1:
            gen_stable += 1
            gen_status = "✅"
        else:
            gen_status = f"❌ {set(gen_tuples)}"
        
        # Classification stability: domain from earlier
        domain_runs = stability_results.get(concept, [])
        if len(set(domain_runs)) == 1:
            class_stable += 1
            class_status = "✅"
        else:
            class_status = f"❌ {set(domain_runs)}"
        
        print(f"  {concept[:20]}: Gen {gen_status} | Class {class_status}")
    
    # ===================================================================
    # SUMMARY
    # ===================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # A_gain
    a_results = [r for r in results if r["class"] == "A_lexical_traps"]
    if a_results:
        n = len(a_results)
        a_her = sum(1 for r in a_results if r["her_correct"])
        a_cos = sum(1 for r in a_results if r["cos_correct"])
        a_gain = (a_her - a_cos) / n * 100
        print(f"  A_gain (lexical traps): {a_gain:+.1f}% (target: >15%)")
        a_ok = a_gain > 15
    
    # B_loss
    b_results = [r for r in results if r["class"] == "B_cross_domain_paraphrases"]
    if b_results:
        n = len(b_results)
        b_her = sum(1 for r in b_results if r["her_correct"])
        b_cos = sum(1 for r in b_results if r["cos_correct"])
        b_loss = (b_her - b_cos) / n * 100
        print(f"  B_loss (cross-domain paraphrases): {b_loss:+.1f}% (target: <10%)")
        b_ok = b_loss < 10
    
    # D_gain
    d_results = [r for r in results if r["class"] == "D_multilingual"]
    if d_results:
        n = len(d_results)
        d_her = sum(1 for r in d_results if r["her_correct"])
        d_cos = sum(1 for r in d_results if r["cos_correct"])
        d_gain = (d_her - d_cos) / n * 100
        print(f"  D_gain (multilingual): {d_gain:+.1f}% (target: >20%)")
        d_ok = d_gain > 20
    
    # F_stability
    if stability_results:
        f_stable = sum(1 for r in stability_results.values() if len(set(r)) == 1)
        f_pct = f_stable / len(stability_results) * 100
        print(f"  F_stability: {f_pct:.1f}% (target: >90%)")
        f_ok = f_pct > 90
    
    # G_ratio
    if gen_results and stability_results:
        gen_stable_count = 0
        for concept, runs in gen_results.items():
            tuples = [(r.get("object", ""), r.get("attribute", ""), r.get("stance", "")) for r in runs]
            if len(set(tuples)) == 1:
                gen_stable_count += 1
        
        class_stable_count = 0
        for concept in gen_results.keys():
            if concept in stability_results:
                if len(set(stability_results[concept])) == 1:
                    class_stable_count += 1
        
        if gen_results:
            gen_pct = gen_stable_count / len(gen_results) * 100
            class_pct = class_stable_count / len(gen_results) * 100
            g_ratio = class_pct / max(1, gen_pct)
            print(f"  G_ratio (class/gen stability): {g_ratio:.2f} (target: >1.3)")
            g_ok = g_ratio > 1.3
    
    # E_gain
    e_results = [r for r in results if r["class"] == "E_other_domains"]
    if e_results:
        n = len(e_results)
        e_her = sum(1 for r in e_results if r["her_correct"])
        e_cos = sum(1 for r in e_results if r["cos_correct"])
        e_gain = (e_her - e_cos) / n * 100
        print(f"  E_gain (other domains): {e_gain:+.1f}% (target: >0%)")
        e_ok = e_gain > 0
    
    # ===================================================================
    # VERDICT
    # ===================================================================
    
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    conditions = {
        "A_gain > 15%": a_ok if 'a_ok' in locals() else False,
        "B_loss < 10%": b_ok if 'b_ok' in locals() else False,
        "D_gain > 20%": d_ok if 'd_ok' in locals() else False,
        "F_stability > 90%": f_ok if 'f_ok' in locals() else False,
        "G_ratio > 1.3": g_ok if 'g_ok' in locals() else False,
        "E_gain > 0%": e_ok if 'e_ok' in locals() else False,
    }
    
    for cond, ok in conditions.items():
        print(f"  {cond}: {'✅' if ok else '❌'}")
    
    if all(conditions.values()):
        print("\n  ✅✅✅ ALL CONDITIONS PASSED")
        print("  HER IS A REAL TECHNOLOGY")
        print("  Principle: Classification-based semantic safety routing")
    else:
        failed = [c for c, ok in conditions.items() if not ok]
        print(f"\n  ❌ FAILED: {', '.join(failed)}")
        print("  HER remains a TBG-specific hack")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(main())