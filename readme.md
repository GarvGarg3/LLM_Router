# Cost-Optimizing Multi-Agent LLM Router — built with Google ADK

A multi-agent system that routes each incoming query to either a cheap 8B model or a strong 70B model, then benchmarks the routed strategy against always-cheap and always-strong baselines on **cost**, **quality** (LLM-as-judge), and **strong-model usage**.

The interesting part of this repo is not the router. It is the eval harness: v1 produced results that were internally impossible, and fixing the harness — not the models — is what made the numbers mean anything.

---

## Results

Benchmark: «N» queries across «easy / medium / hard» tiers. Groq free tier, `temperature=0` on router and judge.

| Strategy | Cost | Quality (1–5) | Strong model used |
| --- | --- | --- | --- |
| Always cheap | $«0.000000» | «0.00» | 0 / «N» |
| Always strong | $«0.000000» | 5.00 (reference) | «N» / «N» |
| **Routed** | **$«0.000000»** | **«0.00»** | **«k» / «N»** |

**Headline:** the router retains «~95%» of strong-model quality at «~75%» of the always-strong cost.

Per-tier averages (below) matter more than the overall number — two-thirds of the query set is easy, which understates the router's value.

| Tier | Cheap quality | Routed quality | Strong quality |
| --- | --- | --- | --- |
| Easy | «0.00» | «0.00» | 5.00 |
| Hard | «0.00» | «0.00» | 5.00 |

> Note on "always strong = 5.00": under reference-based judging the strong model's answer *is* the yardstick, so it scores 5 by definition. The honest framing of this system is therefore "% of strong quality retained at % of the cost", not "the router beats the baselines".

---

## Architecture

**Two models (via Groq free tier):**

- Cheap model → `groq/llama-3.1-8b-instant`
- Strong model → `groq/llama-3.3-70b-versatile`

**Four agents:**

- **RouterAgent** (cheap model, `temperature=0`) → classifies each query as CHEAP or STRONG
- **CheapWorker** (cheap model) → answers queries routed cheap
- **StrongWorker** (strong model) → answers queries routed strong
- **JudgeAgent** (strong model, `temperature=0`) → grades every answer 1–5 against a cached reference

```
query ──► RouterAgent ──┬── CHEAP ──► CheapWorker ──┐
                        │                            ├──► JudgeAgent ──► score 1–5
                        └── STRONG ─► cached ref ────┘         ▲
                                                                │
                                    references.json (strong answers, 5/5 by definition)
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="your_key"  # free tier is enough
python ai_router.py
```

> **Note:** The benchmark runs slower than you would expect because of Groq's free-tier rate limits — there is a deliberate 2.5s sleep after every call. On a paid tier, remove that sleep for a much faster run.

`references.json` is generated on first run and reused afterwards. **Delete it whenever the query list changes**, or the judge will grade new queries against stale references.

---

## v1: The Original Design (and What Went Wrong)

The first version had the router classify each query, send it to the matching worker, and have the judge score every answer 1–5 from its own knowledge.

**v1 results:**

```
Always cheap    cost=$0.000467   quality=4.40/5
Always strong   cost=$0.006123   quality=4.65/5
Routed          cost=$0.000994   quality=4.30/5   (strong used 0/«N»)
```

These results are impossible on their face: the routed strategy cost **more** than always-cheap while scoring **lower** than both baselines, and the strong model was never selected once. Three separate problems were responsible.

### Issue 1: Dispatch never reached the strong worker

`strong used 0/«N»` was not a routing failure — the router *was* classifying queries as STRONG. The dispatch logic compared the router's raw output against an expected label by string match, and the router's response never matched exactly (extra whitespace / surrounding text). Every query silently fell through to the cheap worker.

This also explains the cost anomaly: the routed run paid for a router call *plus* a cheap worker call on every query, which is strictly more expensive than always-cheap.

### Issue 2: Judge scores were noisy

At default temperature, the judge would give the same answer a 4 on one run and a 5 on the next. With only «N» queries, a single flipped grade moves the average by ~0.08 — enough to change which strategy "wins" between runs. The 4.40 vs 4.30 gap in v1 was judge noise, not a real quality difference.

### Issue 3: The judge graded from its own knowledge

Absolute scoring ("rate this 1–5") is a hard task for an LLM — it has to know the correct answer itself *and* map quality onto a number scale consistently. This compressed every score into the 4.3–4.7 band, leaving almost no separation between strategies.

---

## v2: The Fixes

### Fix 1: Parse the router's decision instead of string-matching it

Dispatch now normalizes the router output and checks for the label rather than requiring an exact match, with an explicit fallback if neither label is present. The `strong used 0/«N»` symptom disappeared immediately.

### Fix 2: `temperature=0` on the router and judge

```python
model=LiteLlm(model=CHEAP_MODEL, temperature=0)   # router
model=LiteLlm(model=STRONG_MODEL, temperature=0)  # judge
```

Decision-making agents are now deterministic: the same query always routes the same way, and the same answer always gets the same grade. Run-to-run differences reflect real changes, not sampling noise. Workers keep default temperature — variety in *answers* is fine; variety in *decisions* is noise.

Temperature 0 fixes **variance**, not **bias**. A judge that consistently over-rates an answer will do so reproducibly. That is what Fix 3 is for.

### Fix 3: Reference-based judging

Instead of grading from its own knowledge, the judge receives the strong model's answer as a **reference worth 5/5** and scores each candidate relative to it. Comparing two answers is a far more reliable task for an LLM than absolute scoring, and it removes the judge's own knowledge from the loop.

Strong-model answers are generated once per query set and cached to `references.json`, so re-runs (e.g. after tweaking the router prompt) don't regenerate them.

### Fix 4: Reuse cached strong answers in the routed run

When the router picks STRONG, the system reuses the cached reference answer instead of calling the 70B again for the same query. Same model, same query — and it halves strong-model API calls, which matters on a rate-limited free tier.

---

## Observations

- `llama-3.3-70b-versatile` is consistently high quality across the whole test set. `llama-3.1-8b-instant` is genuinely solid on simple queries but noticeably noisy on multi-step reasoning — proofs, recurrences, distributed-systems questions.
- Because two-thirds of the query set is easy, overall averages understate the router's value. The interesting comparison is the hard tier, where the cheap model's scores drop sharply. Per-tier averages tell the real story.
- Small benchmarks («N» queries) cannot distinguish quality gaps of ~0.25 points at nonzero temperature. Determinism plus reference judging is what makes numbers trustworthy at this scale.

## Lessons Learned

1. **When a benchmark result makes no sense, suspect the harness before the models.** "Strong used 0/«N»" was a string-matching bug in dispatch, not a routing failure.
2. **Separate variance from bias.** Temperature 0 makes results reproducible; reference-based judging makes them accurate. You need both.
3. **LLMs compare better than they score.** Pinning a reference answer at 5/5 produced far more meaningful grades than asking for absolute 1–5 ratings.
4. **Cache anything expensive and deterministic.** A JSON file of reference answers was enough — no database needed at this scale.

## Limitations

- «N»-query benchmark; too small to make strong claims about generalization.
- Cost figures are computed from published Groq per-token pricing, not billed amounts.
- The judge is the same 70B that produces the references, so it is grading its own family of outputs.
