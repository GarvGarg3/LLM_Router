# Cost-Optimizing Multi-Agent LLM Router — built with Google ADK

A multi-agent system that routes each incoming query to either a cheap 8B model or a strong 70B model, then benchmarks the routed strategy against always-cheap and always-strong baselines on **cost**, **quality** (LLM-as-judge), and **strong-model usage**.

## Architecture

**Two models (via Groq free tier):**

* Cheap model → `groq/llama-3.1-8b-instant`
* Strong model → `groq/llama-3.3-70b-versatile`

**Four agents:**

* **RouterAgent** (cheap model) → classifies each query as CHEAP or STRONG
* **CheapWorker** (cheap model) → answers queries routed cheap
* **StrongWorker** (strong model) → answers queries routed strong
* **JudgeAgent** (strong model) → grades every answer 1–5

## Setup

```bash
python -m venv venv && source venv/bin/activate or Windows: venv\Scripts\activate
pip install google-adk litellm
export GROQ_API_KEY="your_key"(Free)
python adk_router.py
```

> **Note:** The benchmark runs slower than expected because of Groq's free-tier rate limits — there is a deliberate 2.5s sleep after every call. On a paid tier you can remove that sleep for a much faster run.

---

## v1: The Original Design (and What Went Wrong)

The first version had the router classify each query, send it to the matching worker, and have the judge score every answer 1–5 from its own knowledge.

**v1 results:**

```
Always cheap    cost=$0.000467   quality=4.40/5
Always strong   cost=$0.006123   quality=4.65/5
Routed          cost=$0.000994   quality=4.30/5   (strong used 0/20)
```

These results made no sense: the routed strategy cost **more** than always-cheap while scoring **lower** than both baselines, and the strong model was never used. Investigating revealed three separate problems.

### Issue 1: Judge scores were noisy

At default temperature, the judge would give the same answer a 4 on one run and a 5 on the next. With only 12 queries, a single flipped grade moves the average by ~0.08 — enough to change which strategy "wins" between runs. The 4.40 vs 4.30 gap in v1 was pure judge noise, not a real quality difference.

### Issue 2: The judge graded from its own knowledge

Absolute scoring ("rate this 1–5") is a hard task for an LLM — it has to know the correct answer itself and map quality onto a number scale consistently. This compressed all scores into the 4.3–4.7 band, leaving almost no separation between strategies.

---

## v2: The Fixes

### Fix 1: `temperature=0` on the router and judge

```python
model=LiteLlm(model=CHEAP_MODEL, temperature=0)   # router
model=LiteLlm(model=STRONG_MODEL, temperature=0)  # judge
```

Decision-making agents are now deterministic: the same query always routes the same way and the same answer always gets the same grade. Run-to-run differences now reflect real changes, not sampling noise. Workers keep default temperature — variety in *answers* is fine; variety in *decisions* is noise.

Note: temp 0 fixes **variance**, not **bias** — a judge that consistently over-rates an answer will do so reproducibly. That's what Fix 3 is for.

### Fix 2: Reference-based judging

Instead of grading from its own knowledge, the judge now receives the strong model's answer as a **reference worth 5/5** and scores each candidate answer relative to it. Comparing two answers is a far more reliable task for an LLM than absolute scoring.

The strong model's answers are generated once per query set and cached to `references.json`, so re-runs (e.g., after tweaking the router prompt) don't regenerate them. Delete the file whenever the query list changes.

A side effect: "always strong" now scores 5.00 **by definition** — it *is* the yardstick. The headline metric becomes *"the router keeps X% of strong-model quality at Y% of the cost"*, which is the honest framing for this system anyway.

### Fix 4: Reuse cached strong answers in the routed run

When the router picks STRONG, the system reuses the cached reference answer instead of calling the 70B again for the same query. Same model, same query — and it halves the strong-model API calls, which matters on a rate-limited free tier.

---

## Observations

* `llama-3.3-70b-versatile` is consistently high quality across the whole test set; `llama-3.1-8b-instant` is genuinely solid on simple queries but noticeably noisy on multi-step reasoning (proofs, recurrences, distributed-systems questions).
* Because two-thirds of the query set is easy, overall averages understate the router's value. The interesting comparison is on the hard tier, where the cheap model's scores drop sharply — reporting per-tier averages tells the real story.
* Small benchmarks (12 queries) cannot distinguish quality gaps of ~0.25 points at nonzero temperature. Determinism + reference judging is what makes the numbers trustworthy at this scale.

## Lessons Learned

1. **When a benchmark result makes no sense, suspect the harness before the models.** "Strong used 0/20" was a string-matching bug, not a routing failure.
2. **Separate variance from bias.** Temperature 0 makes results reproducible; reference-based judging makes them accurate. You need both.
3. **LLMs compare better than they score.** Pinning a reference answer as 5/5 produced far more meaningful grades than asking for absolute 1–5 ratings.
4. **Cache anything expensive and deterministic.** A JSON file of reference answers was enough — no database needed at this scale.