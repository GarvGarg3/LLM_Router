import os
import json
import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

CHEAP_MODEL = "groq/llama-3.1-8b-instant"
STRONG_MODEL = "groq/llama-3.3-70b-versatile"
PRICES = {
    CHEAP_MODEL: (0.05, 0.08),
    STRONG_MODEL: (0.59, 0.79),
}

APP = "llm_router"
USER = "bench_user"
REF_FILE = "references.json"   # cached strong answers live here

# CHANGE 1: temperature=0 on router + judge → same input gives same output every run
router_agent = LlmAgent(
    name="RouterAgent",
    model=LiteLlm(model=CHEAP_MODEL, temperature=0),
    description="Routes queries between a cheap 8B model and a strong 70B model.",
    instruction=(
        "You decide whether a small 8B model can handle a query WELL, or whether it "
        "should go to a larger, stronger model. Judge honestly — do not lean toward "
        "either option to save money.\n\n"
        "Choose CHEAP for straightforward queries an 8B model handles reliably:\n"
        "- factual questions, definitions, lookups\n"
        "- basic math and unit conversions\n"
        "- summarizing, rephrasing, formatting, extraction\n"
        "- short, simple explanations\n\n"
        "Choose STRONG whenever the query involves ANY of these, where an 8B model "
        "often makes mistakes:\n"
        "- mathematical proofs, derivations, or multi-step calculation\n"
        "- algorithm design, complexity analysis, or non-trivial coding\n"
        "- concurrency, distributed systems, or security reasoning\n"
        "- multi-step reasoning where an early error would cascade\n"
        "- weighing several competing trade-offs\n"
        "- careful, precise, or specialized domain analysis\n\n"
        "If a query plausibly needs careful step-by-step reasoning or could trip up a "
        "small model, choose STRONG. Only choose CHEAP when you are confident the 8B "
        "model will answer it well.\n\n"
        "Reply with EXACTLY one word: CHEAP or STRONG."
    ),
)

cheap_worker = LlmAgent(
    name="CheapWorker",
    model=LiteLlm(model=CHEAP_MODEL),
    description="Fast, cheap model for simple queries.",
    instruction="Answer the user's question clearly and concisely.",
)

strong_worker = LlmAgent(
    name="StrongWorker",
    model=LiteLlm(model=STRONG_MODEL),
    description="High-capability model for complex queries.",
    instruction="Answer the user's question thoroughly and accurately.",
)

# CHANGE 2: judge now grades AGAINST a reference answer instead of from its own knowledge
judge_agent = LlmAgent(
    name="JudgeAgent",
    model=LiteLlm(model=STRONG_MODEL, temperature=0),
    description="Grades a candidate answer against a reference answer, 1-5.",
    instruction=(
        "You compare a CANDIDATE answer against a REFERENCE answer that is worth 5/5.\n"
        "Score the candidate 1-5 based on how much of the reference's correctness and "
        "key points it captures:\n"
        "5 = matches the reference on all key points\n"
        "4 = correct but misses one minor point or detail from the reference\n"
        "3 = misses an important point, or is noticeably shallower\n"
        "2 = significant gaps or an error the reference does not have\n"
        "1 = wrong or unhelpful\n"
        "Reply with ONLY the digit."
    ),
)

# ── Runner plumbing ───────────────────────────────────────────────────────────
session_service = InMemorySessionService()
_call_counter = 0

async def ask(agent, prompt):
    """Run `agent` on `prompt`; return (text, input_tokens, output_tokens)."""
    global _call_counter
    _call_counter += 1
    runner = Runner(agent=agent, app_name=APP, session_service=session_service)
    sid = f"{agent.name}-{_call_counter}"
    await session_service.create_session(app_name=APP, user_id=USER, session_id=sid)
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    text, in_tok, out_tok = "", 0, 0
    async for event in runner.run_async(user_id=USER, session_id=sid, new_message=content):
        usage = getattr(event, "usage_metadata", None)
        if usage:
            in_tok = getattr(usage, "prompt_token_count", in_tok) or in_tok
            out_tok = getattr(usage, "candidates_token_count", out_tok) or out_tok
        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text
    await asyncio.sleep(2.5)
    return text, in_tok, out_tok


def cost_of(model_key, in_tok, out_tok):
    pin, pout = PRICES[model_key]
    return (in_tok * pin + out_tok * pout) / 1_000_000


QUERIES = [
    # --- simple (should route cheap) ---
    "What's 48 divided by 6?",
    "Convert 25 kilometers to miles.",
    "List the primary colors.",
    "What day comes after Wednesday?",
    # --- complex (should route strong) ---
    "Derive the time complexity of merge sort using the recurrence T(n)=2T(n/2)+O(n).",
    "Explain why deadlock requires all four Coffman conditions, with a scenario breaking one.",
    "Write a thread-safe singleton in Python and explain the double-checked locking pitfall.",
    "Given a skewed dataset, explain why accuracy misleads and which metrics to use instead.",
    # --- brutal: 8B reliably fails or gets subtly wrong ---
    "Prove there are infinitely many primes, then explain where the proof would break if applied to twin primes.",
    "Given a 3-node distributed system with network partition, walk through how a naive leader election causes split-brain and one fix.",
    "Derive the closed-form of the recurrence T(n) = 3T(n/2) + n using the Master Theorem, and state which case and why.",
    "A company's A/B test shows 5% lift with p=0.04 on 200 users. Explain three reasons this result may not replicate.",
]


# CHANGE 3: generate the strong (70B) answer for every query ONCE, cache to JSON.
# These answers are the "reference = 5/5" and also serve as the always-strong run.
async def get_reference_answers():
    if os.path.exists(REF_FILE):
        with open(REF_FILE) as f:
            refs = json.load(f)
        if all(q in refs for q in QUERIES):
            print(f"Loaded cached references from {REF_FILE}\n")
            return refs

    print("Generating reference answers with the strong model...\n")
    refs = {}
    for q in QUERIES:
        answer, a_in, a_out = await ask(strong_worker, q)
        refs[q] = {"answer": answer, "cost": cost_of(STRONG_MODEL, a_in, a_out)}
    with open(REF_FILE, "w") as f:
        json.dump(refs, f, indent=2)
    return refs


async def judge(query, answer, reference):
    prompt = (
        f"Question: {query}\n\n"
        f"REFERENCE answer (worth 5/5):\n{reference}\n\n"
        f"CANDIDATE answer:\n{answer}\n\n"
        "Score the candidate 1-5. Reply with ONLY the digit."
    )
    verdict, _, _ = await ask(judge_agent, prompt)
    for ch in verdict:
        if ch in "12345":
            return int(ch)
    return 3


async def run_cheap(refs):
    total_cost, scores = 0.0, []
    for q in QUERIES:
        answer, a_in, a_out = await ask(cheap_worker, q)
        total_cost += cost_of(CHEAP_MODEL, a_in, a_out)
        scores.append(await judge(q, answer, refs[q]["answer"]))
    return total_cost, sum(scores) / len(scores)


async def run_routed(refs):
    total_cost, scores, strong_calls = 0.0, [], 0
    for q in QUERIES:
        verdict, r_in, r_out = await ask(router_agent, q)
        total_cost += cost_of(CHEAP_MODEL, r_in, r_out)

        # CHANGE 4 (THE BUG FIX): check for "STRONG", not "COMPLEX"
        if "STRONG" in verdict.upper():
            strong_calls += 1
            # CHANGE 5: reuse the cached strong answer instead of calling the 70B again.
            # Same model + same query, and it halves your Groq calls.
            total_cost += refs[q]["cost"]
            scores.append(5)  # the reference IS the 5/5 answer by definition
        else:
            answer, a_in, a_out = await ask(cheap_worker, q)
            total_cost += cost_of(CHEAP_MODEL, a_in, a_out)
            scores.append(await judge(q, answer, refs[q]["answer"]))
    return total_cost, sum(scores) / len(scores), strong_calls


async def main():
    print(f"Benchmarking {len(QUERIES)} queries across 3 strategies (Google ADK)...\n")

    refs = await get_reference_answers()
    strong_cost = sum(r["cost"] for r in refs.values())
    strong_q = 5.0  # by definition: the reference is the 5/5 standard

    cheap_cost, cheap_q = await run_cheap(refs)
    print(f"Always cheap    cost=${cheap_cost:.6f}  quality={cheap_q:.2f}/5")
    print(f"Always strong   cost=${strong_cost:.6f}  quality={strong_q:.2f}/5 (reference)")

    routed_cost, routed_q, strong_calls = await run_routed(refs)
    print(f"Routed          cost=${routed_cost:.6f}  quality={routed_q:.2f}/5  "
          f"(strong used {strong_calls}/{len(QUERIES)})")

    print("\n── Results ──────────────────────────────────────────")
    saved = (1 - routed_cost / strong_cost) * 100
    kept = routed_q / strong_q * 100
    print(f"Router saved {saved:.1f}% cost vs always-strong, keeping {kept:.0f}% of quality.")


if __name__ == "__main__":
    asyncio.run(main())