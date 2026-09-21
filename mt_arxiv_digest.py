#!/usr/bin/env python3
"""
mt_arxiv_digest.py  - daily digest generator for MT-centric cs.CL papers
Updated for OpenAI GPT-5 + Responses API
"""

from __future__ import annotations

import argparse, datetime as dt, json, os, pathlib, random, re, textwrap, warnings, time
from typing import List, Dict, Tuple

import numpy as np
import arxiv

# -- CONSTANTS -----------------------------------------------------------
MAX_RESULTS       = 400          # hard cap on papers fetched per day
DEFAULT_MAX_PICKS = 5
PREFACE_MODEL     = "gpt-5.4-mini"

# Approximate USD per *million* tokens for PREFACE_MODEL, used only for the
# cost figure in the run log. Update these whenever PREFACE_MODEL changes --
# they are estimates, not billing data.
USD_PER_MTOK_IN   = 5.00
USD_PER_MTOK_OUT  = 5.00

DEFAULT_DATE_LAG_DAYS = 5

EMBED_MODEL_NAME = "intfloat/e5-large-v2"
CONCEPTS = [
    "machine translation", "translation",
    "neural machine translation",
    "interpreting", "interpretation",
    "NMT", "document-level translation",
    "translation evaluation BLEU COMET chrF",
    "post-editing", "mtpe", "mtqe",
    "linguistic quality assurance"
]

BASE_DIR = pathlib.Path(__file__).parent
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore", message=r".*deprecated.*", category=DeprecationWarning)

# -- LAZY MODEL / CLIENT SET-UP ------------------------------------------
# Both the sentence-transformer and the OpenAI client are expensive (or fail
# outright without credentials), so they are built on first use. That keeps
# cheap invocations such as --print-date instant.
_EMBEDDER = None
_CONCEPT_VECTOR = None
_OPENAI = None


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBEDDER


def concept_vector() -> np.ndarray:
    """Single normalised vector for the blended concept string.

    e5 models are trained with "query: " / "passage: " prefixes. Adding them
    keeps the same picks on strong MT days but sharpens the gap between the
    real MT papers and the rest, which is what makes a relevance threshold
    possible (measured on 2026-08-25, 09-03, 09-10 and 09-16).

    Scoring each concept separately and taking the best match was also tried
    and was clearly worse -- short generic concepts such as "translation"
    match almost any cs.CL abstract, so every paper cherry-picks a flattering
    concept and the ranking collapses. Do not reintroduce it.
    """
    global _CONCEPT_VECTOR
    if _CONCEPT_VECTOR is None:
        _CONCEPT_VECTOR = embedder().encode(
            "query: " + " ; ".join(CONCEPTS), normalize_embeddings=True
        )
    return _CONCEPT_VECTOR


def openai_client():
    global _OPENAI
    if _OPENAI is None:
        from openai import OpenAI
        _OPENAI = OpenAI()
    return _OPENAI


# -- HELPERS -------------------------------------------------------------
# arXiv throttles anonymous automated traffic (esp. from shared cloud IPs
# like GitHub Actions runners) and responds with HTTP 429. Identify the
# client politely - arXiv asks automated tools to do so - to reduce blocks.
ARXIV_USER_AGENT = "mt-digest/1.0 (+https://github.com/yukajii/mt-digest)"

# Backoff tuning for the 429 retry loop below.
ARXIV_MAX_RETRIES = 6
ARXIV_BACKOFF_BASE = 30    # seconds; grows exponentially per attempt
ARXIV_BACKOFF_CAP = 300    # seconds; ceiling on any single wait
ARXIV_BACKOFF_JITTER = 15  # seconds; random spread added to each wait


def fetch_cscl(date: dt.date, max_retries: int = ARXIV_MAX_RETRIES) -> List[Dict]:
    day = date.strftime("%Y%m%d")
    q = f'cat:cs.CL AND submittedDate:[{day}0000 TO {day}2359]'
    search = arxiv.Search(
        query=q,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    # num_retries=0 disables the library's own retry loop so our backoff
    # logic below is the sole retry mechanism (avoids rapid-fire requests).
    # delay_seconds paces our own page requests to stay under arXiv's limit.
    client_arxiv = arxiv.Client(num_retries=0, delay_seconds=5.0)
    # Send an identifying User-Agent (best-effort; guards against library
    # internals changing the session attribute name).
    session = getattr(client_arxiv, "_session", None)
    if session is not None:
        session.headers.update({"User-Agent": ARXIV_USER_AGENT})

    attempt = 1
    while True:
        try:
            papers: List[Dict] = []
            for p in client_arxiv.results(search):
                papers.append({
                    "id": p.get_short_id(),
                    "title": p.title.strip().replace("\n", " "),
                    "abstract": re.sub(r"\s+", " ", p.summary.strip()),
                    "url": p.entry_id,          # abs page, not the raw PDF
                })
            if len(papers) >= MAX_RESULTS:
                print(f"[warn] hit MAX_RESULTS ({MAX_RESULTS}) for {date} - "
                      "some papers were not considered for ranking")
            return papers
        except arxiv.HTTPError as e:
            print(f"[warn] arxiv HTTPError on attempt {attempt}/{max_retries}: {e}")
            if attempt >= max_retries:
                raise
            # Exponential backoff with jitter: spreading retries out gives a
            # transient 429 / IP throttle time to clear instead of hammering.
            sleep_for = min(
                ARXIV_BACKOFF_BASE * (2 ** (attempt - 1)), ARXIV_BACKOFF_CAP
            ) + random.uniform(0, ARXIV_BACKOFF_JITTER)
            print(f"[info] retrying arxiv in {sleep_for:.0f} seconds...")
            time.sleep(sleep_for)
            attempt += 1


def rank_mt_papers(papers: List[Dict], max_picks: int) -> Tuple[List[int], List[Dict]]:
    """Return (1-based picked indices, per-paper ranking detail).

    Cosine similarity against the blended concept vector, e5 prefixes on
    both sides. The detail is what feeds the relevance-threshold work.
    """
    cvec = concept_vector()                                     # (dim,)
    texts = [f"passage: {p['title']} {p['abstract']}" for p in papers]
    vecs = embedder().encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
    )                                                           # (n_papers, dim)

    scores = vecs @ cvec                                        # (n_papers,)

    # Stable sort so equal scores keep arXiv's own (most-recent-first) order.
    # The previous sort broke ties by descending index, which was arbitrary.
    order = np.argsort(-scores, kind="stable")

    mean, std = float(scores.mean()), float(scores.std())
    detail = [
        {
            "rank": rank,
            "index": int(i) + 1,
            "arxiv_id": papers[i]["id"],
            "title": papers[i]["title"],
            "score": round(float(scores[i]), 4),
            # z-score against the same day's papers: raw e5 cosines sit in a
            # very narrow band, so the relative figure is the usable signal.
            "z": round((float(scores[i]) - mean) / std, 3) if std else 0.0,
            "picked": rank <= max_picks,
        }
        for rank, i in enumerate(order, start=1)
    ]

    picks = [int(i) + 1 for i in order[:max_picks]]
    return picks, detail


def openai_chat(model: str, system: str, user: str):
    response = openai_client().responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    text = response.output_text

    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "total_tokens": getattr(response.usage, "total_tokens", 0),
    }

    return text.strip(), usage


# -- PREFACE -------------------------------------------------------------
def draft_preface(date: dt.date, papers: List[Dict], picks: List[int]):
    chosen = [papers[i - 1] for i in picks] if picks else []
    titles_block = "\n".join(f"- {p['title']}" for p in chosen) or "(no MT-specific papers today)"

    user_msg = textwrap.dedent(f"""
        You are writing the short introduction for a daily Machine Translation (MT) research digest.
        Today is {date.isoformat()}.

        Please produce exactly 2-3 sentences:
        - Sentence 1 - intro
        - Sentence 2-3 - common themes

        Do not apologise or list papers again.

        Selected papers:
        {titles_block}
    """).strip()

    reply, usage = openai_chat(
        PREFACE_MODEL,
        "You are a helpful research newsletter editor.",
        user_msg,
    )

    return reply, user_msg, usage


# -- OUTPUT --------------------------------------------------------------
def write_md(date: dt.date, preface: str,
             papers: List[Dict], picks: List[int]):

    md: List[str] = [
        preface.strip(),
        "",
        "---",
        "",
    ]

    first = True
    for idx in picks:
        if not first:
            md += ["---", ""]
        first = False

        p = papers[idx - 1]
        md += [
            f"## [{p['title']}]({p['url']})",
            "",
            p["abstract"],
            "",
        ]

    path = BASE_DIR / f"mt_digest_{date.isoformat()}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def write_log(date: dt.date, log: Dict):
    path = LOG_DIR / f"mt_digest_{date.isoformat()}.log"
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# -- MAIN ----------------------------------------------------------------
def resolve_target_date(cli_pos, cli_flag, env_var):
    if cli_pos:
        return dt.datetime.strptime(cli_pos, "%Y-%m-%d").date()
    if cli_flag:
        return cli_flag
    if env_var:
        return dt.datetime.strptime(env_var, "%Y-%m-%d").date()
    d = dt.date.today() - dt.timedelta(days=DEFAULT_DATE_LAG_DAYS)
    if d.weekday() == 5:   # Saturday → Friday
        d -= dt.timedelta(days=1)
    elif d.weekday() == 6: # Sunday → Friday
        d -= dt.timedelta(days=2)
    return d


def main():
    ap = argparse.ArgumentParser(description="Generate daily MT-centric arXiv digest.")
    ap.add_argument("date", nargs="?", help="Target UTC date YYYY-MM-DD")
    ap.add_argument("--date", dest="date_flag",
                    type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date())
    ap.add_argument("--max", dest="max_picks", type=int,
                    default=DEFAULT_MAX_PICKS)
    ap.add_argument("--print-date", action="store_true",
                    help="Print the resolved target date and exit "
                         "(single source of truth for CI).")

    ns = ap.parse_args()

    target_date = resolve_target_date(ns.date, ns.date_flag, os.getenv("DATE"))

    if ns.print_date:
        print(target_date.isoformat())
        return

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY env var missing")

    papers = fetch_cscl(target_date)
    if not papers:
        print(f"No cs.CL papers on {target_date} - nothing to send.")
        return

    picks, ranking = rank_mt_papers(papers, ns.max_picks)

    preface, preface_prompt, preface_usage = draft_preface(target_date, papers, picks)
    md_path = write_md(target_date, preface, papers, picks)

    approx_cost = (
        preface_usage.get("input_tokens", 0) / 1_000_000 * USD_PER_MTOK_IN
        + preface_usage.get("output_tokens", 0) / 1_000_000 * USD_PER_MTOK_OUT
    )

    log_dict = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "target_date": target_date.isoformat(),
        "total_papers": len(papers),
        "picked_indices": picks,
        "picked_scores": [r["score"] for r in ranking if r["picked"]],
        "picked_z": [r["z"] for r in ranking if r["picked"]],
        "ranking_top_15": ranking[:15],
        "token_usage": {
            "preface_call": preface_usage,
            "grand_total": preface_usage.get("total_tokens", 0),
            "approx_cost_usd": round(approx_cost, 6),
        },
        "preface_prompt_sent": preface_prompt,
    }

    log_path = write_log(target_date, log_dict)

    print(f"[ok] Digest -> {md_path.name} | Log -> {log_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
