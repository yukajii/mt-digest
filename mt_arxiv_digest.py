#!/usr/bin/env python3
"""
mt_arxiv_digest.py  – daily digest generator for MT-centric cs.CL papers
*2025-05-31 — “SemRank-lite” edition*
────────────────────────────────────────────────────────────────────────────
Instead of asking GPT to pick papers, we now:
  1. Embed each paper with a local MiniLM (E5-large-v2) model.
  2. Embed a fixed “machine-translation concept” prompt once.
  3. Cosine-score paper vs concept → keep the top-N.
No extra tokens, no external services.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, pathlib, re, textwrap, warnings
from typing import List, Dict, Tuple

import numpy as np
import arxiv, openai
from sentence_transformers import SentenceTransformer   # NEW

# ── CONSTANTS ────────────────────────────────────────────────────────────
MAX_RESULTS       = 222
DEFAULT_MAX_PICKS = 5
PREFACE_MODEL     = "gpt-4o"
USD_PER_TOKEN     = 0.000005

# arXiv’s “new submissions” e-mail typically lags ~4 days;
# using 5 days gives us a safe buffer so that 2 Jun → 28 May, as desired.
DEFAULT_DATE_LAG_DAYS = 5

EMBED_MODEL_NAME = "intfloat/e5-large-v2"               # NEW
CONCEPTS = [
    "machine translation", "localization", "globalization", "internationalization",
    "neural machine translation", "NMT", 
    "document-level translation",
    "cross-lingual transfer",
    "translation evaluation BLEU COMET chrF",
    "post-editing", "mtpe", "mtqe",
    "linguistic quality assurance", "lqa", "mqm",
    "video game"
]

BASE_DIR = pathlib.Path(__file__).parent
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── MODEL SET-UP (done once per run) ─────────────────────────────────────
warnings.filterwarnings("ignore", message=r".*deprecated.*", category=DeprecationWarning)
EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
CONCEPT_VECTOR = EMBEDDER.encode(" ; ".join(CONCEPTS), normalize_embeddings=True)

# ── HELPERS ──────────────────────────────────────────────────────────────
def fetch_cscl(date: dt.date) -> List[Dict]:
    day = date.strftime("%Y%m%d")
    q = f'cat:cs.CL AND submittedDate:[{day}0000 TO {day}2359]'
    search  = arxiv.Search(query=q, max_results=MAX_RESULTS,
                           sort_by=arxiv.SortCriterion.SubmittedDate)
    client  = arxiv.Client()
    papers: List[Dict] = []
    for p in client.results(search):
        papers.append({
            "id": p.get_short_id(),
            "title": p.title.strip().replace("\n", " "),
            "abstract": re.sub(r"\s+", " ", p.summary.strip()),
            "url": p.pdf_url,
        })
    return papers


def rank_mt_papers(papers: List[Dict], max_picks: int) -> List[int]:
    """Return 1-based indices of the top *max_picks* MT-like papers."""
    scores: List[Tuple[float, int]] = []
    for idx, p in enumerate(papers, start=1):
        vec = EMBEDDER.encode(
            f"{p['title']} {p['abstract']}", normalize_embeddings=True
        )
        scores.append((float(np.dot(CONCEPT_VECTOR, vec)), idx))   # cosine sim
    ranked = sorted(scores, reverse=True)[:max_picks]
    return [idx for _, idx in ranked]


def openai_chat(model: str, messages: List[Dict], temperature: float = 0):
    client = openai.OpenAI()
    resp   = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content, resp.usage.model_dump()

# ── PREFACE ──────────────────────────────────────────────────────────────
def draft_preface(date: dt.date, papers: List[Dict], picks: List[int]):
    chosen = [papers[i-1] for i in picks] if picks else []
    titles_block = "\n".join(f"• {p['title']}" for p in chosen) or "(no MT-specific papers today)"

    user_msg = textwrap.dedent(f"""
        You are writing the short introduction for a daily Machine Translation (MT) research digest.
        Today is {date.isoformat()}.

        Please produce **exactly 2–3 sentences**:
        • Sentence 1 – an intro (e.g. “Here is today's selection of cs.CL papers …”).  
        • Sentence 2–3 – a concise summary of the common theme(s).

        Do **not** apologise, explain relevance level, or list the papers again.

        Selected papers (titles only):
        {titles_block}
    """).strip()

    reply, usage = openai_chat(PREFACE_MODEL, [
        {"role": "system", "content": "You are a helpful research newsletter editor."},
        {"role": "user",   "content": user_msg},
    ], temperature=0.7)

    return reply.strip(), user_msg, usage

# ── OUTPUT WRITERS ───────────────────────────────────────────────────────
def write_md(date: dt.date, preface: str,
             papers: List[Dict], picks: List[int]):
    """
    Markdown layout:
      • Preface = plain paragraph
      • '---' rule AFTER preface and between papers
    """
    md: List[str] = [
        preface.strip(),
        "",
        "---",  # separator before first paper
        "",
    ]

    first = True
    for idx in picks:
        if not first:
            md += ["---", ""]          # separator between papers
        first = False

        p = papers[idx - 1]
        md += [
            f"## [{p['title']}]({p['url']})",
            "",
            p["abstract"],
            "",                        # blank line keeps HR from becoming H2
        ]

    path = BASE_DIR / f"mt_digest_{date.isoformat()}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def write_log(date: dt.date, log: Dict):
    """Dump a JSON activity log next to the Markdown file."""
    path = LOG_DIR / f"mt_digest_{date.isoformat()}.log"
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

# ── MAIN ─────────────────────────────────────────────────────────────────
def resolve_target_date(cli_pos, cli_flag, env_var):
    """
    Resolve the intended arXiv ‘submission date’ to use.
    Precedence:
      1. positional CLI argument
      2. --date flag
      3. DATE environment variable (set in CI)
      4. Fallback → today − DEFAULT_DATE_LAG_DAYS
    """
    if cli_pos:
        return dt.datetime.strptime(cli_pos, "%Y-%m-%d").date()
    if cli_flag:
        return cli_flag
    if env_var:
        try:
            return dt.datetime.strptime(env_var, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit(f"Bad DATE env-var format: {env_var} (want YYYY-MM-DD)")
    # automatic lag (covers manual runs & mis-configured CI)
    return dt.date.today() - dt.timedelta(days=DEFAULT_DATE_LAG_DAYS)


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY env var missing")

    ap = argparse.ArgumentParser(description="Generate daily MT-centric arXiv digest.")
    ap.add_argument("date", nargs="?", help="Target UTC date YYYY-MM-DD (positional)")
    ap.add_argument("--date", dest="date_flag",
                    type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date(),
                    help="Target UTC date (flag).")
    ap.add_argument("--max", dest="max_picks", type=int,
                    default=DEFAULT_MAX_PICKS,
                    help=f"Maximum papers to include (default: {DEFAULT_MAX_PICKS}).")
    ns = ap.parse_args()

    target_date = resolve_target_date(ns.date, ns.date_flag, os.getenv("DATE"))

    papers = fetch_cscl(target_date)
    if not papers:
        print("No cs.CL papers on that date.")
        return

    picks = rank_mt_papers(papers, ns.max_picks)

    preface, preface_prompt, preface_usage = draft_preface(target_date, papers, picks)
    md_path = write_md(target_date, preface, papers, picks)

    log_dict = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "target_date": target_date.isoformat(),
        "total_papers": len(papers),
        "picked_indices": picks,
        "token_usage": {
            "preface_call": preface_usage,
            "grand_total": preface_usage.get("total_tokens", 0),
            "approx_cost_usd": round(
                preface_usage.get("total_tokens", 0) * USD_PER_TOKEN, 4
            ),
        },
        "preface_prompt_sent": preface_prompt,
    }
    log_path = write_log(target_date, log_dict)

    print(f"✓ Digest → {md_path.name}   |   Log → {log_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
