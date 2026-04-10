#!/usr/bin/env python3
"""
mt_arxiv_digest.py  – daily digest generator for MT-centric cs.CL papers
Updated for OpenAI GPT-5 + Responses API
"""

from __future__ import annotations

import argparse, datetime as dt, json, os, pathlib, re, textwrap, warnings, time
from typing import List, Dict, Tuple

import numpy as np
import arxiv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ── CONSTANTS ────────────────────────────────────────────────────────────
MAX_RESULTS       = 222
DEFAULT_MAX_PICKS = 5
PREFACE_MODEL     = "gpt-5.4"
USD_PER_TOKEN     = 0.000005

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

# ── MODEL SET-UP ─────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=r".*deprecated.*", category=DeprecationWarning)
EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
CONCEPT_VECTOR = EMBEDDER.encode(" ; ".join(CONCEPTS), normalize_embeddings=True)

# ── OPENAI CLIENT ────────────────────────────────────────────────────────
client = OpenAI()

# ── HELPERS ──────────────────────────────────────────────────────────────
def fetch_cscl(date: dt.date, max_retries: int = 3, backoff_sec: int = 20) -> List[Dict]:
    day = date.strftime("%Y%m%d")
    q = f'cat:cs.CL AND submittedDate:[{day}0000 TO {day}2359]'
    search  = arxiv.Search(
        query=q,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client_arxiv  = arxiv.Client()
    attempt = 1
    while True:
        try:
            papers: List[Dict] = []
            for p in client_arxiv.results(search):
                papers.append({
                    "id": p.get_short_id(),
                    "title": p.title.strip().replace("\n", " "),
                    "abstract": re.sub(r"\s+", " ", p.summary.strip()),
                    "url": p.pdf_url,
                })
            return papers
        except arxiv.HTTPError as e:
            print(f"[warn] arxiv HTTPError on attempt {attempt}/{max_retries}: {e}")
            if attempt >= max_retries:
                raise
            sleep_for = backoff_sec * attempt
            print(f"[info] retrying arxiv in {sleep_for} seconds...")
            time.sleep(sleep_for)
            attempt += 1


def rank_mt_papers(papers: List[Dict], max_picks: int) -> List[int]:
    scores: List[Tuple[float, int]] = []
    for idx, p in enumerate(papers, start=1):
        vec = EMBEDDER.encode(
            f"{p['title']} {p['abstract']}", normalize_embeddings=True
        )
        scores.append((float(np.dot(CONCEPT_VECTOR, vec)), idx))
    ranked = sorted(scores, reverse=True)[:max_picks]
    return [idx for _, idx in ranked]


def openai_chat(model: str, system: str, user: str, temperature: float = 0):
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        reasoning={"effort": "medium"},
    )

    text = response.output_text

    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "total_tokens": getattr(response.usage, "total_tokens", 0),
    }

    return text.strip(), usage

# ── PREFACE ──────────────────────────────────────────────────────────────
def draft_preface(date: dt.date, papers: List[Dict], picks: List[int]):
    chosen = [papers[i-1] for i in picks] if picks else []
    titles_block = "\n".join(f"• {p['title']}" for p in chosen) or "(no MT-specific papers today)"

    user_msg = textwrap.dedent(f"""
        You are writing the short introduction for a daily Machine Translation (MT) research digest.
        Today is {date.isoformat()}.

        Please produce exactly 2-3 sentences:
        • Sentence 1 - intro
        • Sentence 2-3 - common themes

        Do not apologise or list papers again.

        Selected papers:
        {titles_block}
    """).strip()

    reply, usage = openai_chat(
        PREFACE_MODEL,
        "You are a helpful research newsletter editor.",
        user_msg,
        temperature=0.7
    )

    return reply, user_msg, usage

# ── OUTPUT ───────────────────────────────────────────────────────────────
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

# ── MAIN ─────────────────────────────────────────────────────────────────
def resolve_target_date(cli_pos, cli_flag, env_var):
    if cli_pos:
        return dt.datetime.strptime(cli_pos, "%Y-%m-%d").date()
    if cli_flag:
        return cli_flag
    if env_var:
        return dt.datetime.strptime(env_var, "%Y-%m-%d").date()
    return dt.date.today() - dt.timedelta(days=DEFAULT_DATE_LAG_DAYS)


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY env var missing")

    ap = argparse.ArgumentParser(description="Generate daily MT-centric arXiv digest.")
    ap.add_argument("date", nargs="?", help="Target UTC date YYYY-MM-DD")
    ap.add_argument("--date", dest="date_flag",
                    type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date())
    ap.add_argument("--max", dest="max_picks", type=int,
                    default=DEFAULT_MAX_PICKS)

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

    print(f"✓ Digest → {md_path.name} | Log → {log_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()