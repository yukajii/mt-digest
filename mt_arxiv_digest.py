#!/usr/bin/env python3
"""
mt_arxiv_digest.py  – daily digest generator for MT‑centric cs.CL papers

❖ New in 2025‑05‑24
───────────────────
* **Engaging Preface.** After selecting papers, we now call a *second* (usually stronger)
  OpenAI model to draft a 2‑to‑3‑sentence intro that:
    ▸ highlights the day’s common themes                      
    ▸ comments on how practical & MT‑relevant the selection is           
    ▸ mentions if the set is weakly related but still useful.
  The text is inserted right under the title in the Markdown.
* **Two‑stage LLM usage recorded** – token counts & raw replies for **both** calls
  go into the per‑run log.
* **Minor** – cost estimate (USD) in the log, cooler CLI via *argparse*.

Usage
─────
$ python mt_arxiv_digest.py                 # yesterday (UTC), keep 5 papers
$ python mt_arxiv_digest.py --date 2025-05-22 --max 7

Produces:
  •  `mt_digest_YYYY‑MM‑DD.md`   – newsletter‑ready Markdown
  •  `logs/mt_digest_YYYY‑MM‑DD.log` – full audit trail
"""

from __future__ import annotations
import argparse, datetime as dt, json, os, pathlib, re, textwrap, warnings
from typing import List, Dict, Tuple
import arxiv, openai

# ── CONSTANTS ────────────────────────────────────────────────────────────────
MAX_RESULTS = 100
DEFAULT_MAX_PICKS = 5
SELECT_MODEL   = "gpt-4o-mini"   # fast / cheap for classification
PREFACE_MODEL  = "gpt-4o"        # higher‑quality style generation
USD_PER_TOKEN  = 0.000005        # adjust to your pricing tier

BASE_DIR = pathlib.Path(__file__).parent
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

warnings.filterwarnings(
    "ignore",
    message=r".*deprecated.*Search\.results.*",
    category=DeprecationWarning,
)

# ── HELPERS ──────────────────────────────────────────────────────────────────

def fetch_cscl(date: dt.date) -> List[Dict]:
    day = date.strftime("%Y%m%d")
    query = f'cat:cs.CL AND submittedDate:[{day}0000 TO {day}2359]'
    search = arxiv.Search(query=query, max_results=MAX_RESULTS,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    client = arxiv.Client()
    papers = []
    for p in client.results(search):
        papers.append({
            "id": p.get_short_id(),
            "title": p.title.strip().replace("\n", " "),
            "abstract": re.sub(r"\s+", " ", p.summary.strip()),
            "url": p.pdf_url,
        })
    return papers


def openai_chat(model: str, messages: List[Dict], temperature: float = 0):
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content, resp.usage.model_dump()


# ── 1️⃣  SELECT PAPERS ───────────────────────────────────────────────────────

def pick_mt_papers(papers: List[Dict], max_picks: int) -> Tuple[List[int], str, Dict]:
    system_msg = "You are an expert researcher in machine translation."
    catalogue = "\n\n".join(
        f"{i+1}. {p['title']}\nAbstract: {p['abstract']}" for i, p in enumerate(papers)
    )
    user_msg = textwrap.dedent(f"""
        From the list below, pick up to {max_picks} papers that are MOST closely related to
        *machine translation* OR the use of *large language models* for MT. Prefer practical
        or implementation‑oriented work over purely theoretical work.
        Reply **only** with a JSON array of the chosen item numbers, e.g. [1,4,5].

        {catalogue}
    """).strip()

    reply, usage = openai_chat(SELECT_MODEL, [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ])

    try:
        picks = json.loads(re.search(r"\[.*?\]", reply, re.S).group())
    except Exception as e:
        raise RuntimeError(f"Could not parse selection reply:\n{reply}") from e

    return picks, reply, usage


# ── 2️⃣  GENERATE PREFACE ────────────────────────────────────────────────────

def draft_preface(date: dt.date, papers: List[Dict], picks: List[int]) -> Tuple[str, str, Dict]:
    chosen = [papers[i-1] for i in picks] if picks else []
    # Build a mini‑catalogue of chosen titles only (keeps tokens low)
    titles_block = "\n".join(f"• {p['title']}" for p in chosen) or "(no MT‑specific papers today)"

    user_msg = textwrap.dedent(f"""
        You are writing the short introduction for a daily Machine Translation (MT) research digest.
        Today is {date.isoformat()}.

        1. Summarise in **2–3 concise sentences** the main common thread(s) of the selected papers below.
        2. Add a brief editorial comment on how practically relevant they are to MT. If they are only
           loosely connected, say so diplomatically and hint why they might still interest MT practitioners.
        Do **not** list the papers again.

        Selected papers:\n{titles_block}
    """).strip()

    reply, usage = openai_chat(PREFACE_MODEL, [
        {"role": "system", "content": "You are a helpful research newsletter editor."},
        {"role": "user",   "content": user_msg},
    ], temperature=0.7)

    return reply.strip(), user_msg, usage


# ── WRITE OUTPUTS ────────────────────────────────────────────────────────────

def write_md(date: dt.date, preface: str, papers: List[Dict], picks: List[int]):
    md = [f"# MT‑related cs.CL papers for {date.isoformat()}", "", preface, ""]
    for idx in picks:
        p = papers[idx-1]
        md += [f"## [{p['title']}]({p['url']})", "", p['abstract'], ""]

    path = BASE_DIR / f"mt_digest_{date.isoformat()}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def write_log(date: dt.date, log: Dict):
    path = LOG_DIR / f"mt_digest_{date.isoformat()}.log"
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY env var missing")

    argp = argparse.ArgumentParser(description="Generate daily MT‑centric arXiv digest.")
    argp.add_argument("--date", type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date(),
                      help="Target UTC date (YYYY-MM-DD). Defaults to yesterday.")
    argp.add_argument("--max", dest="max_picks", type=int, default=DEFAULT_MAX_PICKS,
                      help="Maximum papers to include (default: %(default)s).")
    ns = argp.parse_args()
    target_date = ns.date or (dt.date.today() - dt.timedelta(days=1))

    # 1. Fetch papers
    papers = fetch_cscl(target_date)
    if not papers:
        print("No cs.CL papers on that date.")
        return

    # 2. Select MT‑relevant papers
    picks, select_reply, select_usage = pick_mt_papers(papers, ns.max_picks)

    # 3. Preface generation
    preface, preface_prompt, preface_usage = draft_preface(target_date, papers, picks)

    # 4. Write digest file
    md_path = write_md(target_date, preface, papers, picks)

    # 5. Log everything
    total_tokens = select_usage.get("total_tokens", 0) + preface_usage.get("total_tokens", 0)
    log_dict = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "target_date": target_date.isoformat(),
        "total_papers": len(papers),
        "picked_indices": picks,
        "preface": preface,
        "token_usage": {
            "selection_call": select_usage,
            "preface_call": preface_usage,
            "grand_total": total_tokens,
            "approx_cost_usd": round(total_tokens * USD_PER_TOKEN, 4),
        },
        "selection_reply_raw": select_reply,
        "preface_prompt_sent": preface_prompt,
    }
    log_path = write_log(target_date, log_dict)

    print(f"✓ Digest saved at {md_path.name}  |  Log → {log_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
