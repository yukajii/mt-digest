#!/usr/bin/env python3
"""
mt_arxiv_digest.py  – manual daily digest generator for MT‑centric cs.CL papers

Changes 2025‑05‑21
──────────────────
* Uses **OpenAI Python ≥ 1.0** client style.
* Writes a per‑run log file under `logs/` capturing:
  • # cs.CL papers on the date  • prompt & completion token counts  • full prompt we sent  • raw JSON reply  • list of chosen indices.
* `MAX_PICKS` can be overridden from the command line, e.g.  `python mt_arxiv_digest.py 2025‑05‑20 3`.
* Suppresses arXiv deprecation warnings.

Usage
─────
    python mt_arxiv_digest.py               # digest for yesterday (UTC)
    python mt_arxiv_digest.py 2025‑05‑20    # digest for that day
    python mt_arxiv_digest.py 2025‑05‑20 7  # choose up to 7 papers

The script creates two artefacts side‑by‑side:
  1.  `mt_digest_YYYY‑MM‑DD.md`   – digest markdown
  2.  `logs/mt_digest_YYYY‑MM‑DD.log` – run log
"""

from __future__ import annotations
import sys, os, datetime as dt, re, json, pathlib, textwrap, warnings, logging
import arxiv, openai

# ─── CONFIGURABLE CONSTANTS ──────────────────────────────────────────────────
MAX_RESULTS = 100            # cap on API call (rarely >40)
DEFAULT_MAX_PICKS = 5        # default number of papers to keep
OPENAI_MODEL = "gpt-4o-mini" # change if unavailable to you
OUT_DIR = pathlib.Path(__file__).parent
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Silence the arXiv Search deprecation chatter
warnings.filterwarnings("ignore", message=r".*deprecated.*Search\.results.*", category=DeprecationWarning)

# ─── SMALL HELPERS ────────────────────────────────────────────────────────────

def date_arg(idx: int = 1) -> dt.date:
    """Return the date supplied as argv[idx] or yesterday (UTC)."""
    if len(sys.argv) > idx:
        return dt.datetime.strptime(sys.argv[idx], "%Y-%m-%d").date()
    return dt.date.today() - dt.timedelta(days=1)


def max_picks_arg(idx: int = 2) -> int:
    """Return user‑supplied max‑picks or the default."""
    if len(sys.argv) > idx:
        return int(sys.argv[idx])
    return DEFAULT_MAX_PICKS


def fetch_cscl(date: dt.date):
    day = date.strftime("%Y%m%d")
    query = f'cat:cs.CL AND submittedDate:[{day}0000 TO {day}2359]'
    search = arxiv.Search(query=query, max_results=MAX_RESULTS,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    # Use the newer arXiv client to future‑proof
    client = arxiv.Client()
    for p in client.results(search):
        papers.append({
            "id": p.get_short_id(),
            "title": p.title.strip().replace("\n", " "),
            "abstract": re.sub(r"\s+", " ", p.summary.strip()),
            "url": p.pdf_url,
        })
    return papers


def ask_llm_select(papers: list[dict], max_picks: int):
    """Ask the OpenAI model to choose relevant papers.

    Returns (picked_indices, prompt_str, raw_reply_str, usage_dict)
    """
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

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=0,
    )
    reply_txt = resp.choices[0].message.content
    usage = resp.usage.model_dump() if hasattr(resp, "usage") else {}

    try:
        picks = json.loads(re.search(r"\[.*?\]", reply_txt, re.S).group())
    except Exception as e:
        raise RuntimeError(f"Could not parse LLM response:\n{reply_txt}") from e
    return picks, user_msg, reply_txt, usage


def write_md(date: dt.date, picks, papers):
    md_lines = [f"# MT‑related cs.CL papers for {date.isoformat()}", ""]
    for idx in picks:
        p = papers[idx - 1]
        md_lines.append(f"## [{p['title']}]({p['url']})")
        md_lines.append("")
        md_lines.append(p['abstract'])
        md_lines.append("")

    path = OUT_DIR / f"mt_digest_{date.isoformat()}.md"
    path.write_text("\n".join(md_lines), encoding="utf-8")
    return path


def write_log(date: dt.date, log_dict: dict):
    path = LOG_DIR / f"mt_digest_{date.isoformat()}.log"
    with path.open("w", encoding="utf-8") as fh:
        for k, v in log_dict.items():
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False, indent=2)
            fh.write(f"{k}: {v}\n\n")
    return path


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    date = date_arg()
    max_picks = max_picks_arg()

    papers = fetch_cscl(date)
    if not papers:
        print("No cs.CL papers found on that date.")
        return

    picks, prompt_txt, reply_txt, usage = ask_llm_select(papers, max_picks)
    md_path = write_md(date, picks, papers)
    log_path = write_log(date, {
        "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "target_date": date.isoformat(),
        "total_papers": len(papers),
        "picked_indices": picks,
        "openai_usage": usage,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_sent": prompt_txt,
        "reply_received": reply_txt,
    })

    print(f"✓ Digest: {md_path.name}   |   Log: {log_path.relative_to(OUT_DIR)}")


if __name__ == "__main__":
    main()
