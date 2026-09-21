#!/usr/bin/env python3
"""
mt_arxiv_digest.py  - daily digest generator for MT-centric cs.CL papers
Updated for OpenAI GPT-5 + Responses API
"""

from __future__ import annotations

import argparse, datetime as dt, json, os, pathlib, random, re, textwrap, warnings, time
from typing import List, Dict, Tuple
from zoneinfo import ZoneInfo

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

# Relevance floor, in standard deviations above the batch's own mean score.
# Anything below this is dropped even if it would otherwise make the top 5,
# so a thin batch yields a short issue instead of five padded slots. See
# "Relevance floor" in the README for the measurements behind the value.
MIN_RELEVANCE_Z = 2.0

# ...but never ship an empty issue. A couple of loosely relevant papers beat
# nothing at all, so the top MIN_PICKS are kept whatever their score. Over
# fourteen batches this padded below the floor only twice, where raising it
# to three would have padded seven - which would defeat the floor entirely.
MIN_PICKS = 2

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


# -- ABSTRACT CLEAN-UP ---------------------------------------------------
# arXiv abstracts are LaTeX source, so roughly one in six carries markup that
# renders as literal noise in an e-mail: "${<}100$", "8$\times$ more",
# "\href{...}{github}". These turn it back into prose.
_MATH_SYMBOLS = {
    r"\times": "x", r"\sim": "~", r"\approx": "~", r"\pm": "+/-",
    r"\leq": "<=", r"\le": "<=", r"\geq": ">=", r"\ge": ">=",
    r"\neq": "!=", r"\ll": "<<", r"\gg": ">>",
    r"\rightarrow": "->", r"\to": "->", r"\leftarrow": "<-",
    r"\cdot": ".", r"\ldots": "...", r"\dots": "...", r"\infty": "inf",
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\Delta": "Delta", r"\epsilon": "epsilon",
    r"\lambda": "lambda", r"\mu": "mu", r"\sigma": "sigma", r"\theta": "theta",
}

_TEXT_CMD = re.compile(
    r"\\(?:textbf|textit|textsf|textsc|texttt|textrm|textnormal"
    r"|emph|text|mathrm|mathbf|mathit|mathsf|mathcal|mbox|underline)\s*\{([^{}]*)\}"
)
_FONT_GROUP = re.compile(r"\{\\(?:sf|bf|it|tt|rm|em|sc)\s+([^{}]*)\}")
_BIBTEX_BRACES = re.compile(r"\{([^{}\\]*)\}")


def _unwrap(pattern: re.Pattern, text: str, passes: int = 6) -> str:
    """Apply an innermost-brace pattern until it stops matching.

    Formatting commands nest -- \\textbf{\\textsf{PACE}} is real arXiv input --
    and a single pass leaves the outer command stranded.
    """
    for _ in range(passes):
        new = pattern.sub(r"\1", text)
        if new == text:
            break
        text = new
    return text


def _demath(expr: str) -> str:
    """Render the inside of an inline-math span as plain text."""
    for cmd, sym in _MATH_SYMBOLS.items():
        expr = expr.replace(cmd, sym)
    expr = _unwrap(_TEXT_CMD, expr)
    expr = expr.replace("{,}", ",").replace(r"\,", "").replace(r"\;", " ")
    expr = expr.replace("{", "").replace("}", "")
    expr = re.sub(r"\\([a-zA-Z]+)", r"\1", expr)   # unknown command: keep the name
    return expr.strip()


def clean_abstract(text: str) -> str:
    """Strip LaTeX markup that would otherwise render literally in the e-mail."""
    t = text
    t = re.sub(r"\\href\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"[\2](\1)", t)
    t = re.sub(r"\\(?:url|nolinkurl)\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"~?\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", t)
    t = re.sub(r"\\(?:footnote|label|ref|eqref)\s*\{[^{}]*\}", "", t)

    # Inline and display math.
    t = re.sub(r"\$\$(.+?)\$\$", lambda m: _demath(m.group(1)), t, flags=re.S)
    t = re.sub(r"\$([^$]*)\$", lambda m: _demath(m.group(1)), t)

    t = _unwrap(_TEXT_CMD, t)
    t = _unwrap(_FONT_GROUP, t)
    t = _unwrap(_BIBTEX_BRACES, t)                 # BibTeX capitalisation braces
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"\\([%&_#${}])", r"\1", t)         # escaped specials
    t = t.replace("\\ ", " ")                      # escaped space, as in "vs.\ "
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)          # any command left over
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# -- ARXIV ANNOUNCEMENT SCHEDULE -----------------------------------------
# https://info.arxiv.org/help/availability.html
#
# arXiv announces five batches a week, never on Friday or Saturday. Each
# batch closes at 14:00 ET and goes live at 20:00 ET the same day:
#
#   submitted Mon 14:00 - Tue 14:00  ->  announced Tue 20:00
#   submitted Tue 14:00 - Wed 14:00  ->  announced Wed 20:00
#   submitted Wed 14:00 - Thu 14:00  ->  announced Thu 20:00
#   submitted Thu 14:00 - Fri 14:00  ->  announced Sun 20:00
#   submitted Fri 14:00 - Mon 14:00  ->  announced Mon 20:00   <- weekend
#
# The last row is why this module works in batches rather than calendar
# days. Saturdays and Sundays carry real cs.CL submissions (38-56 a day in
# a three-week sample); they are simply announced together with Friday
# afternoon and Monday morning. Digesting one calendar day at a time, and
# skipping the days arXiv does not *announce* on, silently dropped every
# weekend submission - roughly 100 cs.CL papers a week.
ARXIV_ET = ZoneInfo("America/New_York")
ARXIV_DEADLINE_HOUR = 14          # 14:00 ET, the submission cut-off

# announcement weekday -> (window start, window end) as day offsets from it.
# Friday (4) and Saturday (5) are absent: arXiv announces nothing on those.
_BATCH_SPAN = {
    0: (-3, 0),    # Mon announces Fri 14:00 -> Mon 14:00
    1: (-1, 0),    # Tue announces Mon 14:00 -> Tue 14:00
    2: (-1, 0),    # Wed announces Tue 14:00 -> Wed 14:00
    3: (-1, 0),    # Thu announces Wed 14:00 -> Thu 14:00
    6: (-3, -2),   # Sun announces Thu 14:00 -> Fri 14:00
}


def is_announcement_day(day: dt.date) -> bool:
    return day.weekday() in _BATCH_SPAN


def previous_announcement_day(day: dt.date) -> dt.date:
    """Roll back to the most recent day arXiv actually announced on."""
    while not is_announcement_day(day):
        day -= dt.timedelta(days=1)
    return day


def batch_window(announce_day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    """UTC [start, end] of the submission window announced on `announce_day`.

    Deadlines are wall-clock 14:00 ET, so the UTC offset shifts with US
    daylight saving. Converting through the tz database keeps the window
    correct across the March and November transitions.
    """
    if not is_announcement_day(announce_day):
        raise ValueError(
            f"{announce_day} ({announce_day:%A}) is not an arXiv announcement day"
        )

    start_off, end_off = _BATCH_SPAN[announce_day.weekday()]
    deadline = dt.time(ARXIV_DEADLINE_HOUR, 0)

    start_et = dt.datetime.combine(
        announce_day + dt.timedelta(days=start_off), deadline, tzinfo=ARXIV_ET
    )
    end_et = dt.datetime.combine(
        announce_day + dt.timedelta(days=end_off), deadline, tzinfo=ARXIV_ET
    )
    # The window is half-open in real time: (previous deadline, this one].
    # arXiv's range filter is inclusive at both ends and minute-granular, so
    # the start is nudged a minute forward rather than the end a minute back.
    # Measured over the week of 2026-09-14: nudging the start loses 1 paper
    # in 522 at the boundaries, nudging the end loses 6 -- submissions spike
    # in the final minute before the deadline, so that minute must land
    # inside a batch, not between two.
    start_et += dt.timedelta(minutes=1)

    return (start_et.astimezone(dt.timezone.utc),
            end_et.astimezone(dt.timezone.utc))


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


def fetch_cscl(window: Tuple[dt.datetime, dt.datetime],
               max_retries: int = ARXIV_MAX_RETRIES) -> List[Dict]:
    start, end = window
    q = ('cat:cs.CL AND submittedDate:'
         f'[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]')
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
                    # abs page, not the raw PDF; entry_id comes back as http
                    "url": p.entry_id.replace("http://", "https://", 1),
                })
            if len(papers) >= MAX_RESULTS:
                print(f"[warn] hit MAX_RESULTS ({MAX_RESULTS}) for "
                      f"{start:%Y-%m-%d %H:%M}Z..{end:%Y-%m-%d %H:%M}Z - "
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


def rank_mt_papers(papers: List[Dict], max_picks: int,
                   min_z: float = MIN_RELEVANCE_Z,
                   min_picks: int = MIN_PICKS) -> Tuple[List[int], List[Dict]]:
    """Return (1-based picked indices, per-paper ranking detail).

    Cosine similarity against the blended concept vector, e5 prefixes on both
    sides, then three cuts: keep at most `max_picks`, drop anything below
    `min_z`, but always keep at least `min_picks` so a flat batch still
    produces an issue. Detail entries carry `below_floor` so the log, and the
    preface, can tell a padded pick from an earned one.
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

    def zscore(i) -> float:
        return round((float(scores[i]) - mean) / std, 3) if std else 0.0

    n_above = sum(1 for i in order[:max_picks] if zscore(i) >= min_z)
    n_keep = min(max_picks, max(n_above, min_picks), len(order))

    detail = [
        {
            "rank": rank,
            "index": int(i) + 1,
            "arxiv_id": papers[i]["id"],
            "title": papers[i]["title"],
            "score": round(float(scores[i]), 4),
            # z-score against the same batch: raw e5 cosines sit in a very
            # narrow band (0.72-0.83) whatever the day held, so only the
            # relative figure carries usable signal.
            "z": zscore(i),
            "picked": rank <= n_keep,
            "below_floor": rank <= n_keep and zscore(i) < min_z,
        }
        for rank, i in enumerate(order, start=1)
    ]

    picks = [int(i) + 1 for i in order[:n_keep]]

    padded = n_keep - n_above
    if n_keep < max_picks:
        print(f"[info] relevance floor z>={min_z} kept {n_above} of "
              f"{max_picks} candidate slots")
    if padded > 0:
        print(f"[info] batch is thin: padding to the {min_picks}-paper minimum "
              f"with {padded} paper(s) below the floor")
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


# -- EDITORIAL -----------------------------------------------------------
EDITOR_SYSTEM = (
    "You edit Daily MT Picks, a digest read by localisation engineers, MT "
    "researchers and translation-industry practitioners. They know the field: "
    "do not explain what BLEU or post-editing is. Write plainly, in the register "
    "of a knowledgeable colleague, never marketing copy."
)

# Every issue in the archive opens "Today's MT digest highlights/spotlights...".
# Rotating the opening move by date breaks that groove without needing state.
PREFACE_ANGLES = [
    "Open by naming the single most consequential or surprising finding in the set.",
    "Open with the question these papers are collectively circling.",
    "Open by characterising what kind of day this was for MT on arXiv - dense, thin, evaluation-heavy, dominated by one language family, whatever actually fits.",
    "Open with one concrete result, number or benchmark drawn from a specific paper.",
    "Open with what is at stake here for someone actually shipping translation systems.",
    "Open by naming a tension or disagreement between two of the papers.",
    "Open with the shift in what researchers appear to be measuring or optimising for.",
]

BANNED_OPENERS = ("today", "this digest", "in today", "this week", "welcome")


def draft_preface(date: dt.date, papers: List[Dict], picks: List[int],
                  n_below_floor: int = 0):
    chosen = [papers[i - 1] for i in picks] if picks else []
    block = "\n\n".join(
        f"- {p['title']}\n  {clean_abstract(p['abstract'])[:600]}" for p in chosen
    ) or "(no MT-specific papers today)"

    angle = PREFACE_ANGLES[date.toordinal() % len(PREFACE_ANGLES)]

    # When the batch was too thin to fill the issue on merit, say so. The
    # alternative is the Aug 25 failure mode: padded picks written up as
    # though they were a coherent day of MT research.
    thin_note = ""
    if n_below_floor:
        thin_note = (
            f"\n        This batch was thin: {n_below_floor} of the "
            f"{len(chosen)} papers below scored under our relevance bar and "
            "are included only so the issue is not empty. Say so in passing, "
            "plainly and without apologising, and do not imply the set hangs "
            "together better than it does.\n"
        )

    user_msg = textwrap.dedent(f"""
        Write the introduction for the {date.isoformat()} issue. Two or three
        sentences, no heading, no list, no sign-off.

        {angle}
        {thin_note}
        Hard constraints:
        - Do not begin with "Today", "This digest", "In today's", or any variant.
          Vary the sentence shape from issue to issue.
        - Do not use the words "spotlights", "highlights", "showcases", "delves",
          "landscape" or "a common thread".
        - Name specifics. Prefer the actual language pair, metric, benchmark or
          number over abstractions like "advances in evaluation".
        - Do not claim the papers are about machine translation if they are not.
          If the day's selection is mostly adjacent NLP work, say so plainly -
          a thin day is worth reporting as a thin day.
        - Do not re-list the paper titles.

        Papers in this issue:
        {block}
    """).strip()

    reply, usage = openai_chat(PREFACE_MODEL, EDITOR_SYSTEM, user_msg)

    if reply.lower().lstrip("*_# ").startswith(BANNED_OPENERS):
        print(f"[warn] preface still opens with a banned phrase: {reply[:60]!r}")

    return reply, user_msg, usage


def _extract_json(text: str):
    """Parse a JSON object out of a model reply, tolerating code fences."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    candidates = [stripped]

    outermost = re.search(r"\{.*\}", stripped, re.S)
    if outermost:
        candidates.append(outermost.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _tidy_takeaway(text: str) -> str:
    """Normalise one takeaway.

    Terminal punctuation is asked for in the prompt but enforced here: it is
    the one rule that can be fixed deterministically, and the first live run
    came back with no full stops at all on any of the five.
    """
    t = " ".join(text.split()).strip()
    t = t.strip('"').strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


def draft_takeaways(papers: List[Dict], picks: List[int]):
    """One practitioner-facing line per paper.

    Returns (list aligned to picks with None where it failed, usage dict).
    A failure here must not block the issue, so every error path degrades to
    "no takeaway" rather than raising.
    """
    chosen = [papers[i - 1] for i in picks]
    listing = "\n\n".join(
        f"[{n}] {p['title']}\n{clean_abstract(p['abstract'])[:1400]}"
        for n, p in enumerate(chosen, 1)
    )

    user_msg = textwrap.dedent(f"""
        For each paper below, write one sentence saying what a working
        translation or localisation practitioner should take from it.

        Rules:
        - One complete, grammatical sentence in the present tense, ending in a
          full stop. Up to 35 words. If it will not fit, drop a detail rather
          than dropping articles or verbs - it must read as English prose, not
          as compressed notes.
        - Concrete: name the method, the number, or the limitation that matters.
        - No hype, no "this paper shows", no restating the title.
        - A paper counts as being about translation if it touches machine
          translation, human translation, interpreting, localisation,
          subtitling, or the evaluation of any of these. That includes
          low-resource and sign-language translation, and it includes papers
          whose contribution is only a corpus, benchmark or baseline for them.
          Never describe such a paper as adjacent.
        - Only when a paper has no translation component at all, open by naming
          what it actually is, then say why an MT practitioner might still care.

        Return only JSON of this exact shape:
        {{"takeaways": [{{"n": 1, "text": "..."}}, {{"n": 2, "text": "..."}}]}}

        Papers:
        {listing}
    """).strip()

    empty = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        reply, usage = openai_chat(PREFACE_MODEL, EDITOR_SYSTEM, user_msg)
    except Exception as e:                      # noqa: BLE001 - never block the send
        print(f"[warn] takeaway call failed, shipping without them: {e}")
        return [None] * len(chosen), empty

    parsed = _extract_json(reply)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("takeaways"), list):
        print(f"[warn] could not parse takeaways, shipping without them: {reply[:120]!r}")
        return [None] * len(chosen), usage

    by_n = {}
    for item in parsed["takeaways"]:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            try:
                by_n[int(item["n"])] = _tidy_takeaway(item["text"])
            except (TypeError, ValueError):
                continue

    out = [by_n.get(n) for n in range(1, len(chosen) + 1)]
    missing = [n for n, t in enumerate(out, 1) if not t]
    if missing:
        print(f"[warn] no takeaway returned for paper(s) {missing}")
    return out, usage


# -- OUTPUT --------------------------------------------------------------
def write_md(date: dt.date, preface: str,
             papers: List[Dict], picks: List[int],
             takeaways: List[str] | None = None,
             window: Tuple[dt.datetime, dt.datetime] | None = None):

    md: List[str] = [
        preface.strip(),
        "",
        "---",
        "",
    ]

    for n, idx in enumerate(picks):
        if n:
            md += ["---", ""]

        p = papers[idx - 1]
        md += [f"## [{p['title']}]({p['url']})", ""]

        # The takeaway carries no label: five identical "Why it matters:"
        # headers per issue, every day, is its own kind of repetition.
        takeaway = takeaways[n] if takeaways else None
        if takeaway:
            md += [f"**{takeaway}**", ""]

        md += [clean_abstract(p["abstract"]), ""]

    # Say which submission window this covers. An issue keyed to a Monday
    # announcement spans the previous Friday afternoon through Monday
    # morning, and a reader should not have to guess that.
    if window:
        start, end = window
        # %-d is a glibc extension and blows up on Windows, so build the
        # day number by hand and keep the script runnable locally.
        def _long(d: dt.date) -> str:
            return f"{d.day} {d:%B %Y}"

        span = (_long(start.date()) if start.date() == end.date()
                else f"{start.day} {start:%B} to {_long(end.date())}")
        md += [
            "---",
            "",
            f"*Papers announced by arXiv on {date:%A}, {_long(date)}, "
            f"covering submissions from {span}.*",
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
    """Resolve the arXiv *announcement* day this run should digest.

    An explicit date is taken at face value and rolled back if it is not an
    announcement day, so `mt_arxiv_digest.py 2026-09-19` (a Saturday) still
    does something sensible rather than raising.
    """
    explicit = None
    if cli_pos:
        explicit = dt.datetime.strptime(cli_pos, "%Y-%m-%d").date()
    elif cli_flag:
        explicit = cli_flag
    elif env_var:
        explicit = dt.datetime.strptime(env_var, "%Y-%m-%d").date()

    if explicit is not None:
        return previous_announcement_day(explicit)

    # The lag covers the gap between a batch closing and the API indexing it.
    # Worst case is a Thursday close, announced Sunday 20:00 ET, so four days
    # is the true minimum and five leaves a day of slack for holidays.
    return previous_announcement_day(
        dt.date.today() - dt.timedelta(days=DEFAULT_DATE_LAG_DAYS)
    )


def main():
    ap = argparse.ArgumentParser(description="Generate daily MT-centric arXiv digest.")
    ap.add_argument("date", nargs="?", help="Target UTC date YYYY-MM-DD")
    ap.add_argument("--date", dest="date_flag",
                    type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date())
    ap.add_argument("--max", dest="max_picks", type=int,
                    default=DEFAULT_MAX_PICKS)
    ap.add_argument("--min-picks", dest="min_picks", type=int,
                    default=MIN_PICKS,
                    help="Always include at least this many papers, even "
                         f"below the floor (default: {MIN_PICKS}).")
    ap.add_argument("--min-z", dest="min_z", type=float,
                    default=MIN_RELEVANCE_Z,
                    help="Relevance floor in standard deviations above the "
                         f"batch mean (default: {MIN_RELEVANCE_Z}). "
                         "Pass a large negative number to disable.")
    ap.add_argument("--print-date", action="store_true",
                    help="Print the resolved announcement date and exit.")

    ns = ap.parse_args()

    target_date = resolve_target_date(ns.date, ns.date_flag, os.getenv("DATE"))

    if ns.print_date:
        print(target_date.isoformat())
        return

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY env var missing")

    window = batch_window(target_date)
    print(f"[info] {target_date} ({target_date:%A}) announcement batch: "
          f"{window[0]:%Y-%m-%d %H:%M}Z .. {window[1]:%Y-%m-%d %H:%M}Z")

    papers = fetch_cscl(window)
    if not papers:
        print(f"No cs.CL papers in the {target_date} batch - nothing to send.")
        return

    picks, ranking = rank_mt_papers(papers, ns.max_picks, ns.min_z, ns.min_picks)
    if not picks:
        # Only reachable when the batch itself came back empty, since the
        # minimum keeps at least MIN_PICKS whenever there is anything to keep.
        print(f"[info] the {target_date} batch yielded no rankable papers "
              "- no issue written.")
        write_log(target_date, {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "target_date": target_date.isoformat(),
            "total_papers": len(papers),
            "skipped": "no rankable papers",
            "min_relevance_z": ns.min_z,
            "ranking_top_15": ranking[:15],
        })
        return

    n_below_floor = sum(1 for r in ranking if r.get("below_floor"))

    takeaways, takeaway_usage = draft_takeaways(papers, picks)
    preface, preface_prompt, preface_usage = draft_preface(
        target_date, papers, picks, n_below_floor)
    md_path = write_md(target_date, preface, papers, picks, takeaways, window)

    total_in = (preface_usage.get("input_tokens", 0)
                + takeaway_usage.get("input_tokens", 0))
    total_out = (preface_usage.get("output_tokens", 0)
                 + takeaway_usage.get("output_tokens", 0))
    approx_cost = (total_in / 1_000_000 * USD_PER_MTOK_IN
                   + total_out / 1_000_000 * USD_PER_MTOK_OUT)

    log_dict = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "target_date": target_date.isoformat(),
        "total_papers": len(papers),
        "picked_indices": picks,
        "picked_scores": [r["score"] for r in ranking if r["picked"]],
        "picked_z": [r["z"] for r in ranking if r["picked"]],
        "min_relevance_z": ns.min_z,
        "min_picks": ns.min_picks,
        "n_picked": len(picks),
        "n_below_floor": n_below_floor,
        "ranking_top_15": ranking[:15],
        "takeaways": takeaways,
        "token_usage": {
            "preface_call": preface_usage,
            "takeaway_call": takeaway_usage,
            "grand_total": (preface_usage.get("total_tokens", 0)
                            + takeaway_usage.get("total_tokens", 0)),
            "approx_cost_usd": round(approx_cost, 6),
        },
        "preface_prompt_sent": preface_prompt,
    }

    log_path = write_log(target_date, log_dict)

    print(f"[ok] Digest -> {md_path.name} | Log -> {log_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
