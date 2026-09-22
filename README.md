# Yukajii MT Digest

Daily machine-translation (MT) research digest automatically pulled from the
cs.CL section of arXiv, re-ranked for MT relevance, and delivered to your inbox
via [Buttondown](https://buttondown.email).

Published as **[Daily MT Picks](https://buttondown.com/daily-mt-picks/archive/)**.

---

## What this repo contains

| File / Dir                     | Purpose                                                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mt_arxiv_digest.py`           | Fetches one arXiv announcement batch of `cs.CL` pre-prints, embeds them with [*e5-large-v2*](https://huggingface.co/intfloat/e5-large-v2), picks the top-*k* MT-related papers, calls the model in `PREFACE_MODEL` for a 2-3 sentence intro, and writes `mt_digest_YYYY-MM-DD.md` plus a JSON run log. |
| `send_digest.py`               | Posts the generated Markdown to Buttondown via its REST API. Idempotent: a repeat call for an already-queued date is a no-op.                                                    |
| `set_canonical.py`             | Points each Buttondown archive page at its yukajii.com counterpart, but only once that page actually answers 200. Dry run unless `--apply`.                       |
| `sync_to_site.py`              | Merges the e-mail body, the run log's headline and the send receipt into one Markdown file with front matter, for the yukajii.com archive.                        |
| `export_archive.py`            | Read-only backfill: pulls every past issue out of Buttondown. GET requests only.                                                                                 |
| `check_already_sent.py`        | Guard step: inspects the workflow's own artifacts and reports whether this date's digest already went out, so a reattempt run can skip the heavy steps.                          |
| `.github/workflows/digest.yml` | GitHub Actions workflow. Runs at 07:20 UTC with reattempts at 11:20 and 15:20 (or on demand): builds the digest, e-mails it, uploads the Markdown and log as private artifacts, and files an issue if the last reattempt fails. |
| `logs/`                        | JSON run logs, including per-paper relevance scores. Git-ignored; kept 30 days as CI artifacts.                                                                                 |

---

## Quick start (local)

```bash
git clone https://github.com/yukajii/mt-digest.git && cd mt-digest

# Python >=3.11 recommended
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export BUTTONDOWN_TOKEN=...

python mt_arxiv_digest.py             # builds mt_digest_YYYY-MM-DD.md
python send_digest.py mt_digest_YYYY-MM-DD.md
```

The first run downloads the e5-large-v2 weights (~1.3 GB) into your
HuggingFace cache.

### Command-line flags

```text
<date>              positional announcement date, YYYY-MM-DD
--date YYYY-MM-DD   same thing as a flag
--max N             include at most N papers (default: 5)
--min-z Z           relevance floor in std devs above the batch mean (default: 2.0)
--min-picks N       always keep at least N papers, even below the floor (default: 2)
--print-date        print the resolved announcement date and exit
```

The date is an arXiv **announcement day**, not a submission day - see
[Announcement batches](#announcement-batches) below. With no date given the
script targets **today minus `DEFAULT_DATE_LAG_DAYS`** (currently 5), rolled
back to the most recent announcement day. The lag exists because arXiv's
`submittedDate` filter only settles once a batch has been announced. The
`DATE` environment variable is honoured as a fallback, which is how CI passes
the date in.

`--print-date` resolves the date without loading the embedding model or the
OpenAI client, so it returns instantly. It is a local convenience only: the
workflow deliberately computes the same date in bash instead, because the
already-sent guard has to run *before* `pip install`. That means the lag and
the announcement-day rollback are implemented twice, in `resolve_target_date()` and in
the "Determine DATE" step. **Change one and you must change the other.**

---

## Announcement batches

arXiv announces five batches a week and **never on Friday or Saturday**
([policy](https://info.arxiv.org/help/availability.html)). Each closes at
14:00 ET and goes live at 20:00 ET the same day:

| Submitted (ET) | Announced (ET) |
| -------------- | -------------- |
| Mon 14:00 - Tue 14:00 | Tue 20:00 |
| Tue 14:00 - Wed 14:00 | Wed 20:00 |
| Wed 14:00 - Thu 14:00 | Thu 20:00 |
| Thu 14:00 - Fri 14:00 | Sun 20:00 |
| **Fri 14:00 - Mon 14:00** | **Mon 20:00** |

That last row is why this digest works in batches rather than calendar days.
**Saturdays and Sundays carry real cs.CL submissions** - 38 to 56 a day in a
three-week sample - they are just announced together with Friday afternoon and
Monday morning. Treating "arXiv does not announce at the weekend" as "there
are no weekend papers" silently dropped every one of them, about 100 cs.CL
papers a week.

`batch_window()` converts an announcement day into the UTC window it covers,
going through the tz database so the 14:00 ET deadline stays correct across
daylight saving. The windows tile the week with no gaps and no overlaps.

The pool is also simply better. The Monday batch for 2026-09-14 holds 197
papers against 100 for that Monday alone, and its top five score z = 4.29,
3.26, 3.01, 2.97, 2.09 - every one a real MT paper. The same day under
single-day selection bottomed out at z = 1.38, where the fifth pick was a coin
flip against the sixth.

One caveat: arXiv's range filter is inclusive at both ends and minute-granular,
so batch edges are not perfectly clean. Measured over the week of 2026-09-14,
the tiling accounts for 521 of 522 papers. The single straggler is submitted
within a minute of a deadline.

## How papers are chosen

1. Fetch every `cs.CL` paper submitted inside the target announcement batch
   (capped at `MAX_RESULTS`; the script warns if the cap is hit).
2. Embed title + abstract with e5-large-v2.
3. Score each paper against the `CONCEPTS` list, keep the top `--max`, then
   drop anything below the relevance floor (see below).
4. Ask `PREFACE_MODEL` for one practitioner takeaway per paper, then a web
   headline, then the issue intro.

Those three calls come to roughly **$0.004 an issue**, about **$1.10 a year**,
at gpt-5.4-mini's rates ($0.75/1M in, $4.50/1M out). `USD_PER_MTOK_IN` and
`USD_PER_MTOK_OUT` only feed the figure in the run log - update them whenever
`PREFACE_MODEL` changes, or that figure quietly becomes fiction.

Every run writes the top 15 scores to `logs/mt_digest_<date>.log`, so the
ranking can be audited after the fact. Each entry carries a raw cosine score
and a `z` — its distance from that day's mean in standard deviations.

**The floor uses `z`, not the raw score.** e5 cosines sit in a narrow
0.72-0.83 band regardless of how good the batch was, so the absolute number
says almost nothing. The within-batch z-score does.

## Relevance floor

`MIN_RELEVANCE_Z` (default **2.0**) drops any paper scoring less than that many
standard deviations above its own batch's mean, even when it would otherwise
fill one of the five slots. Issue length follows the batch instead of always
padding to five.

This exists because of issues like
[2026-08-25](https://buttondown.com/daily-mt-picks/archive/machine-translation-digest-for-aug-25-2026/),
which shipped five papers - a computational-semantics primer, machine-generated
text detection, a transformer-circuits study, spoken hallucination detection
and repo-scale QA - not one about translation, under a preface obliged to claim
they "shape machine translation systems". Its z-scores were 2.13, 1.91, 1.76,
1.64, 1.53: no signal, just the top of a flat distribution.

Measured over fourteen consecutive announcement batches (2026-09-01 to
2026-09-20):

| floor | mean papers/issue | empty issues |
| ----- | ----------------- | ------------ |
| none (old) | 5.0 | 0 |
| z >= 1.8 | 3.4 | 0 of 14 |
| **z >= 2.0** | **2.6** | **1 of 14** |
| z >= 2.2 | 1.9 | 1 of 14 |
| z >= 2.5 | 1.5 | 3 of 14 |

### Never an empty issue

`MIN_PICKS` (default **2**) overrides the floor from below: the top two papers
are always kept, whatever they score. A couple of loosely relevant papers beats
a missing issue.

The minimum is deliberately small. Over the same fourteen batches:

| `MIN_PICKS` | mean papers/issue | empty issues | batches padded below the floor |
| ----------- | ----------------- | ------------ | ------------------------------ |
| 0 (floor only) | 2.57 | 1 | 0 |
| 1 | 2.64 | 0 | 1 |
| **2** | **2.79** | **0** | **2** |
| 3 | 3.29 | 0 | 7 |

At 3 the minimum would pad half the batches and the floor would stop meaning
much. At 2 it engages twice in fourteen, which is where it is actually needed:
2026-09-02 (nothing above z = 1.86) and 2026-09-06 (one paper above the floor).

A pick admitted this way is flagged `below_floor` in the run log, and the
preface prompt is told how many there were, with an instruction to say so
plainly. That is the guard against the original failure mode - padded picks
written up as though they were a coherent day of MT research.

### One caveat on the value

**2.0 is a compromise, not a clean separator.** On 2026-08-25 it admits one
paper on merit (z = 2.13, machine-generated-text detection, not MT). Raising to
2.2 would reject that batch correctly but also costs the English-Syriac paper
on 2026-09-16 (z = 2.14), a genuine pick. Dropping a real paper was judged
worse than admitting a marginal one on a rare flat batch.

Override per run with `--min-z` and `--min-picks`, or change
`MIN_RELEVANCE_Z` / `MIN_PICKS`. A large negative `--min-z` restores the old
always-five behaviour.

## The web headline

`draft_title()` asks for a headline naming what the issue is actually about -
the task, metric, language or benchmark - and stores it as `title` in the run
log. "Machine Translation Digest for Sep 16 2026" is a filing label; nobody
searches for it. The headline is what becomes the `<h1>` and `<title>` on the
yukajii.com page.

**The e-mail subject is deliberately left alone.** Both the Buttondown slug and
`check_already_sent.py` key off it, so changing it would renumber the archive
and break the reattempt guard.

The title travels in the log rather than in the Markdown, which keeps
`mt_digest_<date>.md` a clean e-mail body with no front matter for Buttondown
to trip over. `_tidy_title()` strips wrapping quotes, a trailing period and a
leading label ("MT Digest: ..."), and rejects anything over fourteen words or
carrying a banned phrase. A rejected or failed title is simply empty, and the
page falls back to a dated heading - the issue is never blocked on a nicety.

## Publishing to yukajii.com

Each issue is staged into [yukajii-site](https://github.com/yukajii/yukajii-site),
which renders `content/mt-digest/` into static pages.

Three files hold the pieces, deliberately kept apart:

| file | carries |
| ---- | ------- |
| `mt_digest_<date>.md` | the e-mail body, no front matter, exactly as Buttondown received it |
| `logs/mt_digest_<date>.log` | the run log, including the web headline |
| `logs/sent_<date>.json` | the send receipt, including the real Buttondown slug |

The receipt exists because **the slug cannot be derived from the subject**.
Buttondown appends a random suffix when a subject repeats, which has happened
six times in the archive (`...-for-jul-24-2026-9525`). `sync_to_site.py` merges
the three into `<publish-date>-<slug>.md` and exits quietly if any piece is
missing, so a quiet batch or a skipped send never fails the build.

**The workflow opens a pull request rather than pushing to the site's default
branch.** That branch deploys, and publishing to a live domain on every send
is a decision to make explicitly rather than inherit from a sync. Merging the
PR is what publishes. To make it automatic instead, replace the branch-and-PR
block with a commit straight onto the default branch.

Requires `SITE_REPO_TOKEN`, a fine-grained PAT with Contents:write and
Pull requests:write on `yukajii/yukajii-site`. Without it the step is a no-op.

## Canonical URLs

Both pages exist on purpose. The Buttondown archive keeps the full abstracts;
the site page leads with the headline and the takeaways. `canonical_url` tells
search engines which is primary so the overlap is consolidated rather than
split between them.

**This is not done at send time.** When an issue sends, its site page does not
exist yet - the workflow stages it as a pull request, and the URL 404s until
that is merged. A canonical pointing at a missing page is worse than none. So
`set_canonical.py` runs separately and sets the canonical only once the page
answers 200, which makes it self-healing: run it again after merging and it
catches up on whatever is now live.

It is read-only unless `--apply` is passed, and the workflow
(`.github/workflows/set-canonical.yml`) is manual-only with `apply` defaulting
to off. It is the one workflow that modifies already-sent emails, so it is
deliberately not scheduled.

The liveness check is by content length, not status code. The site is a
single-page app, so an unknown path still answers 200 with the ~2 kB app
shell; a real issue page is an order of magnitude bigger. A status-code-only
check reported the archive live before it had deployed.

## Issue format

Each paper gets its abstract plus a one-line takeaway aimed at practitioners
rather than reviewers. Abstracts are passed through `clean_abstract()` first,
which turns LaTeX source back into prose - roughly one arXiv abstract in seven
carries markup such as `${<}100$` or `\href{...}{github}` that would otherwise
render literally in the e-mail.

The intro rotates through `PREFACE_ANGLES` by date, and the prompt bans the
openers and stock verbs the archive had settled into ("Today's MT digest
spotlights...", "a common thread"). Both the takeaway call and the preface
call are allowed to fail: a parse error or an API outage costs you the extra
copy, not the issue.

---

## Setting up Buttondown

1. **Create an account** at <https://buttondown.email/register> and finish
   sender-address verification.
2. Go to **Settings -> API**, generate a token, and store it as
   `BUTTONDOWN_TOKEN` (GitHub repo secret, or a local env var).
3. `send_digest.py` posts to whichever newsletter is the default on the account
   that owns the token.

The sender creates the email with `status=about_to_send`, which queues it for
all active subscribers immediately. It is idempotent: re-running it for a date
that is already queued or sent is a no-op, so a retried CI job will not
double-send.

---

## Running automatically on GitHub Actions

Two encrypted repository secrets are required:

| Secret name        | Value                                   |
| ------------------ | --------------------------------------- |
| `OPENAI_API_KEY`   | OpenAI key with access to `PREFACE_MODEL`. |
| `BUTTONDOWN_TOKEN` | The Buttondown API token from above.    |

The job runs on three crons - 07:20, 11:20 and 15:20 UTC - which all resolve
to the *same* target date. The later two are reattempts for when arXiv throttles
the runner's shared IP with an HTTP 429. They are cheap no-ops on a good day:
`check_already_sent.py` inspects the run's artifacts and short-circuits every
heavy step once that date's digest has gone out, and the Buttondown send is
idempotent on top of that.

The job also:

* rolls Friday and Saturday back to Thursday, the last day arXiv announced on
  (Sunday is a real announcement day and is left alone);
* caches the e5 weights, which otherwise cost a ~1.3 GB download three times
  a day;
* uploads `mt_digest_md-YYYY-MM-DD` and `mt_digest_log-YYYY-MM-DD` as
  30-day private artifacts;
* opens (or comments on) a `digest-failure` issue **only when the 15:20
  reattempt fails**. An earlier failure is what the reattempts exist for, so
  reporting it then would be noise; if the last one has also failed, that
  date's digest is genuinely stranded.

No files are pushed back to the repo - digests live in Buttondown and as
ephemeral artifacts.

---

## Contributing

PRs welcome. Better ranking heuristics, alternative embeddings, and richer
per-paper summaries are all fair game.

---

## License

MIT (c) 2025 yukajii / yukajii.com
