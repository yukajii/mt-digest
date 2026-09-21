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
4. Ask `PREFACE_MODEL` for one practitioner takeaway per paper, then for the
   issue intro.

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

Measured over eight announcement batches:

| floor | mean papers/issue | empty issues |
| ----- | ----------------- | ------------ |
| none (old) | 5.0 | 0 |
| z >= 1.8 | 3.2 | 0 |
| **z >= 2.0** | **2.2** | **1 of 8** |
| z >= 2.2 | 1.8 | 1 of 8 |
| z >= 2.5 | 1.4 | 2 of 8 |

Two consequences worth knowing before changing the value:

- **Some batches produce no issue at all.** The 2026-09-02 batch topped out at
  z = 1.86 with everything bunched below it, so nothing clears 2.0 and no `.md`
  is written. The workflow treats that as a quiet day rather than a failure: it
  skips the send and still uploads the run log, which is what you would tune
  the floor from.
- **2.0 is a compromise, not a clean separator.** On 2026-08-25 it still admits
  one paper (z = 2.13, machine-generated-text detection, not MT). Raising to
  2.2 empties that batch correctly but also costs the English-Syriac paper on
  2026-09-16 (z = 2.14), which is a genuine pick. Dropping a real paper was
  judged worse than admitting a marginal one on a rare flat batch.

Override per run with `--min-z`, or change `MIN_RELEVANCE_Z`. Pass a large
negative number to restore the old always-five behaviour.

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
