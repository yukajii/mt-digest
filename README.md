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
| `arxiv_schedule.py`            | The announcement-day rules and batch windows. Standard library only, so `pick_batch.py` can use it before `pip install`.                                                         |
| `pick_batch.py`                | Chooses which batch to digest: the oldest one that has aged enough and has no sent-artifact.                                                                                     |
| `check_already_sent.py`        | Whether a given date's digest already went out, inferred from its artifact.                                                                                                     |
| `.github/workflows/digest.yml` | GitHub Actions workflow. Runs at 11:20 UTC with reattempts at 15:20 and 19:20 (or on demand): builds the digest, e-mails it, uploads the Markdown and log as private artifacts, and files an issue if the last reattempt fails. |
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
[Announcement batches](#announcement-batches) below. Run by hand with no date,
the script falls back to **today minus `DEFAULT_DATE_LAG_DAYS`** (1), rolled
back to the most recent announcement day, so a bare `python mt_arxiv_digest.py`
lands on roughly the batch CI would be working on. The `DATE` environment
variable is honoured as a fallback, which is how the workflow passes in the
date `pick_batch.py` chose.

`--print-date` resolves the date without loading the embedding model or the
OpenAI client, so it returns instantly.

In CI the date does not come from this arithmetic at all - `pick_batch.py`
picks the oldest batch still outstanding. See
[Which batch gets sent](#which-batch-gets-sent).

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

## Which batch gets sent

`pick_batch.py` picks the **oldest** announcement batch that has aged at least
`MIN_AGE_DAYS` and has no `mt_digest_md-<DATE>` artifact. It runs before
`pip install`, so it and `arxiv_schedule.py` are standard library only.

This replaced arithmetic of the form "today minus five, rolled back to an
announcement day", which had two faults:

- **It went quiet mid-week.** Friday and Saturday both roll back to Thursday,
  so every Wednesday and Thursday run re-targeted a batch already sent. On
  2026-09-23 all three runs resolved to 2026-09-17 and skipped, while five
  announced batches sat unsent behind them.
- **It lost batches.** The Sunday, Monday, Tuesday and Wednesday batches each
  got exactly one run-day. If those three crons all failed, that batch was
  never targeted again.

Picking from what is outstanding fixes both: a run only idles when there is
genuinely nothing to send, and a failed batch is simply still outstanding the
next day.

### Why `MIN_AGE_DAYS` is 1

arXiv publishes in **discrete batches, not continuously**. Five times a week,
at 20:00 ET, a whole batch goes live at once. So the wait a digest needs is
hours past an announcement, not days past a submission - which is what an
earlier five-day lag was really compensating for.

Announcement days are Sunday to Thursday, so sending one day later lands on
Monday to Friday:

| email | batch | covers papers submitted |
| ----- | ----- | ----------------------- |
| Mon | Sun batch | Thu 14:00 - Fri 14:00 ET |
| Tue | Mon batch | **Fri 14:00 - Mon 14:00 ET** (the weekend, 150-200 papers) |
| Wed | Tue batch | Mon 14:00 - Tue 14:00 ET |
| Thu | Wed batch | Tue 14:00 - Wed 14:00 ET |
| Fri | Thu batch | Wed 14:00 - Thu 14:00 ET |

Weekday issues, quiet weekends, and nothing older than a day past its
announcement. Tuesday's is the big one, because arXiv welds Friday afternoon,
both weekend days and Monday morning into a single announcement - there is no
way to give the weekend its own issue without inventing a split arXiv does not
make.

### Cron times are part of the safety margin

The margin is not in `MIN_AGE_DAYS`, it is in when the crons fire. A batch is
announced at 20:00 ET, which is 00:00 UTC the next day (01:00 in winter), and
the API takes a few hours to reflect it. Measured on 2026-09-24:

```
batch           announced (UTC)   hours ago   papers
2026-09-23 Wed  09-24 00:00            10.2       92
2026-09-24 Thu  09-25 00:00           -13.8        0   <- not announced yet
```

Fully indexed at 10.2 hours, nothing at all before announcement. The crons run
at 11:20, 15:20 and 19:20 UTC, so the first is ~11 hours past announcement,
inside the proven range.

**Do not move the crons earlier without re-measuring.** A partially indexed
batch would ship as a short issue and be marked sent, and the relevance floor
would make it look like a legitimately thin day.

### The lookback window

`LOOKBACK_DAYS` is 14 and **must stay well inside the 30-day artifact
retention** in `digest.yml`. "Sent" is inferred from the artifact, so once one
expires its batch looks outstanding again - a lookback near the retention
window would quietly re-send month-old issues.

If the artifact API cannot be reached, `pick_batch.py` stands down rather than
guessing. Treating an outage as "not sent" would re-send a batch subscribers
already have.

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
the three into `<announcement-date>-<slug>.md` - the date in the page URL,
rather than the send date, which a freshly created email does not have yet -
and exits quietly if any piece is missing, so a quiet batch or a skipped send
never fails the build.

**The workflow commits straight to the site's default branch, which deploys.**
It used to open a pull request instead, so that publishing to a live domain was
a deliberate act rather than something inherited from wiring up a sync. A week
of real running settled that: five issues a week meant five pull requests a
week, and the site fell behind - six were sitting unmerged before anyone
noticed. The e-mail already goes out unreviewed, so a review gate on the web
copy was buying nothing.

The push retries three times, rebasing on `master` between attempts. Three
crons a day can overlap with each other or with a hand-run, and a rejected
push should not fail an issue that has already been e-mailed.

**The sent-marker is uploaded before this step, not after.** The
`mt_digest_md-<DATE>` artifact means "the e-mail went out", which is the
irreversible part, so it has to land immediately after the send. It used to sit
after the publish, which meant a failed publish left a sent issue looking
unsent: the queue picked it up again and rebuilt it three times a day for the
whole lookback window. Buttondown's duplicate detection stopped a second
e-mail, but the marker's one invariant - artifact means sent - was false.

The trade is that a publish failure is no longer retried automatically, since
the batch is correctly marked done. Recover by dispatching the workflow with
that explicit date: `pick_batch.py` honours an explicit date even when the
batch is already sent, `send_digest.py` no-ops on the duplicate and rewrites
the receipt, and the publish then succeeds. The failed run shows red, and the
19:20 cron files a `digest-failure` issue.

**Pushing is not publishing.** Cloudflare still has to build, and a failed
build is silent - the site just stops keeping up, which is how six issues once
went unnoticed. So the step waits up to six minutes for the page to answer
before calling itself done, and fails the run if it never does.

The check measures **response size, not status code**. The site is a
single-page app and answers 200 with a ~2 kB shell for any unknown path, so a
status-code check would pass on a deploy that never happened. Measured live: a
real issue page is 7-12 kB, the shell is 2,104 B, and the threshold is 4 kB.

### Empty batches

If arXiv returns no papers at all, no issue is written and no sent-marker
exists - so `pick_batch.py` would pick that batch again on every run and
**block every newer batch behind it** until it aged out of the lookback. One
bare batch would stall the newsletter for a fortnight.

The workflow writes a separate `mt_digest_empty-<DATE>` artifact, which
`pick_batch.py` also treats as "dealt with". It is deliberately a different
name from the sent-marker, because nothing was sent.

It is only written once the batch is **three days old**. arXiv defers
announcements for holidays and a deferred batch fills in a day or two, so a
freshly empty batch stays queued and gets retried; only a persistently bare
one is written off. An empty batch costs almost nothing to retry in the
meantime - `mt_arxiv_digest.py` returns before the embedding step, so it is a
fetch, not a full run.

Requires `SITE_REPO_TOKEN`, a fine-grained PAT with Contents:write on
`yukajii/yukajii-site`. Without it the step is a no-op.

## Canonical URLs

> **Setting `canonical_url` does not add a meta tag. It makes Buttondown 302
> the issue's archive page to that URL.** The
> [documentation](https://docs.buttondown.com/canonical-url) says the value
> "appears in the meta tags of the email's HTML". Measured against a control
> on 2026-09-22:
>
> ```
> Sep 17 (canonical set)  HTTP 302  ->  yukajii.com/mt-digest/2026-09-17/
> Sep 16 (no canonical)   HTTP 200  30,624B
> ```
>
> So this is not a hint to search engines, it retires the Buttondown archive
> page for that issue. Only the yukajii.com copy stays readable. That is the
> intended outcome here - one public page per issue - but it is a bigger
> action than the name suggests, so do not run this expecting a meta tag.

The newsletter root and the archive index are unaffected, because the
canonical is per-email. Only individual issue pages redirect.

Because the archive page stops serving, an issue page must not link back to
it: that link would return the reader to the page they are already on. The
footer links to the newsletter root instead.

**This is not done at send time.** When an issue sends, its site page does not
exist yet: the commit lands moments later and Cloudflare still has to build,
so the URL 404s for a few minutes. Redirecting an archive page at a missing
page would be worse than doing nothing. So `set_canonical.py` runs separately
and acts only once the page answers 200, which makes it self-healing: run it
again later and it catches up on whatever is now live.

It is read-only unless `--apply` is passed, and the workflow
(`.github/workflows/set-canonical.yml`) is manual-only with `apply` defaulting
to off. It is the one workflow that modifies already-sent emails, and given
what it actually does, it is deliberately not scheduled.

The liveness check is by content length, not status code. The site is a
single-page app, so an unknown path still answers 200 with the ~2 kB app
shell; a real issue page is an order of magnitude bigger. A status-code-only
check reported the archive live before it had deployed - had this script used
one, it would have redirected every issue at the app shell.

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

The job runs on three crons - 11:20, 15:20 and 19:20 UTC - which all resolve
through `pick_batch.py`. The later two are reattempts for when arXiv throttles
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
* opens (or comments on) a `digest-failure` issue **only when the 19:20
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
