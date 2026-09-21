# Yukajii MT Digest

Daily machine-translation (MT) research digest automatically pulled from the
cs.CL section of arXiv, re-ranked for MT relevance, and delivered to your inbox
via [Buttondown](https://buttondown.email).

Published as **[Daily MT Picks](https://buttondown.com/daily-mt-picks/archive/)**.

---

## What this repo contains

| File / Dir                     | Purpose                                                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mt_arxiv_digest.py`           | Fetches one day of `cs.CL` pre-prints, embeds them with [*e5-large-v2*](https://huggingface.co/intfloat/e5-large-v2), picks the top-*k* MT-related papers, calls the model in `PREFACE_MODEL` for a 2-3 sentence intro, and writes `mt_digest_YYYY-MM-DD.md` plus a JSON run log. |
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
<date>              positional UTC date, YYYY-MM-DD
--date YYYY-MM-DD   same thing as a flag
--max N             include at most N papers (default: 5)
--print-date        print the resolved target date and exit
```

With no date given, the script targets **today minus `DEFAULT_DATE_LAG_DAYS`**
(currently 5), rolling back to Friday if that lands on a Saturday or Sunday.
The lag exists because arXiv's `submittedDate` filter only settles once papers
have been announced. The `DATE` environment variable is honoured as a fallback,
which is how CI passes the date in.

`--print-date` resolves the date without loading the embedding model or the
OpenAI client, so it returns instantly. It is a local convenience only: the
workflow deliberately computes the same date in bash instead, because the
already-sent guard has to run *before* `pip install`. That means the lag and
the weekend rollback are implemented twice, in `resolve_target_date()` and in
the "Determine DATE" step. **Change one and you must change the other.**

---

## How papers are chosen

1. Fetch every `cs.CL` paper whose `submittedDate` falls on the target day
   (capped at `MAX_RESULTS`; the script warns if the cap is hit).
2. Embed title + abstract with e5-large-v2.
3. Score each paper against the `CONCEPTS` list and keep the top `--max`.
4. Ask `PREFACE_MODEL` for one practitioner takeaway per paper, then for the
   issue intro.

Every run writes the top 15 scores to `logs/mt_digest_<date>.log`, so the
ranking can be audited after the fact. Each entry carries a raw cosine score
and a `z` — its distance from that day's mean in standard deviations.

**Use `z`, not the raw score, for any relevance threshold.** e5 cosines sit in
a narrow 0.72-0.83 band regardless of how good the day was, so the absolute
number says almost nothing. The within-day z-score does: measured over four
archive days, the top paper scored z=+3.91 and z=+3.50 on days with real MT
work, versus z=+2.13 on a day with none.

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

* rolls a weekend target date back to Friday rather than skipping it;
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
