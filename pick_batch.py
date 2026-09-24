#!/usr/bin/env python3
"""
pick_batch.py [--min-age N] [--lookback N] [--date YYYY-MM-DD]

Choose which arXiv announcement batch this run should digest: the *oldest* one
that is old enough to be fully announced and has not been sent yet.

This replaced date arithmetic of the form "today minus five days, rolled back
to an announcement day". That looked fine and was not, because Friday and
Saturday both roll back to Thursday: every Wednesday and Thursday run
re-targeted a batch already sent, so the digest went quiet mid-week and five
announced batches sat unsent behind it. It was also lossy - each of the
Sunday, Monday, Tuesday and Wednesday batches got exactly one run-day, so a
batch whose three crons all failed was never targeted again.

Picking from what is actually outstanding fixes both: idle days only happen
when there is genuinely nothing to send, and a failed batch is simply still
outstanding tomorrow.

arXiv publishes in discrete batches, not continuously: five times a week, at
20:00 ET, a whole batch goes live at once. So the wait needed is hours past
an announcement, not days past a submission. MIN_AGE_DAYS is 1, which puts
each issue out the day after its batch is announced - announcement days are
Sun-Thu, so issues land Mon-Fri with quiet weekends.

Standard library only: this runs before `pip install` in CI, so the heavy
generation dependencies are not available yet.

Writes `date` and `found` to $GITHUB_OUTPUT.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

from arxiv_schedule import (announcement_days_between, is_announcement_day,
                            previous_announcement_day)
from check_already_sent import ArtifactCheckUnavailable, already_sent

# One day: a batch announced at 20:00 ET on day A is queried on day A+1.
# Announcement days are Sun-Thu, so A+1 lands on Mon-Fri - weekday issues,
# quiet weekends, and papers one day past announcement.
#
# This is an age in days, not a safety margin in hours. The margin comes from
# the cron times: announcement is 00:00 UTC (01:00 in winter) and the first
# run is 11:20 UTC. Measured on 2026-09-24, the 2026-09-23 batch was fully
# indexed 10.2 hours after announcement and returned 0 before it, so 11 hours
# is inside the proven range. Do not move the crons earlier without
# re-measuring: a partially indexed batch would ship as a short issue and be
# marked sent.
MIN_AGE_DAYS = 1

# How far back to hunt for something outstanding. This MUST stay well inside
# the artifact retention in digest.yml (30 days), because "sent" is inferred
# from the presence of mt_digest_md-<DATE>. Once an artifact expires the batch
# looks unsent again, so a lookback at or near the retention window would
# quietly re-send month-old issues. Fourteen days recovers a bad week twice
# over and leaves a fortnight of margin.
LOOKBACK_DAYS = 14


def emit(date: str, found: str) -> None:
    """Publish the choice as step outputs, and as $DATE for later steps.

    Writing DATE here rather than parsing it back out of GITHUB_OUTPUT in the
    workflow keeps the YAML free of nested heredocs, which is what broke it
    the first time round.
    """
    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"date={date}\nfound={found}\n")

    env = os.getenv("GITHUB_ENV")
    if env and date:
        with open(env, "a", encoding="utf-8") as fh:
            fh.write(f"DATE={date}\n")

    print(f"date={date} found={found}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-age", type=int, default=MIN_AGE_DAYS,
                    help=f"days a batch must have aged (default: {MIN_AGE_DAYS})")
    ap.add_argument("--lookback", type=int, default=LOOKBACK_DAYS,
                    help=f"how far back to hunt (default: {LOOKBACK_DAYS})")
    ap.add_argument("--date", help="force this date, skipping the queue entirely")
    ns = ap.parse_args()

    # An explicit date is an override: honour it even if already sent, since
    # that is the whole point of asking for one by hand.
    if ns.date:
        day = dt.datetime.strptime(ns.date, "%Y-%m-%d").date()
        if not is_announcement_day(day):
            rolled = previous_announcement_day(day)
            print(f"[info] {day} ({day:%A}) is not an announcement day; "
                  f"using {rolled} ({rolled:%A})")
            day = rolled
        emit(day.isoformat(), "true")
        return 0

    today = dt.date.today()
    newest = today - dt.timedelta(days=ns.min_age)
    oldest = today - dt.timedelta(days=ns.lookback)

    candidates = list(announcement_days_between(oldest, newest))
    if not candidates:
        print(f"[info] no announcement days between {oldest} and {newest}")
        emit("", "false")
        return 0

    print(f"[info] {len(candidates)} candidate batches from {candidates[0]} "
          f"to {candidates[-1]}, oldest first")

    for day in candidates:
        try:
            sent = already_sent(day.isoformat())
        except ArtifactCheckUnavailable as e:
            # Cannot tell. Stopping is the safe move: carrying on would treat
            # an API outage as "not sent" and re-send a batch subscribers
            # already have.
            print(f"[warn] cannot check {day} ({e}); standing down for this run",
                  file=sys.stderr)
            emit("", "false")
            return 0

        if not sent:
            age = (today - day).days
            print(f"[ok] oldest unsent batch: {day} ({day:%A}), {age} days old")
            emit(day.isoformat(), "true")
            return 0

    print(f"[info] every batch back to {candidates[0]} has already been sent")
    emit("", "false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
