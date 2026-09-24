#!/usr/bin/env python3
"""
arxiv_schedule.py - when arXiv announces, and what each batch covers.

Standard library only, on purpose. pick_batch.py has to resolve the target
date *before* `pip install` runs in CI, so this cannot import numpy, arxiv or
anything else heavy. Keeping the rules here rather than duplicating them is
what stops the workflow and the digest disagreeing about which day they mean -
that duplication used to live in bash and was a standing trap.

https://info.arxiv.org/help/availability.html

arXiv announces five batches a week, never on Friday or Saturday. Each batch
closes at 14:00 ET and goes live at 20:00 ET the same day:

  submitted Mon 14:00 - Tue 14:00  ->  announced Tue 20:00
  submitted Tue 14:00 - Wed 14:00  ->  announced Wed 20:00
  submitted Wed 14:00 - Thu 14:00  ->  announced Thu 20:00
  submitted Thu 14:00 - Fri 14:00  ->  announced Sun 20:00
  submitted Fri 14:00 - Mon 14:00  ->  announced Mon 20:00   <- weekend

The last row is why this works in batches rather than calendar days.
Saturdays and Sundays carry real cs.CL submissions (38-56 a day in a
three-week sample); they are simply announced together with Friday afternoon
and Monday morning.
"""
from __future__ import annotations

import datetime as dt
from typing import Tuple

ARXIV_DEADLINE_HOUR = 14          # 14:00 ET, the submission cut-off
_ET_NAME = "America/New_York"

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


def announcement_days_between(oldest: dt.date, newest: dt.date):
    """Every announcement day in [oldest, newest], oldest first."""
    day = oldest
    while day <= newest:
        if is_announcement_day(day):
            yield day
        day += dt.timedelta(days=1)


def batch_window(announce_day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    """UTC [start, end] of the submission window announced on `announce_day`.

    Deadlines are wall-clock 14:00 ET, so the UTC offset shifts with US
    daylight saving. Converting through the tz database keeps the window
    correct across the March and November transitions.

    ZoneInfo is imported here rather than at module scope so that callers who
    only need the weekday rules - pick_batch.py, running before pip install -
    never touch the tz database.
    """
    from zoneinfo import ZoneInfo

    if not is_announcement_day(announce_day):
        raise ValueError(
            f"{announce_day} ({announce_day:%A}) is not an arXiv announcement day"
        )

    eastern = ZoneInfo(_ET_NAME)
    start_off, end_off = _BATCH_SPAN[announce_day.weekday()]
    deadline = dt.time(ARXIV_DEADLINE_HOUR, 0)

    start_et = dt.datetime.combine(
        announce_day + dt.timedelta(days=start_off), deadline, tzinfo=eastern
    )
    end_et = dt.datetime.combine(
        announce_day + dt.timedelta(days=end_off), deadline, tzinfo=eastern
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
