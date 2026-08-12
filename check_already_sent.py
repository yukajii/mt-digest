#!/usr/bin/env python3
"""
check_already_sent.py <YYYY-MM-DD>

Guard for the digest workflow's reattempt schedule. The workflow runs a few
times per UTC day so a transient arXiv block at one run time doesn't strand
the day's digest. All those runs target the *same* date, so this guard asks
Buttondown whether that digest is already sent/queued and, if so, writes
`already_sent=true` to $GITHUB_OUTPUT — letting the workflow skip the heavy
generation + send steps (and, importantly, avoid re-hitting arXiv) on days
that already succeeded.

Uses only the Python standard library (no pip install needed) and never
fails the build: on any error it reports already_sent=false so a genuine
run can proceed.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime

BTN_API = "https://api.buttondown.email/v1"

# Statuses that mean the digest is already delivered or on its way, so a
# reattempt run should skip. A lone `draft` is deliberately NOT here: a stuck
# draft should be retried (send_digest.py finalises it to about_to_send).
DONE_STATUSES = {"about_to_send", "in_flight", "sent", "scheduled", "imported"}


def emit(already_sent: bool) -> None:
    line = f"already_sent={'true' if already_sent else 'false'}"
    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    print(line)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: check_already_sent.py YYYY-MM-DD", file=sys.stderr)
        emit(False)
        return

    date_str = sys.argv[1]
    token = os.getenv("BUTTONDOWN_TOKEN")
    if not token:
        # No token → can't check; let the run proceed (the send step will
        # error loudly if the token really is missing).
        emit(False)
        return

    try:
        pretty = datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d %Y")
    except ValueError:
        emit(False)
        return
    subject = f"Machine Translation Digest for {pretty}"

    url = f"{BTN_API}/emails?search=" + urllib.parse.quote_plus(subject)
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # network / API hiccup → don't block a real run
        print(f"[warn] Buttondown check failed, proceeding anyway: {e}", file=sys.stderr)
        emit(False)
        return

    for email in data.get("results", []):
        if email.get("subject") == subject and email.get("status") in DONE_STATUSES:
            print(f"✓ Digest for {date_str} already '{email.get('status')}' → skipping")
            emit(True)
            return

    emit(False)


if __name__ == "__main__":
    main()
