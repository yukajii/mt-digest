#!/usr/bin/env python3
"""
check_already_sent.py <YYYY-MM-DD>

Guard for the digest workflow's same-day reattempt schedule. The workflow
runs a few times per UTC day so a transient arXiv block at one run time
doesn't strand the day's digest — but every run targets the SAME date, so
without a guard the later runs would send the newsletter a second time.

Signal used: a run uploads the `mt_digest_md-<DATE>` artifact only AFTER it
has successfully built *and* sent the digest. So the presence of that
artifact is a reliable, self-contained marker that the day's digest already
went out. (An earlier attempt that reused Buttondown's `?search=` lookup was
unreliable and let a reattempt re-send — the artifact check is deterministic
and fully under our control.)

This script asks the GitHub API whether that artifact exists and, if so,
writes `already_sent=true` to $GITHUB_OUTPUT, letting the workflow skip the
heavy generation + send steps (and avoid re-hitting arXiv).

Uses only the Python standard library and never fails the build: on any
error it reports already_sent=false so a genuine run can proceed.

Requires GITHUB_TOKEN (with `actions: read`) and GITHUB_REPOSITORY, both
provided automatically inside GitHub Actions.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

API = "https://api.github.com"


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
    repo = os.getenv("GITHUB_REPOSITORY")     # e.g. "yukajii/mt-digest"
    token = os.getenv("GITHUB_TOKEN")
    if not repo or not token:
        # Outside Actions or missing token → can't check; let the run proceed.
        emit(False)
        return

    name = f"mt_digest_md-{date_str}"
    url = f"{API}/repos/{repo}/actions/artifacts?name={name}&per_page=100"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # network / API hiccup → don't block a real run
        print(f"[warn] artifact check failed, proceeding anyway: {e}", file=sys.stderr)
        emit(False)
        return

    for art in data.get("artifacts", []):
        if art.get("name") == name and not art.get("expired", False):
            print(f"✓ Digest for {date_str} already produced (artifact {name}) → skipping")
            emit(True)
            return

    emit(False)


if __name__ == "__main__":
    main()
