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


class ArtifactCheckUnavailable(Exception):
    """The API could not answer. Distinct from a confident "not sent"."""


def already_sent(date_str: str, repo: str | None = None,
                 token: str | None = None) -> bool:
    """True when this date's digest artifact exists.

    A run uploads mt_digest_md-<DATE> only after it has successfully built
    *and* sent, so the artifact is a self-contained marker that the day's
    digest went out.

    Raises ArtifactCheckUnavailable when the API cannot be reached, so callers
    can tell "definitely not sent" apart from "could not find out" - pick_batch
    must not treat an outage as a licence to re-send an old batch.
    """
    repo = repo or os.getenv("GITHUB_REPOSITORY")
    token = token or os.getenv("GITHUB_TOKEN")
    if not repo or not token:
        raise ArtifactCheckUnavailable("GITHUB_REPOSITORY or GITHUB_TOKEN missing")

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
    except Exception as e:
        raise ArtifactCheckUnavailable(str(e)) from e

    return any(a.get("name") == name and not a.get("expired", False)
               for a in data.get("artifacts", []))


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: check_already_sent.py YYYY-MM-DD", file=sys.stderr)
        emit(False)
        return

    date_str = sys.argv[1]
    try:
        sent = already_sent(date_str)
    except ArtifactCheckUnavailable as e:
        # Outside Actions, or a network hiccup -> don't block a real run.
        print(f"[warn] artifact check failed, proceeding anyway: {e}", file=sys.stderr)
        emit(False)
        return

    if sent:
        print(f"Digest for {date_str} already produced "
              f"(artifact mt_digest_md-{date_str}) - skipping")
    emit(sent)


if __name__ == "__main__":
    main()
