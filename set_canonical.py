#!/usr/bin/env python3
"""
set_canonical.py [--apply] [--limit N] [--site URL]

Point each Buttondown archive page at its yukajii.com counterpart.

WARNING: despite the name and the documentation, setting canonical_url does
not add a meta tag. Buttondown 302s that issue's archive page to the URL, so
the archive copy stops serving and only the yukajii.com page stays readable.
Measured against a control on 2026-09-22:

    Sep 17 (canonical set)  HTTP 302  ->  yukajii.com/mt-digest/2026-09-17/
    Sep 16 (no canonical)   HTTP 200  30,624B

That is the intended outcome here - one public page per issue - but it is a
bigger action than "set a canonical" suggests. The newsletter root and the
archive index are unaffected; the canonical is per-email.

Why this is not done at send time: the site page does not exist yet. The
workflow stages each issue as a pull request, and until that is merged the URL
404s. A canonical pointing at a missing page is worse than none, so this runs
separately and **only sets the canonical once the page actually answers 200**.
That makes it self-healing - run it again after merging and it catches up.

Read-only unless --apply is passed. Without it, every intended change is
printed and nothing is sent.

Environment variable required:
    BUTTONDOWN_TOKEN - your personal API token.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterator

import requests

BTN_API = "https://api.buttondown.email/v1"
DEFAULT_SITE = "https://yukajii.com"
TIMEOUT = 30
PAUSE = 0.4

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def bail(msg: str) -> None:
    print(f"\033[91m{msg}\033[0m", file=sys.stderr)
    sys.exit(1)


def date_from_subject(subject: str) -> str | None:
    """The subject carries the arXiv announcement date, which is the URL key."""
    m = re.search(r"for\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})", subject or "")
    if not m or m.group(1) not in MONTHS:
        return None
    return f"{m.group(3)}-{MONTHS.index(m.group(1)) + 1:02d}-{int(m.group(2)):02d}"


def iter_emails(s: requests.Session, limit: int | None) -> Iterator[Dict[str, Any]]:
    url: str | None = f"{BTN_API}/emails"
    params: Dict[str, Any] | None = {"page": 1}
    seen = 0
    while url:
        r = s.get(url, params=params, timeout=TIMEOUT)
        if r.status_code == 401:
            bail("Buttondown rejected the token (401). Check BUTTONDOWN_TOKEN.")
        r.raise_for_status()
        payload = r.json()
        results = payload if isinstance(payload, list) else payload.get("results", [])
        nxt = None if isinstance(payload, list) else payload.get("next")
        for item in results:
            yield item
            seen += 1
            if limit and seen >= limit:
                return
        url, params = nxt, None
        if url:
            time.sleep(PAUSE)


def page_is_live(s: requests.Session, url: str) -> bool:
    """A HEAD that only counts as live for a real page.

    The site is a single-page app, so an unknown path still answers 200 with
    the app shell. Content-length is the tell: the shell is ~2 kB, a real
    issue page an order of magnitude more. Learned this the hard way - a
    status-code-only check reported the archive live before it had deployed.
    """
    try:
        r = s.head(url, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            return False
        length = int(r.headers.get("content-length") or 0)
        if length and length < 4096:
            return False
        if not length:                       # no header: fall back to a GET
            g = s.get(url, timeout=TIMEOUT)
            return g.status_code == 200 and len(g.content) >= 4096
        return True
    except requests.RequestException:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually PATCH. Without it, nothing is sent.")
    ap.add_argument("--limit", type=int, help="only look at the N most recent emails")
    ap.add_argument("--site", default=DEFAULT_SITE, help=f"site root (default: {DEFAULT_SITE})")
    ns = ap.parse_args()

    token = os.getenv("BUTTONDOWN_TOKEN")
    if not token:
        bail("Env var BUTTONDOWN_TOKEN is missing")

    api = requests.Session()
    api.headers.update({"Authorization": f"Token {token}",
                        "Accept": "application/json"})
    web = requests.Session()
    web.headers.update({"User-Agent": "mt-digest-canonical/1.0"})

    if not ns.apply:
        print("DRY RUN - nothing will be sent. Pass --apply to write.\n")

    checked = skipped_have = skipped_nodate = skipped_nopage = 0
    would = done = failed = 0

    for email in iter_emails(api, ns.limit):
        checked += 1
        if email.get("canonical_url"):
            skipped_have += 1
            continue
        date = date_from_subject(email.get("subject", ""))
        if not date:
            skipped_nodate += 1
            continue

        target = f"{ns.site.rstrip('/')}/mt-digest/{date}/"
        if not page_is_live(web, target):
            skipped_nopage += 1
            continue

        if not ns.apply:
            print(f"  would set {date} -> {target}")
            would += 1
            continue

        r = api.patch(f"{BTN_API}/emails/{email['id']}",
                      headers={"Content-Type": "application/json"},
                      data=json.dumps({"canonical_url": target}),
                      timeout=TIMEOUT)
        if r.ok:
            print(f"  set {date} -> {target}")
            done += 1
        else:
            body = r.text[:200]
            print(f"  FAILED {date}: {r.status_code} {body}", file=sys.stderr)
            failed += 1
            if "extra_forbidden" in body or r.status_code == 422:
                bail("Buttondown rejected canonical_url as an unknown field. "
                     "Stopping rather than hammering the API.")
        time.sleep(PAUSE)

    print(f"\nchecked {checked} emails")
    print(f"  already had a canonical : {skipped_have}")
    print(f"  no date in subject      : {skipped_nodate}")
    print(f"  no live page yet        : {skipped_nopage}")
    print(f"  {'would set' if not ns.apply else 'set'}               : "
          f"{would if not ns.apply else done}")
    if failed:
        print(f"  failed                  : {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
