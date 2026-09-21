#!/usr/bin/env python3
"""
export_archive.py [--out DIR] [--limit N] [--schema-only]

Read-only exporter for the Daily MT Picks back catalogue.

Pulls every email from Buttondown and writes one Markdown file per issue,
with YAML front matter, so the archive lives somewhere you control instead of
only inside a third-party service. Nothing is ever written back to Buttondown:
this script issues GET requests and nothing else.

It also prints the Email object's actual field names on first fetch. That is
deliberate - the public API reference does not document the full schema, and
we specifically need to know whether `canonical_url` is present and writable
before any send path starts setting it. Adding an unrecognised field to
POST /v1/emails is what caused the 422 that broke sending once already.

Environment variable required:
    BUTTONDOWN_TOKEN - your personal API token.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, Iterator, List

import requests

BTN_API = "https://api.buttondown.email/v1"
TIMEOUT = 30
PAGE_PAUSE = 0.4          # be polite; the archive is small but not tiny

# Fields we care about downstream. Anything else is preserved in the sidecar
# JSON rather than silently dropped.
FRONT_MATTER_FIELDS = (
    "id", "subject", "slug", "publish_date", "status",
    "canonical_url", "absolute_url", "secondary_id",
)


def bail(msg: str) -> None:
    print(f"\033[91m{msg}\033[0m", file=sys.stderr)
    sys.exit(1)


def session() -> requests.Session:
    token = os.getenv("BUTTONDOWN_TOKEN")
    if not token:
        bail("Env var BUTTONDOWN_TOKEN is missing")
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Token {token}",
        "Accept": "application/json",
    })
    return s


def iter_emails(s: requests.Session, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    """Yield every email, following whatever pagination style the API uses."""
    url: str | None = f"{BTN_API}/emails"
    params: Dict[str, Any] | None = {"page": 1}
    seen = 0
    first = True

    while url:
        resp = s.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 401:
            bail("Buttondown rejected the token (401). Check BUTTONDOWN_TOKEN.")
        resp.raise_for_status()
        payload = resp.json()

        if isinstance(payload, list):           # unpaginated list response
            results, nxt = payload, None
        else:
            results = payload.get("results", [])
            nxt = payload.get("next")

        if first and results:
            # The point of the exercise: show what the API actually returns.
            print("=" * 68)
            print("Email object fields as returned by the API:")
            for k in sorted(results[0].keys()):
                v = results[0][k]
                preview = str(v).replace("\n", " ")[:56]
                print(f"  {k:24s} {type(v).__name__:8s} {preview}")
            print("=" * 68)
            print(f"  canonical_url present: {'canonical_url' in results[0]}")
            print("=" * 68)
            first = False

        for item in results:
            yield item
            seen += 1
            if limit and seen >= limit:
                return

        url, params = nxt, None                 # `next` is a full URL
        if url:
            time.sleep(PAGE_PAUSE)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "issue"


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def yaml_quote(value: Any) -> str:
    """Double-quoted YAML scalar, safe for any subject line.

    Control characters are stripped rather than escaped: a stray one makes the
    whole front-matter block unparseable, and no subject line needs them.
    """
    if value is None:
        return '""'
    text = _CONTROL_CHARS.sub("", str(value))
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_issue(email: Dict[str, Any], out: pathlib.Path) -> pathlib.Path:
    date = (email.get("publish_date") or "")[:10] or "undated"
    slug = email.get("slug") or slugify(email.get("subject", ""))
    path = out / f"{date}-{slug}.md"

    lines = ["---"]
    for field in FRONT_MATTER_FIELDS:
        if field in email:
            lines.append(f"{field}: {yaml_quote(email[field])}")
    lines += ["---", "", (email.get("body") or "").strip(), ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="archive", help="output directory (default: archive)")
    ap.add_argument("--limit", type=int, help="stop after N emails (for a quick look)")
    ap.add_argument("--schema-only", action="store_true",
                    help="print the field listing and exit without writing files")
    ns = ap.parse_args()

    s = session()
    out = pathlib.Path(ns.out)

    if ns.schema_only:
        for _ in iter_emails(s, limit=1):
            pass
        return

    out.mkdir(parents=True, exist_ok=True)
    raw: List[Dict[str, Any]] = []
    written = 0
    statuses: Dict[str, int] = {}
    with_canonical = 0

    for email in iter_emails(s, ns.limit):
        raw.append(email)
        status = email.get("status", "?")
        statuses[status] = statuses.get(status, 0) + 1
        if email.get("canonical_url"):
            with_canonical += 1
        write_issue(email, out)
        written += 1

    # Sidecar with the untouched API payload, so nothing is lost to the
    # front-matter whitelist and a later pass can use fields we ignored.
    (out / "_raw.json").write_text(
        json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nwrote {written} issues to {out}/")
    print(f"  by status: {statuses}")
    print(f"  already carrying a canonical_url: {with_canonical}")
    if raw:
        dates = sorted(e.get("publish_date", "")[:10] for e in raw if e.get("publish_date"))
        if dates:
            print(f"  date range: {dates[0]} .. {dates[-1]}")


if __name__ == "__main__":
    main()
