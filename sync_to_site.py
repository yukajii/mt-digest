#!/usr/bin/env python3
"""
sync_to_site.py <YYYY-MM-DD> --out DIR

Compose the yukajii.com content file for one issue and write it into DIR.

Three files hold the pieces, deliberately:

  mt_digest_<date>.md      the e-mail body, with no front matter, exactly as
                           Buttondown received it
  logs/mt_digest_<date>.log  the run log, which carries the web headline
  logs/sent_<date>.json    the send receipt, which carries the real Buttondown
                           slug (it cannot be derived - Buttondown appends a
                           random suffix on a repeated subject)

This merges them into `<publish-date>-<slug>.md` with YAML front matter, the
shape build-mt-digest.mjs on the site expects.

Exits 0 and writes nothing when a piece is missing, so a scheduled run that
skipped the send, or predates the receipt, does not fail the build.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

BASE_DIR = pathlib.Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def yaml_quote(value: Any) -> str:
    """Double-quoted YAML scalar, safe for any headline.

    Control characters are stripped rather than escaped: one stray byte makes
    the whole front-matter block unparseable and no headline needs them.
    """
    if value is None:
        return '""'
    text = _CONTROL.sub("", str(value))
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def load_json(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"[warn] {path.name} is not valid JSON ({e})", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", help="announcement date, YYYY-MM-DD")
    ap.add_argument("--out", required=True,
                    help="content directory in the site repo")
    ns = ap.parse_args()

    date = ns.date
    md_path = BASE_DIR / f"mt_digest_{date}.md"
    if not md_path.exists():
        print(f"[info] no {md_path.name} - nothing to sync (quiet batch, or "
              "the send was skipped)")
        return 0

    receipt = load_json(LOG_DIR / f"sent_{date}.json")
    if not receipt or not receipt.get("slug"):
        print(f"[info] no send receipt for {date} - not synced. The issue is "
              "only published once Buttondown has confirmed it.")
        return 0

    run_log = load_json(LOG_DIR / f"mt_digest_{date}.log") or {}
    title = (run_log.get("title") or "").strip()
    if not title:
        print("[info] no headline in the run log; the page will fall back to a "
              "dated heading")

    publish_date = (receipt.get("publish_date") or "")[:10] or date
    slug = receipt["slug"]

    front = {
        "subject": receipt.get("subject"),
        "slug": slug,
        "title": title,
        "publish_date": receipt.get("publish_date"),
        "status": receipt.get("status"),
        "absolute_url": receipt.get("absolute_url"),
        "id": receipt.get("id"),
    }
    lines = ["---"]
    lines += [f"{k}: {yaml_quote(v)}" for k, v in front.items() if v is not None]
    lines += ["---", "", md_path.read_text(encoding="utf-8").strip(), ""]

    out_dir = pathlib.Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{publish_date}-{slug}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote {out_path.name}"
          + (f' with headline "{title}"' if title else " (dated heading)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
