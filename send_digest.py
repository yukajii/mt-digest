#!/usr/bin/env python3
"""
send_digest.py <digest_markdown_file>

Automated helper for the MT digest workflow.

Steps
-----
1.  Reads a local *Markdown* file created by **mt_arxiv_digest.py**.
2.  Queues it as a *draft* e‑mail in Buttondown.
3.  If Buttondown returns `email_duplicate`, fetches the existing draft ID
    instead of failing.
4.  Sends the draft to **all active subscribers**.
    • If a send already occurred (duplicate), exits with 0 so the CI run
      is still green.

Required environment variables
------------------------------
* **BUTTONDOWN_TOKEN** – API token from Buttondown dashboard.

Exit status is non‑zero on any *unexpected* HTTP failure so GitHub
Actions will mark the job as failed.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import sys
import time
import urllib.parse
from datetime import datetime
from typing import Any, Dict

import requests

BTN_API = "https://api.buttondown.email/v1"
TIMEOUT = 30  # seconds for all HTTP calls


def die(msg: str) -> None:
    """Print *msg* in red and exit 1."""
    print(f"\033[91m{msg}\033[0m", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 0⃣  Parse CLI & env vars
# ---------------------------------------------------------------------------
if len(sys.argv) != 2:
    die("Usage: python send_digest.py mt_digest_YYYY-MM-DD.md")

md_path = pathlib.Path(sys.argv[1]).resolve()
if not md_path.exists():
    die(f"File not found: {md_path}")

TOKEN = os.getenv("BUTTONDOWN_TOKEN")
if not TOKEN:
    die("Env var BUTTONDOWN_TOKEN is missing")

subject_date = md_path.stem[-10:]  # YYYY-MM-DD at end of filename

# subject_date currently looks like '2025-05-19'
date_obj = datetime.strptime(subject_date, "%Y-%m-%d")
pretty = date_obj.strftime("%b %d %Y")     # May 19 2025
subject = f"Machine-Translation Digest — {pretty}"

headers: Dict[str, str] = {
    "Authorization": f"Token {TOKEN}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# 1⃣  Create draft (or reuse existing duplicate)
# ---------------------------------------------------------------------------
print("⏳ Uploading draft…")

payload: Dict[str, Any] = {
    "subject": SUBJECT,
    "body": md_path.read_text(encoding="utf-8"),
    "markdown": True,
    "publish_url": False,
}

resp = requests.post(f"{BTN_API}/emails", headers=headers, data=json.dumps(payload), timeout=TIMEOUT)

if not resp.ok:
    err = resp.json()
    if err.get("code") == "email_duplicate":
        print("ℹ️  Draft already exists – fetching its ID")
        q = urllib.parse.quote_plus(SUBJECT)
        time.sleep(1)
        drafts = requests.get(
            f"{BTN_API}/emails?state=draft&search={q}", headers=headers, timeout=TIMEOUT
        )
        drafts.raise_for_status()
        try:
            email_id = drafts.json()["results"][0]["id"]
        except (IndexError, KeyError):
            die("Duplicate reported but existing draft not found – aborting")
        print("✓ Re‑using draft:", email_id)
    else:
        die(f"Draft upload failed → {err}")
else:
    email_id = resp.json()["id"]
    print("✓ Draft created:", email_id)

# ---------------------------------------------------------------------------
# 2⃣  Send draft to full subscriber list
# ---------------------------------------------------------------------------
print("⏳ Sending to subscribers…")

send_resp = requests.post(
    f"{BTN_API}/emails/{email_id}/send-draft",
    headers=headers,
    data=json.dumps({}),  # empty → full list
    timeout=TIMEOUT,
)

if not send_resp.ok:
    err = send_resp.json()
    # 400 with duplicate happens if the email was already sent earlier
    if send_resp.status_code == 400 and err.get("code") == "email_duplicate":
        print("ℹ️  Email already sent earlier – nothing to do")
        sys.exit(0)
    die(f"Send failed → {send_resp.status_code}: {send_resp.text}")

print("✅ Sent at", datetime.datetime.utcnow().isoformat(" "), "UTC")
