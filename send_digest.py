#!/usr/bin/env python3
"""
send_digest.py <digest_markdown_file>

Pushes a markdown digest created by **mt_arxiv_digest.py** to Buttondown
and sends it once — without the “[PREVIEW]” duplicate.

How it works
------------
1. POST /v1/emails              → create (or reuse) a *draft* e-mail.
2. PATCH /v1/emails/{id}        → set `"status": "about_to_send"`.
   Buttondown immediately queues the message for delivery to **all**
   active subscribers.

Environment
-----------
* BUTTONDOWN_TOKEN – API token from Buttondown.

Exit 1 is reserved for _unexpected_ HTTP failures so GitHub Actions
marks genuine problems only.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict

import requests

BTN_API = "https://api.buttondown.email/v1"
TIMEOUT  = 30  # seconds


def bail(msg: str) -> None:
    """Print *msg* in red and terminate with exit 1."""
    print(f"\033[91m{msg}\033[0m", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 0⃣  CLI + env-var checks
# ---------------------------------------------------------------------------
if len(sys.argv) != 2:
    bail("Usage: python send_digest.py mt_digest_YYYY-MM-DD.md")

md_path = pathlib.Path(sys.argv[1]).resolve()
if not md_path.exists():
    bail(f"File not found: {md_path}")

TOKEN = os.getenv("BUTTONDOWN_TOKEN")
if not TOKEN:
    bail("Env var BUTTONDOWN_TOKEN is missing")

subject_date = md_path.stem[-10:]         # YYYY-MM-DD suffix in filename
try:
    pretty_date = datetime.strptime(subject_date, "%Y-%m-%d").strftime("%b %d %Y")
except ValueError:
    pretty_date = subject_date            # (should never happen)

subject = f"Machine-Translation Digest for {pretty_date}"

HEADERS: Dict[str, str] = {
    "Authorization": f"Token {TOKEN}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# 1⃣  Create draft (or reuse an existing one)
# ---------------------------------------------------------------------------
print("⏳ Uploading draft…")

payload: Dict[str, Any] = {
    "subject": subject,
    "body": md_path.read_text(encoding="utf-8"),
    "markdown": True,
    "publish_url": False,
    "status": "draft",                    # explicit, for clarity
}

resp = requests.post(f"{BTN_API}/emails",
                     headers=HEADERS,
                     data=json.dumps(payload),
                     timeout=TIMEOUT)

if resp.ok:
    email_id = resp.json()["id"]
    print("✓ Draft created:", email_id)
else:
    err = resp.json()
    if err.get("code") == "email_duplicate":
        print("ℹ️  Draft already exists – locating it")
        q = urllib.parse.quote_plus(subject)
        time.sleep(1)  # eventual-consistency guard
        drafts = requests.get(f"{BTN_API}/emails?state=draft&search={q}",
                              headers=HEADERS, timeout=TIMEOUT)
        drafts.raise_for_status()
        try:
            email_id = drafts.json()["results"][0]["id"]
            print("✓ Re-using draft:", email_id)
        except (IndexError, KeyError):
            bail("Duplicate reported but existing draft not found – aborting")
    else:
        bail(f"Draft upload failed → {err}")

# ---------------------------------------------------------------------------
# 2⃣  Finalise & send (no preview!)
# ---------------------------------------------------------------------------
print("⏳ Finalising e-mail…")

finalise = requests.patch(f"{BTN_API}/emails/{email_id}",
                          headers=HEADERS,
                          data=json.dumps({"status": "about_to_send"}),
                          timeout=TIMEOUT)

if finalise.ok:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("✅ Sent at", ts)
    sys.exit(0)

# If we get here, something went sideways — but handle benign cases gracefully.
err_json = {}
try:
    err_json = finalise.json()
except Exception:                           # non-JSON response ⇒ fatal
    bail(f"Send failed → {finalise.status_code}: {finalise.text}")

code = err_json.get("code", "")
if code in {"email_invalid_status", "email_already_sent",
            "email_already_sending"}:
    # The e-mail is already en route / sent — treat as success.
    print("ℹ️  Email already sent or in flight – nothing to do")
    sys.exit(0)

# Anything else is an unexpected failure.
bail(f"Send failed → {finalise.status_code}: {err_json}")
