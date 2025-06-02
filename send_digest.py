#!/usr/bin/env python3
"""
send_digest.py <digest_markdown_file>

*One‑shot* Buttondown sender for mt-digest:
    – Creates the email with **status=about_to_send** so it is queued
      immediately for all active subscribers.
    – No use of the /send‑draft endpoint ⇒ the [PREVIEW] copy disappears.
    – Idempotent: repeated calls become no‑ops once the email is en route
      or already sent.

Environment variable required:
    BUTTONDOWN_TOKEN – your personal API token.

Exit code is non‑zero only on *unexpected* failures so the CI job fails
only when something really broke.
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
TIMEOUT  = 30  # seconds for all HTTP calls


# ── helpers ─────────────────────────────────────────────────────────────

def bail(msg: str) -> None:
    print(f"\033[91m{msg}\033[0m", file=sys.stderr)
    sys.exit(1)


def request_json(method: str, url: str, **kwargs):
    """requests.request wrapper that always returns resp & resp.json()."""
    resp = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    try:
        data = resp.json()
    except Exception:
        data = {}
    return resp, data


# ── 0⃣ CLI & env checks ────────────────────────────────────────────────
if len(sys.argv) != 2:
    bail("Usage: python send_digest.py mt_digest_YYYY-MM-DD.md")

md_path = pathlib.Path(sys.argv[1]).resolve()
if not md_path.exists():
    bail(f"File not found: {md_path}")

TOKEN = os.getenv("BUTTONDOWN_TOKEN")
if not TOKEN:
    bail("Env var BUTTONDOWN_TOKEN is missing")

subject_date = md_path.stem[-10:]            # YYYY‑MM‑DD at file end
try:
    pretty_date = datetime.strptime(subject_date, "%Y-%m-%d").strftime("%b %d %Y")
except ValueError:
    pretty_date = subject_date               # (should never happen)

subject = f"Machine‑Translation Digest — {pretty_date}"

HEADERS: Dict[str, str] = {
    "Authorization": f"Token {TOKEN}",
    "Content-Type": "application/json",
}

# ── 1⃣ Try to create‑and‑send in one go ────────────────────────────────
print("⏳ Creating + queuing e‑mail…")

payload: Dict[str, Any] = {
    "subject": subject,
    "body": md_path.read_text(encoding="utf-8"),
    "markdown": True,
    "publish_url": False,
    "status": "about_to_send",        # <- queues for delivery instantly
}

create_resp, create_data = request_json(
    "POST", f"{BTN_API}/emails", headers=HEADERS, data=json.dumps(payload)
)

if create_resp.ok:
    email_id = create_data["id"]
    print("✓ Email queued – id:", email_id)
    sys.exit(0)

if create_data.get("code") != "email_duplicate":
    bail(f"Email creation failed → {create_resp.status_code}: {create_data}")

# ── 2⃣ Duplicate: fetch existing record and ensure it is sending ───────
print("ℹ️  Duplicate detected – retrieving existing email…")
q = urllib.parse.quote_plus(subject)
list_resp, list_data = request_json(
    "GET", f"{BTN_API}/emails?search={q}", headers=HEADERS
)
list_resp.raise_for_status()

try:
    email = next(e for e in list_data["results"] if e["subject"] == subject)
except (KeyError, StopIteration):
    bail("Duplicate reported but email not found – aborting")

email_id = email["id"]
state    = email.get("status")
print(f"✓ Found email {email_id} with status {state}")

if state in {"about_to_send", "in_flight", "sent"}:
    print("✅ Already on its way – nothing to do")
    sys.exit(0)

# state is still draft ⇒ patch it
print("⏳ Finalising draft → about_to_send…")
patch_resp, patch_data = request_json(
    "PATCH", f"{BTN_API}/emails/{email_id}", headers=HEADERS,
    data=json.dumps({"status": "about_to_send"})
)

if patch_resp.ok:
    ts = datetime.now(timezone.utc).strftime("%Y‑%m‑%d %H:%M:%S UTC")
    print("✅ Sent at", ts)
    sys.exit(0)

# last‑ditch: benign duplicates / races
if patch_data.get("code") in {"email_invalid_status", "email_already_sent", "email_already_sending"}:
    print("✅ Already sending – all good")
    sys.exit(0)

bail(f"Send failed → {patch_resp.status_code}: {patch_data}")
