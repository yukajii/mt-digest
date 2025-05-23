#!/usr/bin/env python3
"""
send_digest.py <digest_markdown_file>

Reads Markdown, queues an email draft in Buttondown, then sends it
to all active subscribers.

Env vars required:
  BUTTONDOWN_TOKEN  – Your Buttondown API token
"""
import os, sys, json, pathlib, requests, datetime

if len(sys.argv) != 2:
    sys.exit("Usage: python send_digest.py mt_digest_YYYY-MM-DD.md")

md_path = pathlib.Path(sys.argv[1]).resolve()
if not md_path.exists():
    sys.exit(f"File not found: {md_path}")

token = os.getenv("BUTTONDOWN_TOKEN")
if not token:
    sys.exit("Set BUTTONDOWN_TOKEN env var first")

subject_date = md_path.stem[-10:]          # get YYYY-MM-DD from filename
subject = f"MT digest – {subject_date}"

headers = {
    "Authorization": f"Token {token}",
    "Content-Type": "application/json",
}

# ---------- 1) Create draft ----------
payload = {
    "subject": subject,
    "body": md_path.read_text("utf-8"),
    "markdown": True,
    "publish_url": False,
}

draft = requests.post(
    "https://api.buttondown.email/v1/emails",
    headers=headers,
    data=json.dumps(payload),
    timeout=30,
)

if not draft.ok:
    print("⚠️  Draft creation failed")
    print("Status :", draft.status_code)
    print("Body   :", draft.text)
    sys.exit(1)

email_id = draft.json()["id"]
print("✓ Draft created:", email_id)

# ---------- 2) Send to list ----------
send_resp = requests.post(
    f"https://api.buttondown.email/v1/emails/{email_id}/send-draft",
    headers=headers,
    data=json.dumps({}),      # empty → send to full list
    timeout=30,
)

if not send_resp.ok:
    print("⚠️  Send-draft failed")
    print("Status :", send_resp.status_code)
    print("Body   :", send_resp.text)
    sys.exit(1)

print("✓ Sent to subscribers")
