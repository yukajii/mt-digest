# Yukajii MT Digest

Daily machine‑translation (MT) research digest automatically pulled from the cs.CL section of arXiv, lightly re‑ranked for MT relevance, and delivered to your inbox via [Buttondown](https://buttondown.email).

---

## What this repo contains

| File / Dir           | Purpose                                                         |
| -------------------- | --------------------------------------------------------------- |
| `mt_arxiv_digest.py` | Scrapes the previous day’s `cs.CL` pre‑prints, embeds them with |

| [*e5‑large‑v2*](https://arxiv.org/abs/2212.07544), picks the top‑*k* MT‑related papers, calls **GPT‑4o** for a 2‑sentence intro, and writes `mt_digest_YYYY‑MM‑DD.md` + a JSON log. |                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `send_digest.py`                                                                                                                                                                    | Posts the generated Markdown to Buttondown using its REST API.                                                                                                |
| `.github/workflows/digest.yml`                                                                                                                                                      | GitHub Actions workflow that runs every morning (06:20 UTC) or on demand: builds the digest, e‑mails it, and uploads the Markdown & log as private artifacts. |
| `logs/` (ignored)                                                                                                                                                                   | JSON run‑logs (kept 30 days as artifacts; not committed).                                                                                                     |
| `.env.example`                                                                                                                                                                      | Template for required environment variables.                                                                                                                  |

---

## Quick start (local)

```bash
# clone & enter
$ git clone https://github.com/yukajii/yukajii-site.git && cd yukajii-site

# Python ≥3.11 recommended
$ python -m venv venv && source venv/bin/activate
$ pip install -r requirements.txt

# export secrets (or copy .env.example → .env)
$ export OPENAI_API_KEY=sk-…
$ export BUTTONDOWN_TOKEN=bd_ApiKey…

# build yesterday’s digest and send it
$ python mt_arxiv_digest.py           # builds mt_digest_YYYY-MM-DD.md
$ python send_digest.py mt_digest_YYYY-MM-DD.md
```

Command‑line flags:

```text
--date YYYY-MM-DD   pick a custom UTC date (default: yesterday)
--max  N            include at most N papers (default: 5)
```

---

## Setting up Buttondown

1. **Create an account** at [https://buttondown.email/register](https://buttondown.email/register) and finish sender‑address verification.
2. Go to **Settings → API** and click **Generate new token**. Add that token as `BUTTONDOWN_TOKEN` (GitHub → Settings → Secrets or local env).
3. Ensure you have at least one *newsletter* (Buttondown creates `default` automatically). `send_digest.py` posts to whichever newsletter is set as *default* in your account.
4. Optional tweaks in `send_digest.py`:

   * change `headers['X-Buttondown-Newsletter']` if you run multiple newsletters;
   * set `publish_status` to `"draft"` instead of `"public"` for manual review.

That’s it — hitting the API with a Markdown body will schedule the e‑mail.

---

## Running automatically on GitHub Actions

The workflow file already expects two encrypted secrets:

| Secret name        | Value                                |
| ------------------ | ------------------------------------ |
| `OPENAI_API_KEY`   | Your OpenAI key (gpt‑4o).            |
| `BUTTONDOWN_TOKEN` | The Buttondown API token from above. |

The job installs dependencies, restores cached model files, generates & sends the digest, then uploads two artifacts per run:

```
mt_digest_md-YYYY-MM-DD   # the Markdown
mt_digest_log-YYYY-MM-DD  # JSON with metadata & token usage
```

No files are pushed back to the repo — digests live solely in the third‑party service and as ephemeral artifacts.

---

## Contributing

PRs are welcome! Feel free to open issues or suggest better ranking heuristics, alternative embeddings, or integrations.

---

## License

MIT © 2025 yukajii / yukajii.com
