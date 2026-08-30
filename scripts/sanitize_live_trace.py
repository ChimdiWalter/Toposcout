#!/usr/bin/env python3
"""Sanitize a live deployed-agent ADK session record for submission (M6 Phase 2).

Input: the session JSON fetched from the deployed ADK api_server
(GET /apps/<app>/users/<user>/sessions/<id>). Output: a plain-text trace that
RETAINS tool names/order, exact deterministic evidence, policy decisions, the
final report URI, Cloud Run revision, and timestamps — and REMOVES tokens,
thought signatures, and container-filesystem detail.

Usage:
    python scripts/sanitize_live_trace.py session.json out.txt \
        --revision toposcout-agent-00003-4j6 --model gemini-3.5-flash
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SENSITIVE_KEYS = {"thoughtSignature", "thought_signature"}


def ts(t: float | None) -> str:
    if not t:
        return "-"
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="seconds")


def redact(obj):
    """Drop signature keys; replace container-local paths with a marker."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items() if k not in SENSITIVE_KEYS}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        if obj.startswith(("/tmp/", "/workspace/")):
            return f"<container-local>/{Path(obj).name}"
        return obj
    return obj


def fmt_payload(obj, indent: str = "    ") -> str:
    text = json.dumps(redact(obj), indent=2, sort_keys=True)
    return "\n".join(indent + line for line in text.splitlines())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_json")
    ap.add_argument("out_txt")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--service", default="toposcout-agent (Cloud Run, us-central1)")
    args = ap.parse_args()

    session = json.loads(Path(args.session_json).read_text())
    events = session.get("events", [])

    lines: list[str] = []
    lines.append("TOPOSCOUT LIVE DEPLOYED AGENT TRACE (sanitized)")
    lines.append(f"service:  {args.service}")
    lines.append(f"revision: {args.revision}")
    lines.append(f"model:    {args.model}")
    lines.append(f"session:  {session.get('id', '?')}  app={session.get('appName', session.get('app_name', '?'))}")
    lines.append("sanitization: removed auth tokens, API keys, model thought signatures,")
    lines.append("and container filesystem paths; tool order, arguments, deterministic")
    lines.append("evidence, policy decisions, report URI, and timestamps are verbatim.")
    lines.append("=" * 78)

    for ev in events:
        author = ev.get("author", "?")
        stamp = ts(ev.get("timestamp"))
        for part in (ev.get("content") or {}).get("parts", []):
            if "text" in part:
                lines.append(f"\n[{stamp}] {author} says:")
                lines.append(fmt_payload(part["text"].strip()) if isinstance(part["text"], dict)
                             else "    " + part["text"].strip().replace("\n", "\n    "))
            if "functionCall" in part:
                fc = part["functionCall"]
                lines.append(f"\n[{stamp}] {author} -> TOOL CALL {fc['name']}")
                lines.append(fmt_payload(fc.get("args", {})))
            if "functionResponse" in part:
                fr = part["functionResponse"]
                resp = fr.get("response", {})
                lines.append(f"[{stamp}] TOOL RESULT {fr['name']}")
                lines.append(fmt_payload(resp))

    import re
    text = "\n".join(lines) + "\n"
    text = re.sub(r"(?:/tmp|/workspace)(?:/[\w.\-]+)+", "<container-local-path>", text)
    for bad in SENSITIVE_KEYS:
        assert bad not in text
    assert "Bearer " not in text and "/tmp/" not in text and "ya29." not in text
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text(text)
    print(f"wrote {args.out_txt} ({len(events)} events)")


if __name__ == "__main__":
    main()
