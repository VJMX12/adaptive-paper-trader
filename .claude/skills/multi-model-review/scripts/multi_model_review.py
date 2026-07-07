#!/usr/bin/env python3
"""
multi_model_review.py

Sends one or more source files to Claude, GPT, and Gemini in parallel,
with an identical prompt and identical pass/fail criteria, then writes
a side-by-side markdown report so you can compare what each model
actually flagged.
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import sys
import urllib.request
import urllib.error

REVIEW_PROMPT_TEMPLATE = """You are doing a focused code review of a quantitative trading system component.

Review the following code for:
1. Correctness bugs
2. Silent failure modes (places where something fails without raising or logging)
3. Edge cases that aren't handled

Grade the code against these specific pass/fail criteria, if applicable to what's shown:
{criteria}

Be concrete. Cite the specific line, function, or block for every issue you raise.
If you find nothing wrong in a category, say so explicitly rather than omitting it.
Do not pad your answer with generic best-practice advice unrelated to what's in the code.

--- CODE ({filename}) ---
{code}
--- END CODE ---
"""


def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    body = json.dumps({
        "model": model,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return "".join(block.get("text", "") for block in data.get("content", []))


def call_openai(prompt: str, model: str = "gpt-5") -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def call_gemini(prompt: str, model: str = "gemini-2.5-pro") -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
    }).encode()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


PROVIDERS = {
    "Claude": call_claude,
    "GPT": call_openai,
    "Gemini": call_gemini,
}


def review_file(filepath: str, criteria: str) -> dict:
    with open(filepath, "r") as f:
        code = f.read()

    prompt = REVIEW_PROMPT_TEMPLATE.format(
        criteria=criteria,
        filename=os.path.basename(filepath),
        code=code,
    )

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fn, prompt): name
            for name, fn in PROVIDERS.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                output = future.result()
                if output is None:
                    results[name] = "_(skipped — no API key set for this provider)_"
                else:
                    results[name] = output
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                results[name] = f"_(request failed: {e})_"
            except Exception as e:
                results[name] = f"_(unexpected error: {e})_"

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True, help="Source files to review")
    parser.add_argument("--criteria", required=True, help="Pass/fail criteria to grade against")
    parser.add_argument("--out", default="review_report.md", help="Output markdown path")
    args = parser.parse_args()

    active_providers = [name for name, fn in PROVIDERS.items()
                         if os.environ.get({
                             "Claude": "ANTHROPIC_API_KEY",
                             "GPT": "OPENAI_API_KEY",
                             "Gemini": "GEMINI_API_KEY",
                         }[name])]
    if not active_providers:
        print("No API keys found in environment. Set at least one of "
              "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    print(f"Active providers: {', '.join(active_providers)}", file=sys.stderr)

    report_lines = [
        f"# Multi-model code review",
        f"",
        f"Generated: {datetime.datetime.now().isoformat()}",
        f"Criteria: {args.criteria}",
        f"Providers queried: {', '.join(active_providers)}",
        f"",
    ]

    for filepath in args.files:
        print(f"Reviewing {filepath}...", file=sys.stderr)
        results = review_file(filepath, args.criteria)
        report_lines.append(f"## {filepath}")
        report_lines.append("")
        for provider_name in PROVIDERS:
            report_lines.append(f"### {provider_name}")
            report_lines.append("")
            report_lines.append(results.get(provider_name, "_(no result)_"))
            report_lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Wrote report to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
