#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib import error, request

CATEGORY_ORDER = (
    "whats_new",
    "improvements",
    "fixes",
    "breaking_changes",
    "upgrade_notes",
)
CATEGORY_TITLES = {
    "whats_new": "What's New",
    "improvements": "Improvements",
    "fixes": "Fixes",
    "breaking_changes": "Breaking Changes",
    "upgrade_notes": "Upgrade Notes",
}
TAG_VERSION_PATTERN = re.compile(r"^v?(\d+(?:\.\d+)*)$")
CONVENTIONAL_PREFIX_PATTERN = re.compile(r"^[a-z]+(?:\([^)]+\))?!?:\s*", re.IGNORECASE)
CONVENTIONAL_TYPE_PATTERN = re.compile(
    r"^(?P<type>[a-z]+)(?:\([^)]+\))?!?:\s*", re.IGNORECASE
)
OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
VERSION_ONLY_COMMIT_PATTERNS = (
    re.compile(r"\b(?:bump|update) version\b", re.IGNORECASE),
    re.compile(r"\brelease v?\d", re.IGNORECASE),
)


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"git {' '.join(args)} failed")
    return result


def _decode(output: bytes) -> str:
    return output.decode("utf-8", errors="replace").strip()


def _read_version_from_pyproject() -> str:
    pyproject_path = Path("pyproject.toml")
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def _parse_version(value: str) -> tuple[int, ...]:
    match = TAG_VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Unsupported version/tag format: {value}")
    return tuple(int(part) for part in match.group(1).split("."))


def _find_previous_tag(current_version: str) -> str | None:
    current_tuple = _parse_version(current_version)
    result = _run_git("tag", "--merged", "HEAD")
    candidates: list[tuple[tuple[int, ...], str]] = []
    for raw_tag in _decode(result.stdout).splitlines():
        tag = raw_tag.strip()
        if not tag:
            continue
        try:
            parsed = _parse_version(tag)
        except ValueError:
            continue
        if parsed < current_tuple:
            candidates.append((parsed, tag))

    if not candidates:
        return None

    candidates.sort()
    return candidates[-1][1]


def _commit_range(previous_tag: str | None) -> str:
    if previous_tag is None:
        return "HEAD"
    return f"{previous_tag}..HEAD"


def _list_commit_subjects(previous_tag: str | None) -> list[str]:
    git_range = _commit_range(previous_tag)
    result = _run_git("log", "--format=%s", "--reverse", "--no-merges", git_range)
    return [line for line in _decode(result.stdout).splitlines() if line]


def _list_changed_files(previous_tag: str | None) -> list[str]:
    git_range = _commit_range(previous_tag)
    result = _run_git("diff", "--name-only", git_range)
    return [line for line in _decode(result.stdout).splitlines() if line]


def _is_version_only_commit(subject: str) -> bool:
    lowered = subject.lower()
    if "pyproject.toml" not in lowered and "version" not in lowered:
        return False
    return any(pattern.search(subject) for pattern in VERSION_ONLY_COMMIT_PATTERNS)


def _normalize_subject(subject: str) -> str:
    cleaned = CONVENTIONAL_PREFIX_PATTERN.sub("", subject).strip()
    if not cleaned:
        cleaned = subject.strip()
    if cleaned and cleaned[-1] == ".":
        cleaned = cleaned[:-1]
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _classify_subject(subject: str) -> str | None:
    lowered = subject.lower()
    conventional_match = CONVENTIONAL_TYPE_PATTERN.match(subject.strip())
    commit_type = None
    if conventional_match is not None:
        commit_type = conventional_match.group("type").lower()

    if _matches_any(
        lowered, (r"\bbreaking\b", r"\bbreaking change", r"\bincompatible\b")
    ):
        return "breaking_changes"

    if _matches_any(
        lowered,
        (
            r"\bmigration\b",
            r"\bmigrate\b",
            r"\bupgrade note\b",
            r"\bdeprecated\b",
            r"\bdeprecate\b",
            r"\brename\b",
            r"\brenamed\b",
            r"\bclear cache\b",
            r"\bmanual action\b",
        ),
    ):
        return "upgrade_notes"

    # Pure internal/tooling — omit from output
    if commit_type in {"docs", "chore", "build", "ci", "style"} or _matches_any(
        lowered,
        (
            r"\bpre-commit\b",
            r"\bpublish\b",
            r"\brelease note",
            r"\bgithub action",
            r"\bautomation\b",
            r"\btooling\b",
        ),
    ):
        return None

    if commit_type == "fix" or _matches_any(
        lowered,
        (
            r"\bfix\b",
            r"\bbug\b",
            r"\bcorrect\b",
            r"\bregression\b",
            r"\bresolve\b",
        ),
    ):
        return "fixes"

    if commit_type in {"refactor", "perf"} or _matches_any(
        lowered,
        (
            r"\bperformance\b",
            r"\breliability\b",
            r"\bstability\b",
            r"\brobust\b",
            r"\boptimi[sz]e\b",
            r"\boptimized\b",
            r"\bcleanup\b",
            r"\bclean up\b",
            r"\breadability\b",
            r"\brefactor\b",
            r"\bcaching option\b",
        ),
    ):
        return "improvements"

    # Features, new options, API/CLI additions, metadata, outputs — all go here
    return "whats_new"


def _categorize_subjects(subjects: list[str]) -> dict[str, list[str]]:
    categories = {name: [] for name in CATEGORY_ORDER}
    seen: set[tuple[str, str]] = set()

    for subject in subjects:
        if _is_version_only_commit(subject):
            continue
        category = _classify_subject(subject)
        if category is None:
            continue
        normalized = _normalize_subject(subject)
        if not normalized:
            continue
        dedupe_key = (category, normalized)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        categories[category].append(normalized)

    return categories


def _format_changed_files(files: list[str]) -> list[str]:
    if not files:
        return ["- No files changed in this release range."]

    lines = [f"- {path}" for path in files[:15]]
    remaining = len(files) - 15
    if remaining > 0:
        lines.append(f"- ... and {remaining} more files")
    return lines


def _build_release_context(version: str, previous_tag: str | None) -> dict[str, object]:
    subjects = _list_commit_subjects(previous_tag)
    changed_files = _list_changed_files(previous_tag)
    categories = _categorize_subjects(subjects)
    return {
        "version": version,
        "previous_tag": previous_tag,
        "commit_range": _commit_range(previous_tag),
        "subjects": subjects,
        "changed_files": changed_files,
        "categories": categories,
    }


def _build_fallback_summary(categories: dict[str, list[str]]) -> str:
    populated_titles = [
        CATEGORY_TITLES[category_name]
        for category_name in CATEGORY_ORDER
        if categories[category_name]
    ]

    if not populated_titles:
        return "This release contains developer-facing maintenance updates."

    if len(populated_titles) == 1:
        return f"This release focuses on {populated_titles[0].lower()} for developers using the published package."

    if len(populated_titles) == 2:
        joined = " and ".join(title.lower() for title in populated_titles)
        return f"This release focuses on {joined} for developers using the published package."

    joined = ", ".join(title.lower() for title in populated_titles[:-1])
    return (
        "This release focuses on "
        f"{joined}, and {populated_titles[-1].lower()} for developers using the published package."
    )


def _has_entries(categories: dict[str, list[str]]) -> bool:
    return any(categories[category_name] for category_name in CATEGORY_ORDER)


def _render_release_notes_from_sections(
    version: str,
    previous_tag: str | None,
    summary: str,
    categories: dict[str, list[str]],
) -> str:

    lines = [f"## Release v{version}", ""]
    if previous_tag is not None:
        lines.append(f"Based on changes since `{previous_tag}`.")
    else:
        lines.append(
            "Based on all changes currently available in the repository history."
        )
    lines.append("")

    lines.append("### Summary")
    lines.append(summary)
    lines.append("")

    for category_name in CATEGORY_ORDER:
        entries = categories[category_name]
        if not entries:
            continue
        lines.append(f"### {CATEGORY_TITLES[category_name]}")
        lines.extend(f"- {entry}" for entry in entries)
        lines.append("")

    lines.append("### Install")
    lines.append("```shell")
    lines.append(f"pip install cruxes=={version}")
    lines.append("```")

    return "\n".join(lines).strip() + "\n"


def _render_fallback_release_notes(context: dict[str, object]) -> str:
    categories = context["categories"]
    assert isinstance(categories, dict)
    summary = _build_fallback_summary(categories)
    return _render_release_notes_from_sections(
        version=str(context["version"]),
        previous_tag=context["previous_tag"],
        summary=summary,
        categories=categories,
    )


def _build_openai_prompt(context: dict[str, object]) -> str:
    return (
        "You are writing developer-focused release notes for a Python package release. "
        "Write concise, concrete notes that summarize what shipped for developers. "
        "Avoid marketing language, avoid inventing details, and rely only on the provided git history context.\n\n"
        "Return valid JSON with this exact shape:\n"
        "{\n"
        '  "summary": "short paragraph",\n'
        '  "whats_new": ["bullet", "bullet"],\n'
        '  "improvements": ["bullet", "bullet"],\n'
        '  "fixes": ["bullet", "bullet"],\n'
        '  "breaking_changes": ["bullet", "bullet"],\n'
        '  "upgrade_notes": ["bullet", "bullet"]\n'
        "}\n\n"
        "Rules:\n"
        "- `summary` must be 1-2 sentences.\n"
        "- Each list item must be a concise developer-facing statement.\n"
        "- `whats_new` is for new features, new CLI flags/commands, new API options, new outputs, and new backends.\n"
        "- `improvements` is for speed, stability, robustness, refactoring, and enhancements to existing functionality.\n"
        "- `fixes` is for bugs and corrected behavior.\n"
        "- `breaking_changes` is only for incompatible changes that require consumer attention.\n"
        "- `upgrade_notes` is for migration guidance, renamed options, deprecations, cache reset notes, or manual upgrade actions.\n"
        "- Omit internal tooling, CI, docs-only, and version-bump commits.\n"
        "- Use empty arrays when a section has nothing worth mentioning.\n\n"
        f"Version: {context['version']}\n"
        f"Previous tag: {context['previous_tag']}\n"
        f"Commit range: {context['commit_range']}\n\n"
        "Commit subjects:\n"
        + json.dumps(context["subjects"], indent=2)
        + "\n\nChanged files:\n"
        + json.dumps(context["changed_files"], indent=2)
        + "\n\nDeterministic pre-classification:\n"
        + json.dumps(context["categories"], indent=2)
        + "\n"
    )


def _extract_response_text(payload: dict[str, object]) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        text_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text_value = content_item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
        if text_parts:
            return "\n".join(text_parts).strip()

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    raise RuntimeError("OpenAI response did not contain text output")


def _call_openai_release_notes(context: dict[str, object], model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    request_payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _build_openai_prompt(context),
                    }
                ],
            }
        ],
    }

    http_request = request.Request(
        OPENAI_API_URL,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=60) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: {exc.code} {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

    response_text = _extract_response_text(response_payload)
    llm_sections = json.loads(response_text)

    if not isinstance(llm_sections, dict):
        raise RuntimeError("OpenAI response JSON was not an object")

    summary = llm_sections.get("summary")
    categories: dict[str, list[str]] = {name: [] for name in CATEGORY_ORDER}
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("OpenAI response did not include a valid summary")

    for category_name in CATEGORY_ORDER:
        entries = llm_sections.get(category_name, [])
        if not isinstance(entries, list):
            raise RuntimeError(
                f"OpenAI response section '{category_name}' was not a list"
            )
        cleaned_entries = []
        for entry in entries:
            if isinstance(entry, str) and entry.strip():
                cleaned_entries.append(entry.strip())
        categories[category_name] = cleaned_entries

    if not _has_entries(categories):
        raise RuntimeError(
            "OpenAI response did not include any categorized release note entries"
        )

    return _render_release_notes_from_sections(
        version=str(context["version"]),
        previous_tag=context["previous_tag"],
        summary=summary.strip(),
        categories=categories,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate categorized release notes from git history."
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Release version. Defaults to project.version in pyproject.toml.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use the OpenAI API to generate a higher-level summary with deterministic fallback.",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        help="OpenAI model to use when --use-openai is enabled.",
    )
    args = parser.parse_args()

    try:
        version = args.version or _read_version_from_pyproject()
        previous_tag = _find_previous_tag(version)
        context = _build_release_context(version, previous_tag)
        if args.use_openai:
            try:
                sys.stdout.write(_call_openai_release_notes(context, args.openai_model))
                return 0
            except (json.JSONDecodeError, RuntimeError) as exc:
                print(
                    f"OpenAI release note generation failed, falling back to deterministic notes: {exc}",
                    file=sys.stderr,
                )

        sys.stdout.write(_render_fallback_release_notes(context))
        return 0
    except (
        KeyError,
        OSError,
        RuntimeError,
        tomllib.TOMLDecodeError,
        ValueError,
    ) as exc:
        print(f"Failed to generate release notes: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
