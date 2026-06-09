#!/usr/bin/env python3
"""Validate Xenon skills and sync host-specific metadata.

This is intentionally small. Canonical skill content lives in `.agents/skills`;
this script only checks the local conventions and emits Codex-facing metadata
that should not have to be hand-maintained.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
CLAUDE_SKILLS = REPO_ROOT / ".claude" / "skills"
LOCK_FILE = REPO_ROOT / "skills-lock.json"


@dataclass(frozen=True)
class Skill:
    name: str
    path: Path
    body: str
    metadata: dict[str, str]

    @property
    def skill_md(self) -> Path:
        return self.path / "SKILL.md"

    @property
    def openai_yaml(self) -> Path:
        return self.path / "agents" / "openai.yaml"


def parse_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    try:
        _, raw_meta, body = text.split("---\n", 2)
    except ValueError as exc:
        raise ValueError(f"{path}: malformed YAML frontmatter") from exc

    metadata: dict[str, str] = {}
    for line in raw_meta.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{path}: unsupported frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, body


def discover_skills() -> list[Skill]:
    skills: list[Skill] = []
    for skill_dir in sorted(p for p in SKILLS_ROOT.iterdir() if p.is_dir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        metadata, body = parse_frontmatter(skill_md)
        skills.append(
            Skill(
                name=metadata.get("name", skill_dir.name),
                path=skill_dir,
                body=body,
                metadata=metadata,
            )
        )
    return skills


def locked_skill_names() -> set[str]:
    if not LOCK_FILE.exists():
        return set()
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    return set(data.get("skills", {}).keys())


def title_from_name(name: str) -> str:
    special = {"llm": "LLM", "v2": "v2"}
    return " ".join(special.get(part, part.capitalize()) for part in name.split("-"))


def first_sentence(description: str) -> str:
    match = re.match(r"(.+?[.!?])(?:\s|$)", description)
    return match.group(1) if match else description


def ui_short_description(description: str) -> str:
    sentence = first_sentence(description).rstrip(".")
    for prefix in ("Use when ", "Use this skill when "):
        if sentence.startswith(prefix):
            sentence = sentence[len(prefix) :]
            break
    return shorten(sentence[:1].upper() + sentence[1:])


def shorten(text: str, limit: int = 64) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    words: list[str] = []
    current = 0
    for word in text.split():
        next_len = current + len(word) + (1 if words else 0)
        if next_len > limit - 1:
            break
        words.append(word)
        current = next_len
    return (" ".join(words) or text[: limit - 3]).rstrip(".,;:") + "..."


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def generated_openai_yaml(skill: Skill) -> str:
    description = skill.metadata.get("description", "")
    short_description = ui_short_description(description)
    prompt = f"Use ${skill.name} to apply this Xenon workflow to the current task."
    return "\n".join(
        [
            "interface:",
            f"  display_name: {yaml_quote(title_from_name(skill.name))}",
            f"  short_description: {yaml_quote(short_description)}",
            f"  default_prompt: {yaml_quote(prompt)}",
            "",
            "policy:",
            "  allow_implicit_invocation: true",
            "",
        ]
    )


def check_claude_host(errors: list[str]) -> None:
    if not CLAUDE_SKILLS.is_symlink():
        errors.append(".claude/skills must be a symlink to ../.agents/skills")
        return
    target = Path(CLAUDE_SKILLS.readlink())
    if target != Path("../.agents/skills"):
        errors.append(f".claude/skills points to {target}, expected ../.agents/skills")


def sync_claude_host() -> None:
    CLAUDE_SKILLS.parent.mkdir(parents=True, exist_ok=True)
    if CLAUDE_SKILLS.exists() or CLAUDE_SKILLS.is_symlink():
        if not CLAUDE_SKILLS.is_symlink():
            raise RuntimeError(".claude/skills exists but is not a symlink")
        if Path(CLAUDE_SKILLS.readlink()) != Path("../.agents/skills"):
            CLAUDE_SKILLS.unlink()
            CLAUDE_SKILLS.symlink_to("../.agents/skills")
        return
    CLAUDE_SKILLS.symlink_to("../.agents/skills")


def relative_markdown_links(skill: Skill) -> list[tuple[str, Path]]:
    links: list[tuple[str, Path]] = []
    pattern = re.compile(r"\[[^\]]+\]\(([^)#]+\.md)(?:#[^)]+)?\)")
    for raw_target in pattern.findall(skill.skill_md.read_text(encoding="utf-8")):
        if "://" in raw_target or raw_target.startswith("#"):
            continue
        links.append((raw_target, (skill.path / raw_target).resolve()))
    return links


def check_skill(skill: Skill, *, locked: bool, errors: list[str]) -> None:
    if skill.metadata.get("name") != skill.path.name:
        errors.append(f"{skill.skill_md}: frontmatter name must match directory name")
    description = skill.metadata.get("description", "")
    if len(description) < 40:
        errors.append(f"{skill.skill_md}: description is too short for reliable routing")

    for raw_target, target in relative_markdown_links(skill):
        if not target.exists():
            errors.append(f"{skill.skill_md}: relative markdown link does not resolve: {raw_target}")

    if locked:
        return

    if "## Gotchas" not in skill.body:
        errors.append(f"{skill.skill_md}: missing ## Gotchas")
    if "evidence_rung" not in skill.body:
        errors.append(f"{skill.skill_md}: missing evidence_rung artifact guidance")

    expected = generated_openai_yaml(skill)
    if not skill.openai_yaml.exists():
        errors.append(f"{skill.openai_yaml}: missing generated Codex metadata")
    elif skill.openai_yaml.read_text(encoding="utf-8") != expected:
        errors.append(f"{skill.openai_yaml}: generated Codex metadata is stale")


def cmd_check(_: argparse.Namespace) -> int:
    errors: list[str] = []
    locked = locked_skill_names()
    check_claude_host(errors)
    skills = discover_skills()
    for skill in skills:
        check_skill(skill, locked=skill.name in locked, errors=errors)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    local_count = sum(1 for skill in skills if skill.name not in locked)
    print(f"skills ok: {local_count} local, {len(locked)} locked")
    return 0


def cmd_sync_hosts(_: argparse.Namespace) -> int:
    sync_claude_host()
    locked = locked_skill_names()
    written = 0
    skipped = 0
    for skill in discover_skills():
        if skill.name in locked:
            skipped += 1
            continue
        skill.openai_yaml.parent.mkdir(parents=True, exist_ok=True)
        skill.openai_yaml.write_text(generated_openai_yaml(skill), encoding="utf-8")
        written += 1
    print(f"synced host metadata: {written} written, {skipped} locked skipped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    check = subparsers.add_parser("check", help="validate local skill conventions")
    check.set_defaults(func=cmd_check)

    sync_hosts = subparsers.add_parser(
        "sync-hosts",
        help="sync Claude symlink and generated Codex metadata",
    )
    sync_hosts.set_defaults(func=cmd_sync_hosts)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
