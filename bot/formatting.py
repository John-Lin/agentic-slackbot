"""Convert standard Markdown to Slack-compatible mrkdwn."""

from __future__ import annotations

import re


def markdown_to_slack_mrkdwn(text: str) -> str:
    """Convert standard Markdown to Slack mrkdwn format.

    Handles: bold, italic, strikethrough, headings, links, lists.
    Preserves: code blocks, inline code, blockquotes.
    Returns the converted text with trailing whitespace stripped.
    """
    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        # Track code block boundaries — don't convert inside them
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        line = _convert_line(line)
        result.append(line)

    return "\n".join(result).strip()


def _convert_line(line: str) -> str:
    """Convert a single non-code-block line."""
    # Headings: ## Heading → *Heading*
    heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
    if heading_match:
        return f"*{heading_match.group(2).strip()}*"

    # Unordered list: - item → • item
    list_match = re.match(r"^(\s*)[-*]\s+(.+)$", line)
    if list_match:
        indent = list_match.group(1)
        return f"{indent}• {list_match.group(2)}"

    return _convert_inline(line)


def _convert_inline(text: str) -> str:
    """Convert inline Markdown formatting, preserving code spans."""
    # Split on inline code to avoid converting inside backticks
    parts = text.split("`")
    for i in range(0, len(parts), 2):  # Only process non-code parts
        if i < len(parts):
            parts[i] = _convert_formatting(parts[i])
    return "`".join(parts)


def _convert_formatting(text: str) -> str:
    """Convert bold, italic, strikethrough, and links in plain text."""
    # Links: [text](url) → <url|text>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

    # Italic first (single *): *text* → _text_
    # Must run before bold so **bold** isn't partially consumed
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", text)

    # Bold: **text** or __text__ → *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"__(.+?)__", r"*\1*", text)

    # Strikethrough: ~~text~~ → ~text~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)

    return text
