"""Tests for Markdown to Slack mrkdwn conversion."""

from __future__ import annotations

from bot.formatting import markdown_to_slack_mrkdwn


class TestBoldConversion:
    def test_double_asterisk_to_single(self):
        assert markdown_to_slack_mrkdwn("**bold**") == "*bold*"

    def test_double_underscore_bold_to_single_asterisk(self):
        assert markdown_to_slack_mrkdwn("__bold__") == "*bold*"

    def test_mixed_bold_and_text(self):
        result = markdown_to_slack_mrkdwn("hello **world** and __foo__")
        assert result == "hello *world* and *foo*"


class TestItalicConversion:
    def test_single_asterisk_italic_preserved(self):
        assert markdown_to_slack_mrkdwn("*italic*") == "_italic_"

    def test_single_underscore_italic_preserved(self):
        assert markdown_to_slack_mrkdwn("_italic_") == "_italic_"


class TestHeadingConversion:
    def test_h1_to_bold(self):
        assert markdown_to_slack_mrkdwn("# Heading") == "*Heading*"

    def test_h2_to_bold(self):
        assert markdown_to_slack_mrkdwn("## Heading") == "*Heading*"

    def test_h3_to_bold(self):
        assert markdown_to_slack_mrkdwn("### Heading") == "*Heading*"

    def test_heading_mid_text(self):
        result = markdown_to_slack_mrkdwn("intro\n## Section\ncontent")
        assert "*Section*" in result


class TestLinkConversion:
    def test_markdown_link_to_slack(self):
        assert markdown_to_slack_mrkdwn("[Google](https://google.com)") == "<https://google.com|Google>"

    def test_link_in_sentence(self):
        result = markdown_to_slack_mrkdwn("Visit [Google](https://google.com) now")
        assert "<https://google.com|Google>" in result


class TestCodePreservation:
    def test_inline_code_preserved(self):
        assert markdown_to_slack_mrkdwn("`code`") == "`code`"

    def test_code_block_preserved(self):
        text = "```python\nprint('hi')\n```"
        result = markdown_to_slack_mrkdwn(text)
        assert "```" in result
        assert "print('hi')" in result


class TestStrikethrough:
    def test_strikethrough_converted(self):
        assert markdown_to_slack_mrkdwn("~~deleted~~") == "~deleted~"


class TestPlainText:
    def test_plain_text_unchanged(self):
        assert markdown_to_slack_mrkdwn("hello world") == "hello world"

    def test_trailing_whitespace_stripped(self):
        assert markdown_to_slack_mrkdwn("hello  \n") == "hello"


class TestBlockquote:
    def test_blockquote_preserved(self):
        result = markdown_to_slack_mrkdwn("> quoted text")
        assert "> quoted text" in result


class TestListConversion:
    def test_unordered_list_preserved(self):
        text = "- item 1\n- item 2"
        result = markdown_to_slack_mrkdwn(text)
        assert "• item 1" in result
        assert "• item 2" in result

    def test_ordered_list_preserved(self):
        text = "1. first\n2. second"
        result = markdown_to_slack_mrkdwn(text)
        assert "1. first" in result
        assert "2. second" in result
