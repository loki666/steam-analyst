"""CSS the Streamlit theme can't express: CJK font fallbacks, mono metric values, card header
type, neutral chat avatars and inline code colour. Injected once per run by `render`."""

from __future__ import annotations

ACCENT = "#2340C8"
PANEL = "#E8ECE5"
INK = "#18201D"
MUTED = "#55605B"

# Review text includes Chinese and Japanese; Public Sans has no CJK glyphs.
BODY_FONTS = '"Public Sans", "Noto Sans CJK SC", "Noto Sans", system-ui, sans-serif'
MONO_FONTS = '"JetBrains Mono", "Noto Sans Mono CJK SC", ui-monospace, monospace'

_USER = '[data-testid="stChatMessage"]:has([aria-label="Chat message from user"])'
_BOT = '[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"])'
_AVATAR = '[data-testid="stChatMessageAvatarCustom"]'

CSS = f"""
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stChatInputTextArea"], [data-testid="stMetricLabel"] {{
  font-family: {BODY_FONTS};
}}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
  font-family: {MONO_FONTS} !important;
  font-size: 28px !important;
}}
.sga-title {{
  font-family: "Archivo", {BODY_FONTS};
  font-weight: 700;
  font-size: 1.05rem;
  margin: 0 0.4rem;
}}
.sga-meta {{
  font-family: {MONO_FONTS};
  font-size: 0.8rem;
  color: {MUTED};
}}
{_USER} {_AVATAR} {{ background: {PANEL}; color: {INK}; }}
{_BOT} {_AVATAR} {{ background: {ACCENT}; }}
{_BOT} {_AVATAR} span {{ color: #FFFFFF !important; }}
[data-testid="stMarkdownContainer"] :not(pre) > code {{ color: inherit !important; }}
"""
