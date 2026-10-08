"""Shared look-and-feel: a dark, sports-analytics style and metric cards."""

from __future__ import annotations

from html import escape

import streamlit as st

_CSS = """
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  h1, h2, h3 { letter-spacing: -0.01em; }

  .sv-hero h1 { font-size: 2.4rem; margin-bottom: 0.1rem; }
  .sv-hero p  { color: #9AA7B8; font-size: 1.05rem; margin-top: 0; }
  .sv-accent  { color: #22D3EE; }

  .sv-card {
    background: #131B2A; border: 1px solid #22304A; border-radius: 10px;
    padding: 14px 16px; height: 100%;
  }
  .sv-card .sv-label {
    color: #8696AC; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
  }
  .sv-card .sv-value { color: #F1F5F9; font-size: 1.7rem; font-weight: 700; line-height: 1.25; }
  .sv-card .sv-sub   { color: #6B7A90; font-size: 0.78rem; }

  .sv-step {
    color: #22D3EE; font-size: 0.78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: -0.4rem;
  }
</style>
"""


def inject_styles() -> None:
    """Add the custom CSS once per page render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "") -> str:
    """Return HTML for one dashboard card. All text is HTML-escaped."""
    sub_html = f'<div class="sv-sub">{escape(sub)}</div>' if sub else ""
    return (
        '<div class="sv-card">'
        f'<div class="sv-label">{escape(label)}</div>'
        f'<div class="sv-value">{escape(value)}</div>'
        f"{sub_html}</div>"
    )


def step_label(text: str) -> None:
    """Small cyan heading such as 'STEP 1 · RACE DETAILS'."""
    st.markdown(f'<div class="sv-step">{escape(text)}</div>', unsafe_allow_html=True)
