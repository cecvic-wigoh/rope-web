"""Component registry, theme and stylesheet for the Rope web UI.

The layout follows FaceFusion's three-column web interface (options |
source/target/output | live preview + face controls); components register
themselves by name so other components can wire events to them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

UI_COMPONENTS: dict[str, Any] = {}


def register(name: str, component: Any) -> Any:
    UI_COMPONENTS[name] = component
    return component


def get_component(name: str) -> Any | None:
    return UI_COMPONENTS.get(name)


def get_components(names: list[str]) -> list[Any]:
    return [UI_COMPONENTS[n] for n in names if n in UI_COMPONENTS]


def theme() -> gr.Theme:
    neutral = gr.themes.colors.neutral
    return gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        secondary_hue=neutral,
        neutral_hue=neutral,
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_md,
        font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        background_fill_primary="*neutral_100",
        background_fill_primary_dark="*neutral_950",
        background_fill_secondary="*neutral_50",
        background_fill_secondary_dark="*neutral_800",
        block_background_fill="white",
        block_background_fill_dark="*neutral_900",
        block_border_width="0px",
        block_padding="0.5rem",
        block_label_background_fill="*neutral_100",
        block_label_background_fill_dark="*neutral_800",
        block_label_text_color="*neutral_700",
        block_label_text_color_dark="white",
        block_label_text_weight="600",
        block_label_margin="0.5rem",
        block_title_text_weight="600",
        block_title_text_color="*neutral_700",
        block_title_background_fill="*neutral_100",
        block_title_background_fill_dark="*neutral_800",
        border_color_primary="transparent",
        border_color_primary_dark="transparent",
        border_color_accent="transparent",
        border_color_accent_dark="transparent",
        input_background_fill="*neutral_50",
        input_background_fill_dark="*neutral_800",
        checkbox_label_background_fill="*neutral_50",
        checkbox_label_background_fill_dark="*neutral_800",
        checkbox_label_background_fill_selected="*primary_500",
        checkbox_label_background_fill_selected_dark="*primary_600",
        checkbox_label_text_color_selected="white",
        checkbox_label_text_color_selected_dark="white",
        checkbox_background_color_selected="*primary_600",
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_dark="*primary_600",
        button_primary_text_color="white",
        button_secondary_background_fill="white",
        button_secondary_background_fill_dark="*neutral_800",
        slider_color="*primary_500",
        slider_color_dark="*primary_600",
        shadow_drop="none",
        error_text_color="*primary_500",
    )


def css() -> str:
    return (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")
