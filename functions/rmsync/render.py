"""Render reMarkable `.rm` v6 stroke data to a grayscale PNG.

Strokes are drawn straight to a Pillow canvas from rmscene's parsed scene items.
Going via SVG (rmc + cairosvg) would drag a native Cairo stack into the Lambda
layer for no benefit - the vision model needs legible ink, not a vector file.

Vision cost scales with image size, so the page is cropped to its ink bounding
box and rendered at the minimum legible width rather than full-page full-res.
"""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageDraw
from rmscene import SceneLineItemBlock, read_blocks
from rmscene import scene_items as si

logger = logging.getLogger(__name__)

# reMarkable page geometry, in page units.
PAGE_WIDTH = 1404
PAGE_HEIGHT = 1872
X_OFFSET = PAGE_WIDTH / 2  # .rm x coordinates are centred on 0

# Stored stroke width is 4x the page-unit width (see rmscene point_from_stream).
WIDTH_DIVISOR = 4.0
MIN_STROKE_PX = 1
MAX_STROKE_PX = 14
PADDING = 32

# Erasers remove ink; drawing them as strokes would corrupt the transcription.
SKIP_TOOLS = {si.Pen.ERASER, si.Pen.ERASER_AREA}
# Highlighter over text renders as a dark smear in grayscale and hurts OCR.
HIGHLIGHTER_TOOLS = {si.Pen.HIGHLIGHTER_1, si.Pen.HIGHLIGHTER_2}


def parse_lines(data: bytes) -> list[si.Line]:
    """Extract drawable lines from a `.rm` v6 blob."""
    lines: list[si.Line] = []
    for block in read_blocks(io.BytesIO(data)):
        if not isinstance(block, SceneLineItemBlock):
            continue
        line = getattr(block.item, "value", None)
        if line is None:  # a tombstone - the stroke was deleted
            continue
        if getattr(line, "tool", None) in SKIP_TOOLS:
            continue
        if len(getattr(line, "points", []) or []) >= 2:
            lines.append(line)
    return lines


def stroke_count(data: bytes) -> int:
    return len(parse_lines(data))


def _bounds(lines: list[si.Line]) -> tuple[float, float, float, float]:
    xs_min = ys_min = float("inf")
    xs_max = ys_max = float("-inf")
    for line in lines:
        for pt in line.points:
            x = pt.x + X_OFFSET
            xs_min, xs_max = min(xs_min, x), max(xs_max, x)
            ys_min, ys_max = min(ys_min, pt.y), max(ys_max, pt.y)
    return xs_min, ys_min, xs_max, ys_max


def render_page(
    data: bytes,
    *,
    width: int = 1400,
    blank_threshold: int = 3,
) -> bytes | None:
    """Render a page to PNG bytes, or None if it is effectively blank.

    Blank-page detection runs before anything expensive: trailing blank pages
    are common in notebooks, and skipping them avoids a model call entirely.
    """
    lines = parse_lines(data)
    if len(lines) < blank_threshold:
        logger.info("Skipping blank page (%d stroke(s) < threshold %d)", len(lines), blank_threshold)
        return None

    x0, y0, x1, y1 = _bounds(lines)
    if not all(map(lambda v: v == v and abs(v) != float("inf"), (x0, y0, x1, y1))):
        logger.warning("Page had strokes but no finite bounds; skipping")
        return None

    x0 = max(0.0, x0 - PADDING)
    y0 = max(0.0, y0 - PADDING)
    x1 = min(float(PAGE_WIDTH), x1 + PADDING)
    y1 = min(float(PAGE_HEIGHT), y1 + PADDING)
    span_x = max(1.0, x1 - x0)
    span_y = max(1.0, y1 - y0)

    scale = width / span_x
    out_w = max(1, int(round(span_x * scale)))
    out_h = max(1, int(round(span_y * scale)))

    # "L" (8-bit grayscale) rather than RGB: a third of the bytes, and colour
    # carries no information the model needs from handwriting.
    canvas = Image.new("L", (out_w, out_h), color=255)
    draw = ImageDraw.Draw(canvas)

    for line in lines:
        shade = 128 if line.tool in HIGHLIGHTER_TOOLS else 0
        pts = [((p.x + X_OFFSET - x0) * scale, (p.y - y0) * scale) for p in line.points]
        widths = [p.width for p in line.points if p.width]
        avg_width = (sum(widths) / len(widths)) if widths else 8.0
        stroke_px = int(
            round(
                max(
                    MIN_STROKE_PX,
                    min(
                        MAX_STROKE_PX,
                        (avg_width / WIDTH_DIVISOR) * scale * (line.thickness_scale or 1.0),
                    ),
                )
            )
        )
        draw.line(pts, fill=shade, width=stroke_px, joint="curve")

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    png = buf.getvalue()
    logger.info(
        "Rendered page: %d stroke(s) -> %dx%d PNG, %d bytes", len(lines), out_w, out_h, len(png)
    )
    return png
