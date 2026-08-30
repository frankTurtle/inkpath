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
import struct

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


# Pre-v6 ".lines" format. Notebooks created on older firmware keep this format
# forever, so a v6-only parser silently drops whole notebooks.
_LEGACY_HEADER_LEN = 43
_LEGACY_PREFIX = b"reMarkable .lines file, version="
# v5 line headers carry one extra int32 over v3.
_LEGACY_LINE_FMT = {3: "<IIIfI", 5: "<IIIfII"}
_LEGACY_POINT_FMT = "<ffffff"


def _legacy_version(data: bytes) -> int | None:
    """Return 3 or 5 for a pre-v6 blob, else None."""
    if not data.startswith(_LEGACY_PREFIX):
        return None
    try:
        version = int(data[:_LEGACY_HEADER_LEN].decode("ascii").split("=")[1].strip())
    except (ValueError, IndexError, UnicodeDecodeError):
        return None
    return version if version in _LEGACY_LINE_FMT else None


def _parse_legacy(data: bytes, version: int) -> list[si.Line]:
    """Parse a v3/v5 `.lines` blob into v6-shaped Line objects.

    Coordinates and widths are normalised to the v6 conventions the renderer
    expects: x is re-centred on 0, and width is scaled by WIDTH_DIVISOR, so
    everything downstream is version-agnostic.
    """
    line_fmt = _LEGACY_LINE_FMT[version]
    line_size = struct.calcsize(line_fmt)
    point_size = struct.calcsize(_LEGACY_POINT_FMT)

    offset = _LEGACY_HEADER_LEN
    lines: list[si.Line] = []
    try:
        (n_layers,) = struct.unpack_from("<I", data, offset)
        offset += 4
        for _ in range(n_layers):
            (n_lines,) = struct.unpack_from("<I", data, offset)
            offset += 4
            for _ in range(n_lines):
                fields = struct.unpack_from(line_fmt, data, offset)
                offset += line_size
                pen_code, colour = fields[0], fields[1]
                n_points = fields[-1]

                points = []
                for _ in range(n_points):
                    x, y, speed, direction, width, pressure = struct.unpack_from(
                        _LEGACY_POINT_FMT, data, offset
                    )
                    offset += point_size
                    points.append(
                        si.Point(
                            x=x - X_OFFSET,           # v6 centres x on 0
                            y=y,
                            speed=int(speed),
                            direction=int(direction),
                            width=int(width * WIDTH_DIVISOR),
                            pressure=int(pressure * 255),
                        )
                    )

                try:
                    tool = si.Pen(pen_code)
                except ValueError:
                    tool = si.Pen.FINELINER_2
                try:
                    color = si.PenColor(colour)
                except ValueError:
                    color = si.PenColor.BLACK

                if len(points) >= 2:
                    lines.append(
                        si.Line(
                            color=color,
                            tool=tool,
                            points=points,
                            thickness_scale=1.0,
                            starting_length=0.0,
                        )
                    )
    except struct.error as exc:
        # Truncated or unexpected layout: keep whatever parsed cleanly rather
        # than losing the page outright.
        logger.warning("Legacy v%d parse stopped early (%s); kept %d line(s)",
                       version, exc, len(lines))
    return lines


def parse_lines(data: bytes) -> list[si.Line]:
    """Extract drawable lines from a `.rm` blob, v3/v5 or v6."""
    legacy = _legacy_version(data)
    if legacy is not None:
        logger.info("Parsing legacy .lines v%d page", legacy)
        candidates = _parse_legacy(data, legacy)
    else:
        candidates = []
        for block in read_blocks(io.BytesIO(data)):
            if not isinstance(block, SceneLineItemBlock):
                continue
            line = getattr(block.item, "value", None)
            if line is None:  # a tombstone - the stroke was deleted
                continue
            candidates.append(line)

    lines: list[si.Line] = []
    for line in candidates:
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
