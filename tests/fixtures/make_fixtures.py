#!/usr/bin/env python3
"""Generate SYNTHETIC .rm v6 fixtures.

Deliberately synthetic: real reMarkable pages contain personal notes and must
never become test data in a public repository.

    python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import io
import math
import pathlib
import struct
from uuid import UUID

from rmscene import (
    AuthorIdsBlock,
    CrdtId,
    CrdtSequenceItem,
    LwwValue,
    MigrationInfoBlock,
    PageInfoBlock,
    SceneLineItemBlock,
    SceneTreeBlock,
    TreeNodeBlock,
    write_blocks,
)
from rmscene import scene_items as si

# Fixed UUID so regenerating the fixtures is byte-stable and diff-free.
AUTHOR_UUID = UUID("2b3c4d5e-6f70-4182-9394-a5b6c7d8e9fa")


def _line(points: list[tuple[float, float]]) -> si.Line:
    return si.Line(
        color=si.PenColor.BLACK,
        tool=si.Pen.FINELINER_2,
        points=[
            si.Point(x=x, y=y, speed=0, direction=0, width=20, pressure=100)
            for x, y in points
        ],
        thickness_scale=1.0,
        starting_length=0.0,
    )


def build(n_lines: int) -> bytes:
    """A page with `n_lines` wavy strokes, laid out like lines of handwriting."""
    blocks: list = [
        AuthorIdsBlock(author_uuids={1: AUTHOR_UUID}),
        MigrationInfoBlock(migration_id=CrdtId(1, 1), is_device=True),
        PageInfoBlock(
            loads_count=1, merges_count=0, text_chars_count=0, text_lines_count=0
        ),
        SceneTreeBlock(
            tree_id=CrdtId(0, 11),
            node_id=CrdtId(0, 0),
            is_update=True,
            parent_id=CrdtId(0, 1),
        ),
        TreeNodeBlock(si.Group(node_id=CrdtId(0, 1))),
        TreeNodeBlock(
            si.Group(node_id=CrdtId(0, 11), label=LwwValue(CrdtId(0, 12), "Layer 1"))
        ),
    ]
    for i in range(n_lines):
        pts = [(100.0 + j * 10, 100.0 + i * 40 + 8 * math.sin(j / 2)) for j in range(20)]
        blocks.append(
            SceneLineItemBlock(
                parent_id=CrdtId(0, 11),
                item=CrdtSequenceItem(
                    item_id=CrdtId(1, 20 + i),
                    left_id=CrdtId(0, 0),
                    right_id=CrdtId(0, 0),
                    deleted_length=0,
                    value=_line(pts),
                ),
            )
        )
    buf = io.BytesIO()
    write_blocks(buf, blocks)
    return buf.getvalue()


# --------------------------------------------------------------- legacy v5 --

LEGACY_HEADER_V5 = b"reMarkable .lines file, version=5          "


def build_legacy_v5(n_lines: int) -> bytes:
    """A pre-v6 `.lines` page.

    Notebooks created on older firmware keep this format forever, so the
    renderer must handle it. Note v5 x coordinates are absolute (0..1404), not
    centred on 0 like v6.
    """
    out = bytearray(LEGACY_HEADER_V5)
    assert len(out) == 43, len(out)
    out += struct.pack("<I", 1)              # one layer
    out += struct.pack("<I", n_lines)
    for i in range(n_lines):
        pts = [(802.0 + j * 10, 100.0 + i * 40 + 8 * math.sin(j / 2)) for j in range(20)]
        # pen, colour, unknown, width, unknown, n_points
        out += struct.pack("<IIIfII", 4, 0, 0, 2.0, 0, len(pts))
        for x, y in pts:
            out += struct.pack("<ffffff", x, y, 1.0, 0.0, 2.0, 0.5)
    return bytes(out)


def main() -> None:
    here = pathlib.Path(__file__).parent
    # 5 strokes: a normal page. 1 stroke: below BlankPageThreshold, so it must
    # be skipped before any model call.
    (here / "synthetic_page.rm").write_bytes(build(5))
    (here / "synthetic_blank.rm").write_bytes(build(1))
    (here / "synthetic_legacy_v5.rm").write_bytes(build_legacy_v5(5))
    print(
        "wrote synthetic_page.rm (v6, 5 strokes), synthetic_blank.rm (v6, 1 stroke) "
        "and synthetic_legacy_v5.rm (v5, 5 strokes)"
    )


if __name__ == "__main__":
    main()
