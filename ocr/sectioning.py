"""Deterministic crop + sectioning of an OCR'd page (no model, CPU-only).

Crop to the text bound (rotation-aware via each block's polygon corners) + MARGIN,
decide the section count from the A-paper rule (N = round(AR / target_ar)), and
place each cut in the whitespace gap between blocks so no text is split.
"""
import math
from typing import Any, Dict, List, Tuple

from PIL import Image

SQRT2 = math.sqrt(2)


def _merge(intervals: List[Tuple[float, float]]) -> List[List[float]]:
    """Merge overlapping (start, end) intervals -> occupied bands."""
    out: List[List[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _snap(p: float, gaps: List[Tuple[float, float]]) -> float:
    """Snap an ideal cut to the midpoint of the nearest inter-block gap."""
    if not gaps:
        return p
    def dist(g):
        return 0 if g[0] <= p <= g[1] else min(abs(p - g[0]), abs(p - g[1]))
    g = min(gaps, key=dist)
    return (g[0] + g[1]) / 2.0


def polygons_from_result(raw: Dict[str, Any], sx: float = 1.0, sy: float = 1.0):
    """Per-block corner polygons (scaled) from a raw_result.json dict. Uses the
    real polygon corners (rotation-aware); falls back to the axis-aligned bbox."""
    res = raw.get("res", raw) if isinstance(raw, dict) else {}
    polys = []
    for b in res.get("parsing_res_list", []) or []:
        pp = b.get("block_polygon_points")
        if isinstance(pp, list) and len(pp) >= 3:
            polys.append([(p[0] * sx, p[1] * sy) for p in pp])
        else:
            bb = b.get("block_bbox")
            if isinstance(bb, list) and len(bb) == 4:
                x1, y1, x2, y2 = bb
                polys.append([(x1 * sx, y1 * sy), (x2 * sx, y1 * sy),
                              (x2 * sx, y2 * sy), (x1 * sx, y2 * sy)])
    return polys


def section_page(image: "Image.Image", polys, margin: int = 20, target_ar: float = SQRT2):
    """Crop `image` to the text bound and split into N sections on block gaps.

    Returns (crop, sections, info). `sections` is a list of PIL images tiling the
    crop along its long axis; `info` carries the geometry (crop box, AR, N, cuts)."""
    W, H = image.size
    boxes = [(min(x for x, _ in p), min(y for _, y in p),
              max(x for x, _ in p), max(y for _, y in p)) for p in polys]
    minx, miny = min(b[0] for b in boxes), min(b[1] for b in boxes)
    maxx, maxy = max(b[2] for b in boxes), max(b[3] for b in boxes)

    cl, ct = max(0, round(minx - margin)), max(0, round(miny - margin))
    cr, cb = min(W, round(maxx + margin)), min(H, round(maxy + margin))
    crop = image.crop((cl, ct, cr, cb))
    cw, chh = cr - cl, cb - ct

    AR = max(cw, chh) / min(cw, chh)
    N = max(1, round(AR / target_ar))
    portrait = chh >= cw

    cuts: List[float] = []
    if N > 1:
        if portrait:
            ivs, c0, c1 = [(b[1], b[3]) for b in boxes], miny, maxy
        else:
            ivs, c0, c1 = [(b[0], b[2]) for b in boxes], minx, maxx
        occ = _merge(ivs)
        gaps = [(occ[i][1], occ[i + 1][0]) for i in range(len(occ) - 1)]
        for k in range(1, N):
            cuts.append(_snap(c0 + k / N * (c1 - c0), gaps))

    edges = ([ct] + [round(c) for c in cuts] + [cb]) if portrait \
        else ([cl] + [round(c) for c in cuts] + [cr])
    sections = []
    for a, b in zip(edges, edges[1:]):
        sec = crop.crop((0, a - ct, cw, b - ct)) if portrait \
            else crop.crop((a - cl, 0, b - cl, chh))
        sections.append(sec)

    info = {
        "crop_box": [cl, ct, cr, cb],
        "aspect_ratio": round(AR, 4),
        "target_ar": round(target_ar, 4),
        "section_count": N,
        "axis": "y" if portrait else "x",
        "cuts": [round(c) for c in cuts],
        "section_sizes": [list(s.size) for s in sections],
    }
    return crop, sections, info
