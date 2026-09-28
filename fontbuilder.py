#!/usr/bin/env python3
"""
PNG image to font file converter

Usage:
    python fontbuilder.py input.png "ABC" output.ttf
    python fontbuilder.py input.png "ABC" output.ttf --name "MyFont"

The PNG should have dark glyphs on a white background, arranged left to right
(and optionally in multiple rows) in the same order as the characters argument.

when building the png file with all the glyphs, sometimes you may find your computer
is out of ram when exporting and it shows up blank. lower the dpi
(for example 120dpi - 60dpi)

"""

import sys
import os
import math
import argparse
import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

EM       = 1000
ASCENT   = 800
DESCENT  = -200

# A gap between two ink runs is treated as an inter-character break only if
# it's at least this many times wider than the smaller cluster of gaps in the
# same row (e.g. the dot-to-stem gap on 'i'). Below this ratio, the gaps in a
# row are treated as uniform and every run is assumed to be its own glyph.
MIN_SPLIT_RATIO = 2.0

# An ink run narrower than this fraction of the row's median run width is
# treated as a satellite mark (a dash in a dashed line, one dot in a spray of
# dots, a stroke of a multi-piece punctuation glyph like a curly '"', etc.)
# rather than a character of its own, and gets folded into its nearest
# neighbour -- see merge_small_runs(). This runs before the gap-based
# clustering above, since a row mixing many very different glyphs (an icon
# set, say) can have gap widths so varied that clustering on gap width alone
# can't reliably tell a multi-piece glyph's internal gaps apart from real
# gaps between glyphs.
SMALL_RUN_FRACTION = 0.2

# Separate, slightly more permissive fraction used only for deciding whether
# a gap is small enough to chain/attach satellite runs across. The pieces of
# a single multi-piece glyph (e.g. the two strokes of a decorative '"') can
# sit a bit further apart than SMALL_RUN_FRACTION alone would allow, without
# being anywhere near as wide as a real gap between two different glyphs.
SMALL_RUN_GAP_FRACTION = 0.3


# ---------------------------------------------------------------------------
# Vectorization parameters
# ---------------------------------------------------------------------------
# These control how raw pixel outlines get turned into smooth vector curves
# instead of jagged, pixel-staircase polygons.

VECTOR_PARAM_DEFAULTS = {
    # Integer upscale factor applied to the source image before tracing.
    # Tracing at a higher resolution gives the curve fitter far more points
    # to work with along rounded strokes, which is the main fix for "choppy"
    # output from small/low-res source images.
    "supersample": 4,
    # Gaussian blur sigma applied at the supersampled resolution, just before
    # re-thresholding. Softens the upscaled pixel edges so the re-threshold
    # produces a rounder boundary instead of a blocky one.
    "blur_sigma": 0.8,
    # Added on top of the auto-detected (Otsu) black/white cutoff before
    # binarizing, in 0-255 intensity units. Source images with soft/wide
    # antialiasing (a many-pixel-wide fade from ink to background, as opposed
    # to a crisp 1-2px edge) lose that whole fade to the background if
    # thresholded at the strict Otsu cutoff, visibly cropping and un-evening
    # curved strokes (most noticeable on round letters like "O"). Raising the
    # cutoff pulls more of that fade into "ink" so the traced outline matches
    # what the fringe visually looks like rather than just its solid core.
    "threshold_bias": 30,
    # Moving-average smoothing window (in simplified polygon points) applied
    # to each traced outline before corner detection. Removes residual
    # pixel-level noise so curves come out smooth rather than wobbly.
    "smoothing_window": 1, #ORIGINALL 5
    # cv2.approxPolyDP tolerance, in source-image pixels (scaled internally
    # by `supersample`). Lower = outline hugs the source more tightly (more
    # points, more faithful to noise); higher = fewer points, blockier shape.
    "curve_epsilon": 0.2, #ORIGINAl 1.2
    # Turning-angle threshold in degrees. Vertices where the outline turns
    # sharper than this are kept as real corners (straight joins); gentler
    # turns are treated as part of a smooth curve and rendered with a
    # quadratic Bezier run instead of straight line segments.
    "corner_angle_deg": 40.0,
}

#build the settings directory for how the outlines get vectorized -----------

def get_vector_params(**overrides):
    """Merge caller-supplied overrides (None values ignored) onto the defaults."""
    params = dict(VECTOR_PARAM_DEFAULTS)
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    return params


# load the image into opencv ------------------------------------------------

def load_binary(path, supersample=1, blur_sigma=0.0, threshold_bias=0):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    if supersample > 1:
        img = cv2.resize(img, None, fx=supersample, fy=supersample, interpolation=cv2.INTER_CUBIC)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    otsu_cutoff, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cutoff = min(254, otsu_cutoff + threshold_bias)
    _, binary = cv2.threshold(img, cutoff, 255, cv2.THRESH_BINARY_INV)
    return binary


# segmentation -----------------------------------------------------------

def find_split_gaps(gap_widths):
    """
    Given the widths of the gaps between consecutive ink runs in a row,
    return the indices of the gaps that separate distinct characters (as
    opposed to gaps between disconnected pieces of the same character, e.g.
    the dot and stem of 'i').

    Works by sorting the gap widths and finding the split point that
    maximizes the between-cluster variance (1D Otsu), giving a "small"
    (same-character) cluster and a "large" (between-character) cluster. This
    looks at the overall separation between the two clusters rather than just
    the adjacent pair of values, so it isn't fooled by noise within a
    cluster. If the two clusters aren't separated by at least MIN_SPLIT_RATIO,
    the gaps are treated as uniform and every gap is considered a character
    split.
    """
    if not gap_widths:
        return set()
    if len(gap_widths) == 1:
        return {0}

    order = sorted(range(len(gap_widths)), key=lambda i: gap_widths[i])
    sorted_vals = [gap_widths[i] for i in order]
    n = len(sorted_vals)

    best_var = -1.0
    best_pos = 0  # number of gaps (from the small end) considered "same-character"
    for pos in range(1, n):
        left, right = sorted_vals[:pos], sorted_vals[pos:]
        w_left, w_right = len(left) / n, len(right) / n
        mean_left = sum(left) / len(left)
        mean_right = sum(right) / len(right)
        between_var = w_left * w_right * (mean_right - mean_left) ** 2
        if between_var > best_var:
            best_var = between_var
            best_pos = pos

    small_max, large_min = sorted_vals[best_pos - 1], sorted_vals[best_pos]
    ratio = large_min / small_max if small_max > 0 else float("inf")
    if ratio < MIN_SPLIT_RATIO:
        # No convincing bimodal split -- assume every run is its own glyph.
        return set(range(len(gap_widths)))

    merged_indices = set(order[:best_pos])
    return set(i for i in range(len(gap_widths)) if i not in merged_indices)


def merge_small_runs(runs, fraction=SMALL_RUN_FRACTION, gap_fraction=SMALL_RUN_GAP_FRACTION):
    """
    Fold satellite runs (much narrower than the row's typical run) into
    their nearest neighbour, so a dashed line, a cloud of dots, a
    multi-stroke punctuation mark, or similar disconnected multi-piece glyph
    detail doesn't get segmented into extra fake glyphs.

    A run counts as "small" if it's narrower than `fraction` of the row's
    median run width. Consecutive small runs only chain together if the gap
    between them is below `gap_fraction` of that same median -- a separate,
    slightly more permissive cutoff than the size check, since two pieces of
    one glyph can sit a bit further apart than either piece is wide. Without
    this gap check, two unrelated small details on opposite sides of a real
    glyph gap (e.g. a dashed line ending one glyph and a dot cloud starting
    the next) could get bridged into one. Each resulting small-run chain is
    folded into whichever neighbouring non-small run is closer, as long as
    that gap is also below the gap cutoff; a chain with no near-enough
    neighbour is left as-is.
    """
    if len(runs) <= 1:
        return list(runs)

    widths = [x2 - x1 for x1, x2 in runs]
    sorted_widths = sorted(widths)
    median_width = sorted_widths[len(sorted_widths) // 2]
    cutoff = fraction * median_width
    gap_cutoff = gap_fraction * median_width
    is_small = [w < cutoff for w in widths]

    if not any(is_small) or all(is_small):
        return list(runs)

    merged = [list(r) for r in runs]
    n = len(merged)
    absorbed = [False] * n

    i = 0
    while i < n:
        if not is_small[i]:
            i += 1
            continue

        # Extend the chain while the next run is also small AND close enough
        # to still plausibly be the same detail.
        j = i
        while j + 1 < n and is_small[j + 1] and (merged[j + 1][0] - merged[j][1]) < gap_cutoff:
            j += 1
        j += 1  # chain is [i, j)

        # Chain members always combine into one run -- they were only
        # chained together because they're mutually close and small, which
        # is already the signal that they're pieces of the same glyph (e.g.
        # the two strokes of a decorative '"'). Whether that combined piece
        # additionally sits close enough to a bigger neighbour to attach to
        # (e.g. a lone dash right next to a letter) is a separate question,
        # decided below -- it must not gate whether the chain itself merges.
        if j - i > 1:
            merged[i][1] = merged[j - 1][1]
            for k in range(i + 1, j):
                absorbed[k] = True

        left = i - 1
        while left >= 0 and absorbed[left]:
            left -= 1
        right = j
        while right < n and absorbed[right]:
            right += 1

        candidates = []
        if left >= 0:
            gap_left = merged[i][0] - merged[left][1]
            if gap_left < gap_cutoff:
                candidates.append((gap_left, left))
        if right < n:
            gap_right = merged[right][0] - merged[i][1]
            if gap_right < gap_cutoff:
                candidates.append((gap_right, right))

        if candidates:
            target = min(candidates)[1]
            merged[target][0] = min(merged[target][0], merged[i][0])
            merged[target][1] = max(merged[target][1], merged[i][1])
            absorbed[i] = True
        i = j

    return sorted(tuple(merged[k]) for k in range(n) if not absorbed[k])


def find_character_groups(binary, supersample=1):
    """
    Return a list of [x1, x2, y1, y2, contour_list] sorted row by row,
    left to right. Handles images with multiple rows of glyphs.

    The number of glyphs is detected automatically from the image -- no
    expected count is required. See find_split_gaps() for how runs of ink
    are grouped into characters.

    `supersample` must match the factor `binary` was upscaled by (see
    load_binary), so the pixel-count thresholds below scale along with it.

    Row boundaries are found via horizontal projection: a band of image rows
    with zero dark pixels anywhere is a guaranteed row separator. This is more
    reliable than y-overlap heuristics when rows are closely spaced.
    """
    img_h = binary.shape[0]
    min_gap = max(3, round(3 * supersample))

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    outer_indices = [i for i, h in enumerate(hierarchy) if h[3] == -1]

    def significant(c):
        x, y, w, h = cv2.boundingRect(c)
        return w > 4 * supersample and h > 4 * supersample

    outer_indices = [i for i in outer_indices if significant(contours[i])]

    # --- Step 1: find row separators via horizontal projection ---
    # A fully-blank band (no dark pixels on any column) cleanly separates rows.
    row_proj = np.sum(binary > 0, axis=1)  # dark-pixel count per image row

    separators = []       # (y_start, y_end) of each blank band
    in_blank = False
    sep_start = 0
    for y in range(img_h):
        if row_proj[y] == 0:
            if not in_blank:
                in_blank = True
                sep_start = y
        else:
            if in_blank:
                in_blank = False
                if y - sep_start >= min_gap:    # ignore single-pixel gaps
                    separators.append((sep_start, y))
    if in_blank and img_h - sep_start >= min_gap:
        separators.append((sep_start, img_h))

    # Content "row regions" are the non-blank bands between separators
    row_regions = []
    prev = 0
    for s, e in separators:
        if s > prev:
            row_regions.append((prev, s))
        prev = e
    if prev < img_h:
        row_regions.append((prev, img_h))

    # --- Step 2: assign each outer contour to a row by its center_y ---
    rows = [[] for _ in row_regions]
    for oi in outer_indices:
        _, cy, _, ch = cv2.boundingRect(contours[oi])
        center_y = cy + ch // 2
        for ri, (ry_min, ry_max) in enumerate(row_regions):
            if ry_min <= center_y < ry_max:
                rows[ri].append(oi)
                break

    # --- Step 3: find symbol slots via column projection, assign contours ---
    # Column projection finds actual blank bands between symbols — no gap
    # threshold needed, so disconnected symbol parts don't cause chain merging.
    all_groups = []
    img_w = binary.shape[1]

    for ri, (ry_min, ry_max) in enumerate(row_regions):
        if not rows[ri]:
            continue

        col_proj = np.sum(binary[ry_min:ry_max, :] > 0, axis=0)
        runs = []
        in_run = False
        run_start = 0
        for x in range(img_w):
            if col_proj[x] > 0:
                if not in_run:
                    in_run = True
                    run_start = x
            else:
                if in_run:
                    in_run = False
                    runs.append((run_start, x))
        if in_run:
            runs.append((run_start, img_w))

        if not runs:
            continue

        # Fold satellite marks (dashes, dot clouds, ...) into their nearest
        # neighbour before clustering on gap width -- see merge_small_runs().
        runs = merge_small_runs(runs)

        # Detect character boundaries automatically from the gap widths.
        gap_widths = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        splits = find_split_gaps(gap_widths)

        symbol_slots = []
        slot_x1 = runs[0][0]
        for i, (rx1, rx2) in enumerate(runs):
            if i in splits:
                symbol_slots.append((slot_x1, rx2))
                slot_x1 = runs[i + 1][0]
        symbol_slots.append((slot_x1, runs[-1][1]))

        # Assign contours to slots by center_x
        groups = []
        for sx1, sx2 in symbol_slots:
            group_contours = []
            gx1, gx2, gy1, gy2 = img_w, 0, ry_max, ry_min

            for oi in rows[ri]:
                bx, by, bw, bh = cv2.boundingRect(contours[oi])
                if sx1 <= bx + bw // 2 < sx2:
                    group_contours.append((contours[oi], False))
                    gx1 = min(gx1, bx)
                    gx2 = max(gx2, bx + bw)
                    gy1 = min(gy1, by)
                    gy2 = max(gy2, by + bh)

                    child = hierarchy[oi][2]
                    while child != -1:
                        if significant(contours[child]):
                            group_contours.append((contours[child], True))
                            hx, hy, hw, hh = cv2.boundingRect(contours[child])
                            gy1 = min(gy1, hy)
                            gy2 = max(gy2, hy + hh)
                        child = hierarchy[child][0]

            if group_contours and gx1 < gx2:
                groups.append([gx1, gx2, gy1, gy2, group_contours])

        if groups:
            row_y1_tag = min(g[2] for g in groups)
            row_y2_tag = max(g[3] for g in groups)
            for g in groups:
                g.append(row_y1_tag)
                g.append(row_y2_tag)

        all_groups.extend(groups)

    return all_groups


#convert the rasterized images to vectors for font 

def smooth_pinned_runs(points, is_corner, iterations):
    """
    Smooth the interior of each run of consecutive non-corner points, using
    simple Laplacian averaging pulled toward each point's neighbours. Corner
    points are never moved, so runs stay pinned to their surrounding corners
    -- this is what keeps genuine corners crisp instead of getting chamfered
    away by smoothing, and keeps sharp back-and-forth detail (e.g. a zigzag)
    from collapsing, since every one of its vertices is its own corner.
    """
    n = len(points)
    pts = list(points)
    for _ in range(iterations):
        new_pts = list(pts)
        for i in range(n):
            if is_corner[i]:
                continue
            prev_x, prev_y = pts[(i - 1) % n]
            next_x, next_y = pts[(i + 1) % n]
            cx, cy = pts[i]
            new_pts[i] = (
                0.5 * cx + 0.25 * (prev_x + next_x),
                0.5 * cy + 0.25 * (prev_y + next_y),
            )
        pts = new_pts
    return pts


def classify_corners(points, angle_deg_threshold):
    """
    Return a bool per point: True where the outline turns sharper than
    `angle_deg_threshold` (a real corner, e.g. the tip of a serif), False
    where the turn is gentle enough to be part of a smooth curve.
    """
    n = len(points)
    if n < 3:
        return [True] * n

    threshold_rad = math.radians(angle_deg_threshold)
    is_corner = []
    for i in range(n):
        px, py = points[(i - 1) % n]
        cx, cy = points[i]
        nx, ny = points[(i + 1) % n]
        v1x, v1y = cx - px, cy - py
        v2x, v2y = nx - cx, ny - cy
        len1 = math.hypot(v1x, v1y)
        len2 = math.hypot(v2x, v2y)
        if len1 == 0 or len2 == 0:
            is_corner.append(True)
            continue
        dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (len1 * len2)))
        angle = math.acos(dot)
        is_corner.append(angle >= threshold_rad)
    return is_corner


def emit_smoothed_contour(pen, points, is_corner):
    """
    Draw a closed contour, using straight lines between real corners and
    quadratic Bezier curves (via TrueType's "implied on-curve midpoint"
    convention) through runs of non-corner points.
    """
    n = len(points)
    if n < 3:
        return False

    if not any(is_corner):
        is_corner = list(is_corner)
        is_corner[0] = True

    start = next(i for i in range(n) if is_corner[i])
    seq = [points[(start + k) % n] for k in range(n)]
    corner_seq = [is_corner[(start + k) % n] for k in range(n)]

    pen.moveTo(seq[0])
    off_curve_run = []
    for pt, corner in zip(seq[1:], corner_seq[1:]):
        if corner:
            if off_curve_run:
                pen.qCurveTo(*off_curve_run, pt)
                off_curve_run = []
            else:
                pen.lineTo(pt)
        else:
            off_curve_run.append(pt)
    if off_curve_run:
        pen.qCurveTo(*off_curve_run, seq[0])
    pen.closePath()
    return True


#convert a piece of a letter from pixels to font outline data -----

def contour_to_pen(contour, is_hole, glyph_x1, glyph_y2, scale, x_offset, y_offset, pen, vector_params):
    #vectorize one raw pixel contour and draw it into the pen
    eff_epsilon = vector_params["curve_epsilon"] * vector_params["supersample"]
    approx = cv2.approxPolyDP(contour, eff_epsilon, True)
    if len(approx) < 3:
        return False

    raw_pts = [(float(p[0][0]), float(p[0][1])) for p in approx]

    # Classify corners on the raw (unsmoothed) polygon first, so real corners
    # get a fair, un-blurred angle measurement -- see smooth_pinned_runs().
    is_corner = classify_corners(raw_pts, vector_params["corner_angle_deg"])

    window = vector_params["smoothing_window"]
    if any(is_corner):
        iterations = max(1, window // 2)
        smoothed_pts = smooth_pinned_runs(raw_pts, is_corner, iterations)
    else:
        smoothed_pts = raw_pts

    pts = []
    for px, py in smoothed_pts:
        fx = int((px - glyph_x1) * scale) + x_offset
        fy = int((glyph_y2 - py) * scale) + y_offset
        pts.append((fx, fy))

    # TrueType winding: outer = clockwise (Y-up), holes = counter-clockwise
    signed_area = sum(
        (pts[i][0] * pts[(i+1) % len(pts)][1] -
         pts[(i+1) % len(pts)][0] * pts[i][1])
        for i in range(len(pts))
    )
    is_cw = signed_area < 0

    if is_hole and is_cw:
        pts = pts[::-1]
        is_corner = is_corner[::-1]
    elif not is_hole and not is_cw:
        pts = pts[::-1]
        is_corner = is_corner[::-1]

    return emit_smoothed_contour(pen, pts, is_corner)


def group_to_glyph(group, vector_params, kerning=0.0):
    """
    Convert a [x1, x2, y1, y2, contour_list, row_y1, row_y2] group to a
    (glyph, advance) pair.

    All glyphs in the same row share one scale derived from the row's total
    height, so every letter/symbol keeps its original proportions relative to
    its neighbours. The bottom of the row maps to the baseline (y=0) and the
    top maps to ASCENT.

    `kerning` is extra space (in font design units, EM=1000) added between
    every pair of glyphs -- negative values tighten spacing. Split evenly
    onto each glyph's side bearings so the gap between any two glyphs is
    exactly the base spacing plus this amount, regardless of which two
    glyphs they are (this is uniform tracking, not per-pair kerning -- a
    raster source image gives no way to know that, say, "AV" should sit
    closer than "AH").
    """
    x1, x2, y1, y2, group_contours, row_y1, row_y2 = group
    px_width   = x2 - x1
    row_height = row_y2 - row_y1
    if row_height == 0 or px_width == 0:
        return None

    # One scale for the whole row — preserves relative sizes and stroke weights
    scale = ASCENT / row_height

    # Side bearing is split evenly left/right so the glyph sits centered in
    # its advance width instead of jammed against the left edge; `kerning`
    # adds (or removes) equal space on top of that base on each side.
    side_bearing = round(EM // 20 + kerning / 2)

    pen = TTGlyphPen(None)
    drew_anything = False
    for contour, is_hole in group_contours:
        # x1 aligns glyph to the left (before the bearing offset); row_y2
        # flips Y so the row bottom = baseline
        if contour_to_pen(contour, is_hole, x1, row_y2, scale, side_bearing, 0, pen, vector_params):
            drew_anything = True

    if not drew_anything:
        return None

    advance = int(px_width * scale) + 2 * side_bearing
    return pen.glyph(), advance, side_bearing


#default empty glyph for all characters not included in your font image--------

def make_notdef():
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((50, 700))
    pen.lineTo((450, 700))
    pen.lineTo((450, 0))
    pen.closePath()
    pen.moveTo((100, 50))
    pen.lineTo((400, 50))
    pen.lineTo((400, 650))
    pen.lineTo((100, 650))
    pen.closePath()
    return pen.glyph()


#font building function ---------------------------------------------------------

def build_font(char_glyph_map, output_path, family_name):
    glyph_names = [".notdef"] + list(char_glyph_map.keys())
    char_map    = {ord(ch): ch for ch in char_glyph_map if len(ch) == 1}

    fb = FontBuilder(EM, isTTF=True)
    fb.setupGlyphOrder(glyph_names)
    fb.setupCharacterMap(char_map)

    glyphs  = {".notdef": make_notdef()}
    metrics = {".notdef": (500, 50)}

    for name, (glyph, advance, lsb) in char_glyph_map.items():
        glyphs[name]  = glyph
        metrics[name] = (advance, lsb)

    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    ps_name = family_name.replace(" ", "-")
    fb.setupNameTable({
        "familyName":            family_name,
        "styleName":             "Regular",
        "uniqueFontIdentifier":  f"{family_name}:Regular:1.0",
        "fullName":              f"{family_name} Regular",
        "version":               "Version 1.0",
        "psName":                ps_name,
    })
    fb.setupOS2(
        sTypoAscender=ASCENT,
        sTypoDescender=DESCENT,
        usWinAscent=ASCENT,
        usWinDescent=abs(DESCENT),
        fsType=0,
        fsSelection=64,
    )
    fb.setupPost()

    base, ext = os.path.splitext(output_path)
    if ext:
        if ext.lower() in ('.woff', '.woff2'):
            fb.font.flavor = ext.lower()[1:]  # 'woff' or 'woff2'
        fb.font.save(output_path)
        return [output_path]

    written = []
    for out_ext, flavor in ((".ttf", None), (".otf", None), (".woff2", "woff2")):
        fb.font.flavor = flavor
        path = base + out_ext
        fb.font.save(path)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Convert a PNG of glyphs to a TTF font.")
    parser.add_argument("image",      help="Input PNG path")
    parser.add_argument("--output", nargs="?", default="testFont", help="Output .ttf path with NAME.ttf, also supports .otf and .woff2, if no file type is chosen, all will be output")
    parser.add_argument("--characters", "-c",
                        default="""abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.,;'\"<>/\\|[]{}`~?#$()&^%*-+_@!=""",
                        help="""by default, will attempt to build glyfs for each character in this order: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.,;'\"<>/\\|[]{}`~?#$()&^%%*-+_@!=""")
    parser.add_argument("--name",     default="nonameFont", help="Font family name, default is nonameFont")
    parser.add_argument("--supersample", type=int, default=None,
                        help="Upscale factor applied before tracing, to smooth pixelated "
                             f"source edges (default {VECTOR_PARAM_DEFAULTS['supersample']})")
    parser.add_argument("--blur-sigma", type=float, default=None,
                        help="Gaussian blur sigma applied at the supersampled resolution "
                             f"before re-thresholding (default {VECTOR_PARAM_DEFAULTS['blur_sigma']})")
    parser.add_argument("--smoothing-window", type=int, default=None,
                        help="Moving-average window (in outline points) used to smooth traced "
                             f"contours before curve fitting (default {VECTOR_PARAM_DEFAULTS['smoothing_window']})")
    parser.add_argument("--curve-epsilon", type=float, default=None,
                        help="Polygon simplification tolerance in source-image pixels; lower "
                             "= more faithful/curvy, higher = blockier/fewer points "
                             f"(default {VECTOR_PARAM_DEFAULTS['curve_epsilon']})")
    parser.add_argument("--corner-angle", type=float, default=None, dest="corner_angle_deg",
                        help="Turning-angle threshold in degrees; sharper turns become straight "
                             "corners, gentler turns become smooth Bezier curves "
                             f"(default {VECTOR_PARAM_DEFAULTS['corner_angle_deg']})")
    parser.add_argument("--threshold-bias", type=float, default=None,
                        help="Added to the auto black/white cutoff (0-255) before binarizing; "
                             "raise this if soft/wide antialiased edges are getting cropped or "
                             "uneven, lower it if strokes come out too fat/blurred "
                             f"(default {VECTOR_PARAM_DEFAULTS['threshold_bias']})")
    parser.add_argument("--kerning", type=float, default=0.0,
                        help="Extra space, in font design units (1000 units/em), added between "
                             "every pair of glyphs; negative values tighten spacing. This is a "
                             "uniform tracking adjustment applied to all glyphs equally, not "
                             "per-pair kerning -- a raster source image gives no way to know a "
                             "specific pair like 'AV' should sit closer than 'AH' (default 0)")
    args = parser.parse_args()

    vector_params = get_vector_params(
        supersample=args.supersample,
        blur_sigma=args.blur_sigma,
        smoothing_window=args.smoothing_window,
        curve_epsilon=args.curve_epsilon,
        corner_angle_deg=args.corner_angle_deg,
        threshold_bias=args.threshold_bias,
    )

    print(f"Loading {args.image} ...")
    binary = load_binary(args.image, supersample=vector_params["supersample"],
                          blur_sigma=vector_params["blur_sigma"],
                          threshold_bias=vector_params["threshold_bias"])

    n_chars = len(args.characters)

    print("Segmenting glyphs ...")
    groups = find_character_groups(binary, supersample=vector_params["supersample"])

    n_found = len(groups)
    print(f"  Auto-detected {n_found} glyph group(s), {n_chars} character(s) provided")

    if n_found != n_chars:
        print(f"  WARNING: count mismatch -- will process min({n_found}, {n_chars})")

    count = min(n_found, n_chars)
    char_glyph_map = {}

    for i in range(count):
        ch = args.characters[i]
        result = group_to_glyph(groups[i], vector_params, kerning=args.kerning)
        if result is None:
            print(f"  Skipping '{ch}' -- no usable contours")
            continue
        glyph, advance, lsb = result
        char_glyph_map[ch] = (glyph, advance, lsb)
        print(f"  '{ch}' -- advance {advance}")

    if not char_glyph_map:
        print("No glyphs processed. Exiting.")
        sys.exit(1)

    if " " not in char_glyph_map:
        #this ensures the space char is always added to font
        advances = sorted(advance for _, advance, _ in char_glyph_map.values())
        space_advance = advances[len(advances) // 2]  # median of the traced glyphs
        char_glyph_map[" "] = (TTGlyphPen(None).glyph(), space_advance, 0)
        print(f"  ' ' -- advance {space_advance} (synthetic, no ink)")

    print(f"Building font '{args.name}' ...")
    written = build_font(char_glyph_map, args.output, args.name)
    print(f"Done -> {', '.join(written)}")


if __name__ == "__main__":
    main()
