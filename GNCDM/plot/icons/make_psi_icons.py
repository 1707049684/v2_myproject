"""Generate small pentagon radar icons for psi_old / psi_new."""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent


def pentagon_points(cx: float, cy: float, r: float, rotation_deg: float = -90):
    pts = []
    rot = math.radians(rotation_deg)
    for i in range(5):
        a = rot + i * 2 * math.pi / 5
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def poly_str(pts):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def make_svg(color: str, radii_profile: list[float], fname: str, label: str) -> Path:
    size = 200
    cx = cy = size / 2
    r_max = 78
    grid_rs = [0.28, 0.48, 0.68, 0.88, 1.0]
    grid_stroke = "#D9D2C5"
    axis_stroke = "#E6E0D6"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f"  <!-- {label} radar icon -->",
        f'  <rect width="{size}" height="{size}" fill="none"/>',
    ]
    for g in grid_rs:
        pts = pentagon_points(cx, cy, r_max * g)
        parts.append(
            f'  <polygon points="{poly_str(pts)}" fill="none" '
            f'stroke="{grid_stroke}" stroke-width="1.2"/>'
        )
    for x, y in pentagon_points(cx, cy, r_max):
        parts.append(
            f'  <line x1="{cx:.2f}" y1="{cy:.2f}" x2="{x:.2f}" y2="{y:.2f}" '
            f'stroke="{axis_stroke}" stroke-width="1"/>'
        )

    rot = math.radians(-90)
    data_pts = []
    for i, t in enumerate(radii_profile):
        a = rot + i * 2 * math.pi / 5
        rr = r_max * t
        data_pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))

    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    parts.append(
        f'  <polygon points="{poly_str(data_pts)}" fill="rgba({r},{g},{b},0.18)" '
        f'stroke="{color}" stroke-width="2.6" stroke-linejoin="round"/>'
    )
    for x, y in data_pts:
        parts.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"/>')
    parts.append("</svg>")

    path = OUT / fname
    path.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", path)
    return path


def main():
    old = make_svg(
        "#47A29A",
        [0.72, 0.38, 0.85, 0.45, 0.58],
        "psi_old.svg",
        "psi_old",
    )
    new = make_svg(
        "#D5BE6F",
        [0.42, 0.78, 0.35, 0.88, 0.55],
        "psi_new.svg",
        "psi_new",
    )

    w, h = 420, 210
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        '  <rect width="100%" height="100%" fill="white"/>',
    ]
    for i, (src, lab, col) in enumerate(
        [
            (old, "ψ_old", "#47A29A"),
            (new, "ψ_new", "#D5BE6F"),
        ]
    ):
        x0 = 10 + i * 210
        body = src.read_text(encoding="utf-8")
        inner = body.split(">", 1)[1].rsplit("</svg>", 1)[0]
        lines.append(f'  <g transform="translate({x0},0)">')
        lines.append(inner)
        lines.append(
            f'    <text x="100" y="198" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="15" '
            f'fill="{col}">{lab}</text>'
        )
        lines.append("  </g>")
    lines.append("</svg>")
    pair = OUT / "psi_old_new_pair.svg"
    pair.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", pair)


if __name__ == "__main__":
    main()
