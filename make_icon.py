"""
Generates icon.ico for PhotoSelector.
Run once before building: python make_icon.py
"""
import math
from PIL import Image, ImageDraw

def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size  # shorthand

    # ── Background: dark rounded square ──────────────────────────────────────
    radius = s // 6
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=radius,
                        fill=(22, 22, 36, 255))

    # ── Camera body ───────────────────────────────────────────────────────────
    bx1, by1 = s * 0.10, s * 0.38
    bx2, by2 = s * 0.90, s * 0.80
    d.rounded_rectangle([bx1, by1, bx2, by2],
                        radius=max(2, s // 14),
                        fill=(210, 215, 225, 255))

    # Viewfinder bump
    vw, vh = s * 0.22, s * 0.10
    vx = (s - vw) / 2
    vy = by1 - vh + 2
    d.rounded_rectangle([vx, vy, vx + vw, by1 + 3],
                        radius=max(1, s // 22),
                        fill=(210, 215, 225, 255))

    # ── Lens ──────────────────────────────────────────────────────────────────
    cx = s * 0.46
    cy = (by1 + by2) / 2
    r_outer = s * 0.215
    r_mid   = s * 0.145
    r_inner = s * 0.075
    r_hi    = s * 0.045

    def circle(cx, cy, r, fill):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    circle(cx, cy, r_outer, (55,  60,  80, 255))   # rim
    circle(cx, cy, r_mid,   (30,  32,  50, 255))   # barrel
    circle(cx, cy, r_inner, (20,  22,  38, 255))   # glass

    # specular highlight
    hx = cx - r_outer * 0.28
    hy = cy - r_outer * 0.28
    circle(hx, hy, r_hi, (200, 215, 255, 160))

    # ── Flash dot ─────────────────────────────────────────────────────────────
    fd = s * 0.055
    fx, fy = s * 0.76, by1 + (by2 - by1) * 0.28
    circle(fx, fy, fd, (255, 240, 180, 255))

    # ── Green selection badge (bottom-right) ──────────────────────────────────
    br  = s * 0.205
    bcx = s * 0.755
    bcy = s * 0.755

    # shadow ring
    circle(bcx, bcy, br + s * 0.02, (0, 0, 0, 120))
    # badge fill
    circle(bcx, bcy, br, (0, 185, 100, 255))

    # checkmark (only meaningful at ≥ 24 px)
    if s >= 24:
        lw = max(1, round(s * 0.055))
        arm = br * 0.58
        p1 = (bcx - arm * 0.72, bcy + arm * 0.05)
        p2 = (bcx - arm * 0.15, bcy + arm * 0.62)
        p3 = (bcx + arm * 0.72, bcy - arm * 0.52)
        d.line([p1, p2], fill=(255, 255, 255, 255), width=lw)
        d.line([p2, p3], fill=(255, 255, 255, 255), width=lw)

    return img


if __name__ == "__main__":
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [draw_icon(s) for s in sizes]

    # Save as .ico (all sizes embedded)
    frames[0].save(
        "icon.ico",
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[1:],
    )

    # Also save a preview PNG so you can eyeball it
    frames[-1].save("icon_preview.png")
    print(f"icon.ico written ({len(sizes)} sizes: {sizes})")
    print("icon_preview.png written (256×256)")
