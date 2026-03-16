"""
Generate PNG icons for the PWA from the SVG source.
Run once: python generate_icons.py
Requires: pip install cairosvg  (or pillow + cairosvg)
Falls back to a simple PIL-drawn icon if cairosvg is unavailable.
"""
import os
import struct
import zlib

ICON_DIR = os.path.join(os.path.dirname(__file__), "static", "icons")


def make_simple_png(size: int, path: str) -> None:
    """
    Create a minimal PNG icon using only the stdlib (no Pillow / cairosvg).
    Draws a dark rounded square with a green heart outline.
    """
    # We'll create a simple solid-colour PNG as a reliable fallback.
    # For a real deployment, regenerate with cairosvg for a proper icon.

    w = h = size
    bg = (10, 10, 26, 255)   # --bg colour
    fg = (0, 255, 136, 255)  # --primary colour

    # Build raw RGBA pixels
    pixels = bytearray(bg * w for _ in range(h))  # type: ignore[arg-type]
    # Actually build it properly:
    pixels = bytearray()
    for y in range(h):
        for x in range(w):
            # Simple heart shape via mathematical formula
            nx = (x / w) * 2 - 1   # -1..1
            ny = (y / h) * 2 - 1   # -1..1  (0 at top)
            ny = -ny                 # flip so heart points up

            # Heart curve: (x²+y²-1)³ - x²y³ < 0
            val = (nx**2 + ny**2 - 1)**3 - nx**2 * ny**3
            # Ring (hollow heart): |val| < threshold
            ring_outer = val < 0.02
            ring_inner = val < -0.04
            on_heart = ring_outer and not ring_inner

            # Border radius mask
            rx, ry = x / w, y / h
            corner_r = 0.18
            in_corner = False
            for cx, cy in [(corner_r, corner_r), (1-corner_r, corner_r),
                           (corner_r, 1-corner_r), (1-corner_r, 1-corner_r)]:
                if ((rx - cx)**2 + (ry - cy)**2) > corner_r**2 and \
                   rx < corner_r*2 and ry < corner_r*2 or \
                   rx > 1-corner_r*2 and ry < corner_r*2 or \
                   rx < corner_r*2 and ry > 1-corner_r*2 or \
                   rx > 1-corner_r*2 and ry > 1-corner_r*2:
                    pass  # simplified - just use full square

            if on_heart:
                pixels += bytes(fg)
            else:
                pixels += bytes(bg)

    # Encode as PNG
    def make_png(w: int, h: int, pixels: bytes) -> bytes:
        def chunk(name: bytes, data: bytes) -> bytes:
            c = name + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        # PNG uses RGB (not RGBA) for colour type 2; use type 6 for RGBA
        ihdr = chunk(b"IHDR", struct.pack(">II", w, h) + bytes([8, 6, 0, 0, 0]))

        raw_rows = b""
        for y in range(h):
            raw_rows += b"\x00"  # filter type None
            raw_rows += pixels[y * w * 4: (y + 1) * w * 4]
        idat = chunk(b"IDAT", zlib.compress(raw_rows, 9))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    png_data = make_png(w, h, bytes(pixels))
    with open(path, "wb") as f:
        f.write(png_data)
    print(f"  Written {path} ({size}×{size})")


def try_cairosvg(svg_path: str, size: int, out_path: str) -> bool:
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(url=svg_path, write_to=out_path,
                         output_width=size, output_height=size)
        print(f"  Written {out_path} ({size}×{size}) via cairosvg")
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    os.makedirs(ICON_DIR, exist_ok=True)
    svg_src = os.path.join(ICON_DIR, "icon.svg")

    for sz, name in [(192, "icon-192.png"), (512, "icon-512.png")]:
        out = os.path.join(ICON_DIR, name)
        if not try_cairosvg(svg_src, sz, out):
            print(f"  cairosvg not available — generating fallback PNG for {name}")
            make_simple_png(sz, out)

    print("Done.")
