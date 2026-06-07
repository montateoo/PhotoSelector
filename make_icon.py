"""
Generates icon.ico for PhotoSelector from the source PNG.
Run once before building: python make_icon.py
"""
from pathlib import Path
from PIL import Image

SOURCE = Path(__file__).parent / "pngtree-photography-camera-icon-vector-png-image_7210634.png"

if __name__ == "__main__":
    src = Image.open(SOURCE).convert("RGBA")

    sizes = [16, 24, 32, 48, 64, 128, 256]

    # Pillow ICO: pass sizes to the save call on the largest source image;
    # it resamples internally — do NOT use append_images for ICO format.
    src.save(
        "icon.ico",
        format="ICO",
        sizes=[(s, s) for s in sizes],
    )

    src.resize((256, 256), Image.LANCZOS).save("icon_preview.png")
    print(f"icon.ico written ({len(sizes)} sizes: {sizes})")
    ico_kb = Path("icon.ico").stat().st_size / 1024
    print(f"icon.ico size: {ico_kb:.1f} KB")
    print("icon_preview.png written (256x256)")
