from PIL import Image
import imageio.v3 as iio
import os

avif_files = [
    "/workspaces/CV/static/office.avif",
    "/workspaces/CV/static/table.avif",
    "/workspaces/CV/static/threemen.avif"
]

output_folder = "static"

for avif_path in avif_files:
    img = iio.imread(avif_path)
    img_pil = Image.fromarray(img)
    base_name = os.path.splitext(os.path.basename(avif_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.jpg")


    img_pil.save(output_path, "JPEG")
    print(f"Saved: {output_path}")

