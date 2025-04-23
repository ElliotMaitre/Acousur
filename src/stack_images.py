from pathlib import Path
import numpy as np
import rasterio

# Base directory where year folders are located
BASE_DIR = Path("/home/cloud-user/work/data/acousur")  # <-- change this
YEARS = ["2020", "2021", "2022", "2023"]
POLARIZATIONS = ["ImagesTif_VH", "ImagesTif_VV"]


def get_max_dimensions(image_dir: Path):
    max_x, max_y = 0, 0
    for img_path in image_dir.glob("*.tif"):
        with rasterio.open(img_path) as src:
            h, w = src.height, src.width
            max_x = max(max_x, h)
            max_y = max(max_y, w)
    return max_x, max_y


def stack_images(image_dir: Path, target_shape):
    max_x, max_y = target_shape
    stack = []

    for img_path in sorted(image_dir.glob("*.tif")):
        with rasterio.open(img_path) as src:
            img = src.read(1)
            padded = np.zeros((max_x, max_y), dtype=img.dtype)
            padded[: img.shape[0], : img.shape[1]] = img
            stack.append(padded)

    return np.stack(stack)


def main():
    for year in YEARS:
        for pol in POLARIZATIONS:
            folder_path = BASE_DIR / year / pol
            print(f"Processing {folder_path}...")
            try:
                target_shape = get_max_dimensions(folder_path)
                print(f"Max shape for {year}-{pol}: {target_shape}")

                image_stack = stack_images(folder_path, target_shape)

                pol_label = "VH" if "VH" in pol else "VV"
                save_path = BASE_DIR / f"{year}_{pol_label}.npy"
                np.save(save_path, image_stack)

                print(f"Saved {save_path} with shape {image_stack.shape}")
            except Exception as e:
                print(f"Failed to process {folder_path}: {e}")


if __name__ == "__main__":
    main()
