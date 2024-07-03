import argparse
from pathlib import Path
import concurrent.futures
import os

import cv2
from tqdm import tqdm
import numpy as np

def process_map(map_path, dest_dir):
    try:
        map_name = map_path.stem
        output_path = dest_dir / f"{map_name}.jpg"

        # Check if output file already exists
        if output_path.exists():
            return

        img = cv2.imread(str(map_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Image at {map_path} could not be read.")

        if len(img.shape) == 2 and img.dtype == np.uint16:
            # Normalize 16-bit to 8-bit
            img = (img / 65535.0 * 255).astype(np.uint8)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # Convert RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:  # already 8-bit single channel
            pass
        else:
            raise ValueError(f"Unsupported image format: {map_path}")

        # Resize to 1024x1024
        img_resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        
        # Convert back to BGR for saving as JPEG
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        # Save as JPG
        cv2.imwrite(str(output_path), img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    except Exception as e:
        print(f"Error processing {map_path}: {e}")

def combine_metallic_roughness(metallic_path, roughness_path, dest_dir):
    try:
        output_path = dest_dir / "metallic_roughness.jpg"

        # Check if output file already exists
        if output_path.exists():
            return

        metallic = cv2.imread(str(metallic_path), cv2.IMREAD_GRAYSCALE)
        roughness = cv2.imread(str(roughness_path), cv2.IMREAD_GRAYSCALE)
        
        if metallic is None:
            raise ValueError(f"Metallic image at {metallic_path} could not be read.")
        if roughness is None:
            raise ValueError(f"Roughness image at {roughness_path} could not be read.")
        
        # Resize images to 1024x1024
        metallic = cv2.resize(metallic, (1024, 1024), interpolation=cv2.INTER_AREA)
        roughness = cv2.resize(roughness, (1024, 1024), interpolation=cv2.INTER_AREA)
        
        # Create combined image
        combined = np.zeros((1024, 1024, 3), dtype=np.uint8)
        combined[:, :, 0] = metallic  # Blue channel
        combined[:, :, 1] = roughness  # Green channel
        
        # Save combined image as JPG
        cv2.imwrite(str(output_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    except Exception as e:
        print(f"Error combining {metallic_path} and {roughness_path}: {e}")

def process_material(file, dest_dir):
    try:
        mat_dir = file.parent
        name = mat_dir.stem
        category = mat_dir.parent.stem
        split = mat_dir.parent.parent.stem

        mat_dest = dest_dir / split / category / name
        mat_dest.mkdir(parents=True, exist_ok=True)

        metallic_path = None
        roughness_path = None

        for map_path in mat_dir.glob("*.png"):
            if "metallic" in map_path.stem:
                metallic_path = map_path
            elif "roughness" in map_path.stem:
                roughness_path = map_path
            process_map(map_path, mat_dest)

        if metallic_path and roughness_path:
            combine_metallic_roughness(metallic_path, roughness_path, mat_dest)
    except Exception as e:
        print(f"Error processing material in {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample and convert images to JPG.")
    parser.add_argument("--source_dir", required=True, help="Directory where the original maps are stored.")
    parser.add_argument("--dest_dir", required=True, help="Destination directory to store the downsampled JPGs.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)

    files = list(source_dir.glob("**/basecolor.png"))

    # Determine the number of CPU cores to use (50% of available cores)
    max_workers = max(1, os.cpu_count() // 2)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_material, files, [dest_dir]*len(files)), total=len(files)))
