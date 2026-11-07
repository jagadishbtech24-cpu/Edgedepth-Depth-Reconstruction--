"""
Automated Dataset Downloader for EdgeDepth.
Downloads real, official benchmark RGB-D pairs from Middlebury Computer Vision:
- Cones
- Teddy
- Tsukuba
- Venus
Extracts and converts them into standardized RGB-D pairs in data/samples/
"""

import os
import sys
import zipfile
import urllib.request
import numpy as np
import cv2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

# Official Middlebury 2-View Benchmark Scenes with ground truth disparity
MIDDLEBURY_SCENES = {
    "cones": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones-png-2.zip",
    "teddy": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy-png-2.zip",
    "tsukuba": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/tsukuba-png-2.zip",
    "venus": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/venus-png-2.zip",
}

def download_file(url: str, dest_path: str):
    """Downloads a file with progress reporting."""
    print(f"Downloading from: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        total_size = int(response.getheader('Content-Length', 0))
        downloaded = 0
        block_size = 64 * 1024
        
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            if total_size > 0:
                percent = downloaded * 100 / total_size
                sys.stdout.write(f"\r[{percent:5.1f}%] {downloaded // 1024} KB / {total_size // 1024} KB")
                sys.stdout.flush()
    print("\nDownload finished.")

def download_and_extract_middlebury():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    temp_zip_dir = os.path.join(DATA_DIR, "temp_downloads")
    os.makedirs(temp_zip_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print(" FETCHING OFFICIAL REAL MIDDLEBURY BENCHMARK DATASET")
    print("=" * 65 + "\n")

    count = 0
    for scene_name, url in MIDDLEBURY_SCENES.items():
        print(f"\n---> Scene: {scene_name.upper()}")
        zip_path = os.path.join(temp_zip_dir, f"{scene_name}.zip")
        try:
            download_file(url, zip_path)
            
            extract_folder = os.path.join(temp_zip_dir, scene_name)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
                
            # Middlebury png-2 structure contains:
            # im2.png (Left image) and disp2.png (Ground truth disparity)
            rgb_candidates = ["im2.png", "view1.png", "im0.png"]
            disp_candidates = ["disp2.png", "disp1.png", "disp0.png"]
            
            found_rgb = None
            found_disp = None
            
            for root, dirs, files in os.walk(extract_folder):
                for f in files:
                    if f.lower() in [c.lower() for c in rgb_candidates] and not found_rgb:
                        found_rgb = os.path.join(root, f)
                    if f.lower() in [c.lower() for c in disp_candidates] and not found_disp:
                        found_disp = os.path.join(root, f)
                        
            if found_rgb and found_disp:
                bgr = cv2.imread(found_rgb)
                disp = cv2.imread(found_disp, cv2.IMREAD_GRAYSCALE)
                
                if disp is not None and bgr is not None:
                    disp_float = disp.astype(np.float32)
                    valid_mask = disp_float > 5.0
                    
                    # Convert disparity to normalized metric depth (1.0m to 4.5m)
                    depth = np.zeros_like(disp_float)
                    if np.any(valid_mask):
                        max_d = np.percentile(disp_float[valid_mask], 99.0)
                        min_d = np.percentile(disp_float[valid_mask], 1.0)
                        depth[valid_mask] = 1.0 + 3.0 * (max_d - disp_float[valid_mask]) / max(max_d - min_d, 1e-4)
                    else:
                        depth = 2.5 * np.ones_like(disp_float)
                        
                    # Save into data/samples/
                    out_rgb_path = os.path.join(SAMPLES_DIR, f"real_middlebury_{scene_name}_rgb.png")
                    out_depth_npy = os.path.join(SAMPLES_DIR, f"real_middlebury_{scene_name}_depth.npy")
                    out_depth_png = os.path.join(SAMPLES_DIR, f"real_middlebury_{scene_name}_depth.png")
                    
                    cv2.imwrite(out_rgb_path, bgr)
                    np.save(out_depth_npy, depth.astype(np.float32))
                    depth_mm = (depth * 1000.0).astype(np.uint16)
                    cv2.imwrite(out_depth_png, depth_mm)
                    
                    print(f" [SUCCESS] Extracted real scene '{scene_name}' ({bgr.shape[1]}x{bgr.shape[0]})")
                    count += 1
            else:
                print(f" [WARNING] Matched RGB/Depth not found in {extract_folder}")
                
        except Exception as e:
            print(f" [ERROR] Could not fetch {scene_name}: {e}")

    print(f"\nCompleted! {count} real Middlebury dataset scenes successfully added to data/samples/.")

if __name__ == "__main__":
    download_and_extract_middlebury()
