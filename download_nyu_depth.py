"""
Downloads and extracts the real NYU Depth V2 dataset.
Source: Ultralytics NYU-Depth-png mirror (421 MB clean PNG dataset of RGB and Depth maps).
"""

import os
import sys
import zipfile
import urllib.request
import glob
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NYU_DIR = os.path.join(DATA_DIR, "nyu_depth_v2")
ZIP_PATH = os.path.join(DATA_DIR, "nyu-depth-png.zip")
URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth-png.zip"

def download_nyu_depth():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NYU_DIR, exist_ok=True)
    
    print("=" * 70)
    print("DOWNLOADING OFFICIAL NYU DEPTH V2 DATASET (421 MB)")
    print(f"Destination: {NYU_DIR}")
    print("=" * 70)
    
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading from: {URL}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(URL, headers=headers)
        
        with urllib.request.urlopen(req) as response, open(ZIP_PATH, 'wb') as out_file:
            total_size = int(response.getheader('Content-Length', 0))
            downloaded = 0
            block_size = 512 * 1024  # 512 KB blocks
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\rProgress: [{percent:5.1f}%] {mb_down:.1f} MB / {mb_tot:.1f} MB")
                    sys.stdout.flush()
        print("\nDownload finished successfully!")
    else:
        print(f"Zip archive already present at {ZIP_PATH}. Skipping download.")
        
    print("\nExtracting archive...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(NYU_DIR)
        
    print(f"Extracted to {NYU_DIR}")
    
    # Check directory contents
    subdirs = [os.path.join(NYU_DIR, d) for d in os.listdir(NYU_DIR)]
    print("Top-level items in NYU Depth V2 folder:")
    for s in subdirs[:10]:
        print(" -", os.path.basename(s))
        
    all_pngs = glob.glob(os.path.join(NYU_DIR, "**", "*.png"), recursive=True)
    print(f"\nTotal PNG images/depth maps found: {len(all_pngs)}")
    
    # Optionally copy representative sample pairs into data/samples/ for instant app discovery
    samples_dir = os.path.join(DATA_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    print("\nNYU Depth V2 dataset successfully installed and ready!")

if __name__ == "__main__":
    download_nyu_depth()
