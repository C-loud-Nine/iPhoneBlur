# ==============================================================================
# HOW TO RUN THIS FILE:
# Open your terminal in the repository root and run:
# python download_dataset.py
# 
# PREREQUISITE:
# You must have your kaggle.json token configured.
# Usually, this means placing it in ~/.kaggle/kaggle.json (Linux/Mac) 
# or C:\Users\<YourUser>\.kaggle\kaggle.json (Windows).
# ==============================================================================

import os
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download the iPhoneBlur Dataset from Kaggle")
    # Default is set to ./data/iphoneblur so training scripts find it instantly
    parser.add_argument('--download_dir', type=str, default='./data/iphoneblur', help="Where to save the dataset")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Preparing to download iPhoneBlur Dataset to {download_dir}...")
    print(f"⚠️  WARNING: This dataset is approximately 9.8 GB in size.")
    
    # ---------------------------------------------------------
    # ASK FOR PERMISSION BEFORE DOWNLOADING
    # ---------------------------------------------------------
    user_input = input("❓ Do you want to proceed with the download? (y/n): ").strip().lower()
    
    if user_input not in ['y', 'yes']:
        print("🛑 Download cancelled by user. Exiting...")
        sys.exit(0)
    
    print("⬇️ Starting download... Please wait.")
    
    # Run Kaggle API to download and unzip directly into the target folder
    exit_code = os.system(f"kaggle datasets download -d shafi09/iphoneblur -p {download_dir} --unzip")
    
    if exit_code != 0:
        print("❌ ERROR: Failed to download. Make sure your kaggle API token (kaggle.json) is properly configured.")
        return

    # Verify the download by checking for the main metadata file
    metadata_path = download_dir / "metadata" / "complete_metadata.csv"
    if metadata_path.exists():
        print(f"✅ Dataset successfully downloaded and verified at {download_dir}!")
        print(f"📁 You are now ready to run the training scripts!")
    else:
        print("⚠️ Download finished, but complete_metadata.csv not found.")
        print("Please check if the Kaggle dataset was unzipped into a nested subfolder by mistake.")

if __name__ == '__main__':
    main()