"""
Download fine-tuned model weights from HuggingFace Hub.
Repository: Shafi99/iPhoneBlur_Finetune_Models

Usage:
    from models.download_weights import download_weights
    weights_path = download_weights('nafnet')
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "Shafi99/iPhoneBlur_Finetune_Models"

WEIGHT_FILES = {
    'nafnet': 'nafnet_final_weights.pth',
    'restormer': 'restormer_final_weights.pth',
    'hinet': 'hinet_final_weights.pth',
    'mimo_unet': 'mimo_final_weights.pth',
    'instructir': 'instructir_final_weights.pth'
}

def download_weights(model_name, force_download=False):
    """
    Download model weights from HuggingFace Hub.
    
    Args:
        model_name: One of ['nafnet', 'restormer', 'hinet', 'mimo_unet', 'instructir']
        force_download: Re-download even if cached
        
    Returns:
        Path to downloaded .pth file
        
    Raises:
        ValueError: If model_name not recognized
        Exception: If download fails
    """
    if model_name not in WEIGHT_FILES:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(WEIGHT_FILES.keys())}")
    
    filename = WEIGHT_FILES[model_name]
    
    print(f"📥 Downloading {model_name} weights from HuggingFace...")
    print(f"   Repository: {REPO_ID}")
    print(f"   File: {filename}")
    
    try:
        weights_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            force_download=force_download
        )
        
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"✅ Downloaded successfully! ({size_mb:.1f} MB)")
        print(f"   Cached at: {weights_path}\n")
        
        return weights_path
        
    except Exception as e:
        print(f"❌ ERROR: Failed to download {model_name} weights")
        print(f"   {str(e)}\n")
        print(f"💡 Troubleshooting:")
        print(f"   1. Check internet connection")
        print(f"   2. Verify HuggingFace Hub is accessible")
        print(f"   3. Try: pip install --upgrade huggingface_hub")
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download model weights')
    parser.add_argument('--model', required=True, choices=list(WEIGHT_FILES.keys()),
                       help='Model name to download')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if cached')
    args = parser.parse_args()
    
    download_weights(args.model, args.force)
