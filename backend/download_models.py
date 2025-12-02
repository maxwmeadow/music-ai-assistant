"""Download model checkpoints from R2 on startup if not present locally."""

import os
import sys
from pathlib import Path
import urllib.request
import urllib.error

# R2 base URL (from CDN configuration)
R2_BASE_URL = os.getenv("MODEL_CDN_URL", "https://models.phonauto.com")

# Model files to download
MODEL_FILES = [
    {
        "url": f"{R2_BASE_URL}/hum2melody/checkpoints/combined_hum2melody_full.pth",
        "path": "hum2melody/checkpoints/combined_hum2melody_full.pth",
        "size_mb": 135
    },
    {
        "url": f"{R2_BASE_URL}/beatbox2drums/checkpoints/onset_detector/best_onset_model.h5",
        "path": "beatbox2drums/checkpoints/onset_detector/best_onset_model.h5",
        "size_mb": 3.7
    },
    {
        "url": f"{R2_BASE_URL}/beatbox2drums/checkpoints/drum_classifier/best_model_multi_input.pth",
        "path": "beatbox2drums/checkpoints/drum_classifier/best_model_multi_input.pth",
        "size_mb": 1.5
    },
    {
        "url": f"{R2_BASE_URL}/beatbox2drums/checkpoints/drum_classifier/feature_normalization.npz",
        "path": "beatbox2drums/checkpoints/drum_classifier/feature_normalization.npz",
        "size_mb": 0.001
    }
]


def download_file(url: str, dest_path: Path, size_mb: float):
    """Download a file with progress reporting."""
    print(f"Downloading {dest_path.name} ({size_mb:.1f} MB)...")

    try:
        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                if block_num % 100 == 0:  # Print every 100 blocks
                    print(f"  Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='\r')

        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print(f"\n✅ Downloaded {dest_path.name}")
        return True

    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error {e.code} downloading {dest_path.name}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"❌ URL Error downloading {dest_path.name}: {e.reason}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {dest_path.name}: {e}")
        return False


def check_and_download_models():
    """Always download models from R2 (Git LFS doesn't work on Railway)."""
    print("=" * 60)
    print("MODEL CHECKPOINT DOWNLOAD")
    print("=" * 60)

    # Determine backend directory
    # When running from /app with --app-dir /app, paths should be relative to /app/backend
    backend_dir = Path(__file__).parent
    print(f"Backend directory: {backend_dir}")
    print(f"Note: Git LFS doesn't work on Railway - always downloading from R2\n")

    # Delete any existing files (they're likely LFS pointers from git clone)
    for model in MODEL_FILES:
        model_path = backend_dir / model["path"]
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"🗑️  Deleting existing {model['path']} ({size_mb:.1f} MB - likely LFS pointer)")
            try:
                model_path.unlink()
            except Exception as e:
                print(f"⚠️ Failed to delete {model['path']}: {e}")

    # Download all files
    print(f"\n📥 Downloading {len(MODEL_FILES)} model files from R2...")
    total_mb = sum(m["size_mb"] for m in MODEL_FILES)
    print(f"Total download size: {total_mb:.1f} MB\n")

    success_count = 0
    for model in MODEL_FILES:
        model_path = backend_dir / model["path"]
        if download_file(model["url"], model_path, model["size_mb"]):
            success_count += 1
        else:
            print(f"⚠️ Failed to download {model['path']}, will use mock predictions")

    print("\n" + "=" * 60)
    print(f"Downloaded {success_count}/{len(MODEL_FILES)} files successfully")
    print("=" * 60)

    return success_count == len(MODEL_FILES)


if __name__ == "__main__":
    try:
        success = check_and_download_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)