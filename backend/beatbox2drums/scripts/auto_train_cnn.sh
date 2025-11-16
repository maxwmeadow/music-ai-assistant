#!/bin/bash
# Automatically start CNN training once data preparation completes

DATA_DIR="/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_data"
OUTPUT_DIR="/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_model"
SCRIPT_DIR="/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums_package_onset_aware/scripts"

echo "Waiting for data preparation to complete..."
echo "Checking for: ${DATA_DIR}/train_windows.npy and ${DATA_DIR}/val_windows.npy"
echo ""

# Wait for both files to exist
while true; do
    if [ -f "${DATA_DIR}/train_windows.npy" ] && [ -f "${DATA_DIR}/val_windows.npy" ]; then
        echo "✓ Data preparation complete!"
        break
    fi
    echo "$(date '+%H:%M:%S') - Waiting for data files..."
    sleep 30
done

# Wait a few extra seconds to ensure files are fully written
sleep 5

# Load metadata
if [ -f "${DATA_DIR}/train_metadata.json" ]; then
    echo ""
    echo "=== Training Data Metadata ==="
    cat "${DATA_DIR}/train_metadata.json"
    echo ""
fi

if [ -f "${DATA_DIR}/val_metadata.json" ]; then
    echo "=== Validation Data Metadata ==="
    cat "${DATA_DIR}/val_metadata.json"
    echo ""
fi

# Start training
echo "=========================================="
echo "Starting CNN Onset Detector Training"
echo "=========================================="
echo ""

module purge
module load Conda
conda activate hum2melody

python "${SCRIPT_DIR}/train_cnn_onset_detector.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --patience 15

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""

# Show results
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo "=== Final Metrics ==="
    cat "${OUTPUT_DIR}/metrics.json"
    echo ""
fi

echo "Model saved to: ${OUTPUT_DIR}"
echo "Use: ${OUTPUT_DIR}/best_onset_model.h5 for inference"
