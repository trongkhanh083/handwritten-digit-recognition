#!/usr/bin/env bash
# scripts/run_training.sh
set -e
export PYTHONPATH="$(pwd)"   # so `python -m src.train` can import your src/ package

# Load defaults from configs/default.yaml
CKPT_PATH=$(grep '^ckpt_path:' configs/default.yaml| awk '{print $2}')
EPOCHS=$(grep '^epochs:' configs/default.yaml      | awk '{print $2}')
BATCH_SIZE=$(grep '^batch_size:' configs/default.yaml | awk '{print $2}')
NUM_IMG=$(grep '^num_img:' configs/default.yaml | awk '{print $2}')

echo "➜ Training with:"
echo "   ckpt_path=$CKPT_PATH"
echo "   epochs=$EPOCHS, batch_size=$BATCH_SIZE"
echo

# 1) Train (also does prepare_data → compile & fit)
python -m src.train \
  --ckpt_path  "$CKPT_PATH" \
  --epochs     "$EPOCHS" \
  --batch_size "$BATCH_SIZE"

echo && echo "➜ Evaluation:"
# 2) Evaluate on test set (reads the same ckpt_path)
python -m src.evaluate \
  --ckpt_path "$CKPT_PATH"

echo && echo "➜ Prediction:"
# 3) Predict on a handful of samples
python -m src.predict \
  --ckpt_path  "$CKPT_PATH" \
  --num_img "$NUM_IMG"

echo
echo "All done! You finished the Handwritten Digit Recognition Project. Enjoy your result!"