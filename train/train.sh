export MODEL_ID="${MODEL_ID:-pretrained/realistic-vision-v51}"
export UNET_ID="${UNET_ID:-pretrained/ic-light-unet}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/harmolight}"
export DATASET_NAME="${DATASET_NAME:-data/train_syn_object.jsonl}"
export OBJECT_DATASET_NAME="${OBJECT_DATASET_NAME:-data/object_train.jsonl}"
export VAL_SYN_PORTRAIT="${VAL_SYN_PORTRAIT:-data/val_synthetic_portrait.jsonl}"
export VAL_OBJECT="${VAL_OBJECT:-data/val_object.jsonl}"
export VAL_REAL_PORTRAIT="${VAL_REAL_PORTRAIT:-data/val_real_portrait.jsonl}"

CUDA_VISIBLE_DEVICES=1 python train_harmolight.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_NAME \
  --object_dataset_name=$OBJECT_DATASET_NAME \
  --resolution=512 \
  --train_batch_size=108 --gradient_accumulation_steps=1 --gradient_checkpointing \
  --max_train_steps=30000 \
  --checkpointing_steps=2000 --checkpoints_total_limit=10 \
  --validation_epochs=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_syn_portrait_image_url=$VAL_SYN_PORTRAIT \
  --val_object_image_url=$VAL_OBJECT \
  --val_real_portrait_image_url=$VAL_REAL_PORTRAIT \
  --validation_prompt="" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint=latest \
  --dataloader_num_workers=4 \
  --report_to=tensorboard \
  --use_8bit_adam \
