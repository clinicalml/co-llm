export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET_NAME="gsm8k-completion"
MODEL_NAME="Llama-2-70b-hf"

DATASET_BASE_PATH="checkpoints/dataset/${DATASET_NAME}/${MODEL_NAME}"
LLAMA_BASE_PATH="./weights"
BASE_SAVE_PATH="checkpoints/qlora_finetuned"

NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_NAME} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

export WANDB_PROJECT="qlora_finetune-${DATASET_NAME}-${MODEL_NAME}"

# Lora training
# Reference args in https://github.com/artidoro/qlora/blob/main/scripts/finetune_guanaco_65b.sh

accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --dataset_name ${DATASET_BASE_PATH} \
    --skip_data_preprocessing \
    --model_name_or_path ${LLAMA_BASE_PATH}/${MODEL_NAME} \
    --tokenizer_name ${LLAMA_BASE_PATH}/${MODEL_NAME} \
    --gradient_checkpointing \
    --use_qlora \
    --use_lora \
    --use_flash_attn \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --use_slow_tokenizer \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir ${BASE_SAVE_PATH}/${DATASET_NAME}/${MODEL_NAME}_lora/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 

python open_instruct/merge_lora.py \
    --base_model_name_or_path ${LLAMA_BASE_PATH}/${MODEL_NAME} \
    --lora_model_name_or_path ${BASE_SAVE_PATH}/${DATASET_NAME}/${MODEL_NAME}_lora/ \
    --output_dir ${BASE_SAVE_PATH}/${DATASET_NAME}/${MODEL_NAME}_merged/
