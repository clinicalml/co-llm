if [ ! -f "data/processed/${DATASET_FOLDER}/${DATASET_NAME}_data.jsonl" ]; then
    python scripts/train/gsm8k/create_train_data.py
fi

LLAMA_BASE_PATH="weights"
DATASET_NAME="gsm8k"
DATASET_FOLDER="gsm8k"
SEQ_LENGTH=1024

# Only 1 GPU is needed
# export CUDA_VISIBLE_DEVICES=0 

MODEL_NAME="Llama-2-7b-hf"
echo "Process Dataset: Completion Format"
echo "Base model:     ${MODEL_NAME}"
echo "Dataset:        ${DATASET_NAME}"
python collm/dataset/scoring.py \
    --model_name_or_path ${LLAMA_BASE_PATH}/${MODEL_NAME} \
    --use_flash_attn \
    --use_completion_format \
    --tokenizer_name ${LLAMA_BASE_PATH}/${MODEL_NAME} \
    --use_slow_tokenizer \
    --train_file data/processed/${DATASET_FOLDER}/${DATASET_NAME}_data.jsonl\
    --max_seq_length $SEQ_LENGTH \
    --preprocessing_num_workers 16 \
    --output_dir checkpoints/dataset/${DATASET_NAME}-completion/${MODEL_NAME} 


MODEL_NAME="EleutherAI@llemma_7b"
echo "Process Dataset: Completion Format"
echo "Base model:     ${MODEL_NAME}"
echo "Dataset:        ${DATASET_NAME}"
python collm/dataset/scoring.py \
    --model_name_or_path ${LLAMA_BASE_PATH}/${MODEL_NAME} \
    --use_flash_attn \
    --use_completion_format \
    --tokenizer_name ${LLAMA_BASE_PATH}/${MODEL_NAME} \
    --use_slow_tokenizer \
    --train_file data/processed/${DATASET_FOLDER}/${DATASET_NAME}_data.jsonl\
    --max_seq_length $SEQ_LENGTH \
    --preprocessing_num_workers 16 \
    --output_dir checkpoints/dataset/${DATASET_NAME}-completion/${MODEL_NAME} 


# You might need to have multiple GPUs to load the 70b model. 
# export CUDA_VISIBLE_DEVICES=0,1

# MODEL_NAME="Llama-2-70b-hf"
# echo "Process Dataset: Completion Format"
# echo "Base model:     ${MODEL_NAME}"
# echo "Dataset:        ${DATASET_NAME}"
# python collm/dataset/scoring.py \
#     --model_name_or_path ${LLAMA_BASE_PATH}/${MODEL_NAME} \
#     --use_flash_attn \
#     --use_completion_format \
#     --tokenizer_name ${LLAMA_BASE_PATH}/${MODEL_NAME} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FOLDER}/${DATASET_NAME}_data.jsonl\
#     --max_seq_length $SEQ_LENGTH \
#     --preprocessing_num_workers 16 \
#     --output_dir checkpoints/dataset/${DATASET_NAME}-completion/${MODEL_NAME} 


# MODEL_NAME="EleutherAI@llemma_34b"
# echo "Process Dataset: Completion Format"
# echo "Base model:     ${MODEL_NAME}"
# echo "Dataset:        ${DATASET_NAME}"
# python collm/dataset/scoring.py \
#     --model_name_or_path ${LLAMA_BASE_PATH}/${MODEL_NAME} \
#     --use_flash_attn \
#     --use_completion_format \
#     --tokenizer_name ${LLAMA_BASE_PATH}/${MODEL_NAME} \
#     --use_slow_tokenizer \
#     --train_file data/processed/${DATASET_FOLDER}/${DATASET_NAME}_data.jsonl\
#     --max_seq_length $SEQ_LENGTH \
#     --preprocessing_num_workers 16 \
#     --output_dir checkpoints/dataset/${DATASET_NAME}-completion/${MODEL_NAME} 
