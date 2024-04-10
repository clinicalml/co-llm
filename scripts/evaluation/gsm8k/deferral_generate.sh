TASK_NAME="gsm8k"
DATASET="gsm-0shot"
SAVE_FOLDER="generate/gsm8k-finetuned"
MAX_TOKENS=512

MODEL_NAME_SAVE="gsm8k-def-finetuned-Llama-2-7b+Llama-2-70b"
ASST_MODEL_PATH="./weights/Llama-2-70b-hf"
ASST_TOKENIZER_PATH="./weights/Llama-2-70b-hf"
ASST_NUM_GPUS=2
BASE_MODEL_PATH="./checkpoints/deferral_finetuned/gsm8k-completion_Llama-2-70b-hf/Llama-2-7b-hf~128bz"
BASE_TOKENIZER_PATH="./checkpoints/deferral_finetuned/gsm8k-completion_Llama-2-70b-hf/Llama-2-7b-hf~128bz"
BASE_NUM_GPUS=1

# mistral + mixtral 
bash scripts/evaluation/generic/deferral_generate.sh \
    $TASK_NAME \
    $ASST_MODEL_PATH \
    $ASST_TOKENIZER_PATH \
    $ASST_NUM_GPUS \
    $BASE_MODEL_PATH \
    $BASE_TOKENIZER_PATH \
    $BASE_NUM_GPUS \
    $DATASET \
    $SAVE_FOLDER \
    $MODEL_NAME_SAVE \
    $MAX_TOKENS \
    9233