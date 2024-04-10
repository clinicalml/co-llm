# All eval code

# Default Llama 
BASE_PORT=8302
MODEL_NAME="weights/Llama-2-7b-hf"
TOKENIZER_NAME="weights/Llama-2-7b-hf"
NUM_GPUS=2
DATASET="gsm-0shot"
SAVE_FOLDER="generate/base-model"
MODEL_NAME_SAVE="Llama-2-7b-hf"
STOP_SEQUENCE="['\n\n']" 
bash scripts/evaluation/generic/default_generate.sh \
    $BASE_PORT \
    $MODEL_NAME \
    $TOKENIZER_NAME \
    $NUM_GPUS \
    $SAVE_FOLDER \
    $DATASET \
    $MODEL_NAME_SAVE \
    $STOP_SEQUENCE \

# mistral ft 
# BASE_PORT=8302
# MODEL_NAME="checkpoints/default_finetuned/gsm8k-completion_Mistral-7B-v0.1/Mistral-7B-v0.1~128bz~2e-5lr"
# TOKENIZER_NAME="checkpoints/default_finetuned/gsm8k-completion_Mistral-7B-v0.1/Mistral-7B-v0.1~128bz~2e-5lr"
# NUM_GPUS=1
# DATASET="gsm-0shot"
# SAVE_FOLDER="generate/gsm8k-finetuned"
# MODEL_NAME_SAVE="gsm8k-finetuned-mistral-7b-v0.1~2e-5lr"
# STOP_SEQUENCE="['\n\n']" 

# bash scripts/evaluation/generic/default_generate.sh \
#     $BASE_PORT \
#     $MODEL_NAME \
#     $TOKENIZER_NAME \
#     $NUM_GPUS \
#     $SAVE_FOLDER \
#     $DATASET \
#     $MODEL_NAME_SAVE \
#     $STOP_SEQUENCE \

# mixtral  
BASE_PORT=8302
MODEL_NAME="weights/Mixtral-8x7B-v0.1"
TOKENIZER_NAME="weights/Mixtral-8x7B-v0.1"
NUM_GPUS=2
DATASET="gsm-0shot"
SAVE_FOLDER="generate/base-model"
MODEL_NAME_SAVE="Mixtral-8x7B-v0.1"
STOP_SEQUENCE="['\n\n']" 
bash scripts/evaluation/generic/default_generate.sh \
    $BASE_PORT \
    $MODEL_NAME \
    $TOKENIZER_NAME \
    $NUM_GPUS \
    $SAVE_FOLDER \
    $DATASET \
    $MODEL_NAME_SAVE \
    $STOP_SEQUENCE \

DATASET="gsm-8shot"
bash scripts/evaluation/generic/default_generate.sh \
    $BASE_PORT \
    $MODEL_NAME \
    $TOKENIZER_NAME \
    $NUM_GPUS \
    $SAVE_FOLDER \
    $DATASET \
    $MODEL_NAME_SAVE \
    $STOP_SEQUENCE \
