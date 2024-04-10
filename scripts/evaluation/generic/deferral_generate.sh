ASST_PORT=${12:-9600}
BASE_PORT=$((ASST_PORT + 1))

TASK_NAME=$1
ASST_MODEL_PATH=$2
ASST_TOKENIZER_PATH=$3
ASST_NUM_GPUS=${4:-2}
BASE_MODEL_PATH=$5
BASE_TOKENIZER_PATH=$6
BASE_NUM_GPUS=${7:-1}
DATASET=$8
SAVE_FOLDER=$9
MODEL_NAME_SAVE=${10}
MAX_TOKENS=${11}

if [ -f "$BASE_MODEL_PATH/gperp_init.bin" ]; then
    mv "$BASE_MODEL_PATH/gperp_init.bin" "$BASE_MODEL_PATH/gperp_init.bin.bak"
fi

echo "ASST_PORT:            $ASST_PORT"
echo "BASE_PORT:            $BASE_PORT"
echo "TASK_NAME:            $TASK_NAME"
echo "ASST_MODEL_PATH:      $ASST_MODEL_PATH"
echo "ASST_TOKENIZER_PATH:  $ASST_TOKENIZER_PATH"
echo "ASST_NUM_GPUS:        $ASST_NUM_GPUS"
echo "BASE_MODEL_PATH:      $BASE_MODEL_PATH"
echo "BASE_TOKENIZER_PATH:  $BASE_TOKENIZER_PATH"
echo "BASE_NUM_GPUS:        $BASE_NUM_GPUS"
echo "DATASET:              $DATASET"
echo "SAVE_FOLDER:          $SAVE_FOLDER"
echo "MODEL_NAME_SAVE:      $MODEL_NAME_SAVE"
echo "MAX_TOKENS:           $MAX_TOKENS"

# Starting the assistant server
CUDA_VISIBLE_DEVICES=0,1 \
    python collm/inference/api_server_simple.py --port $ASST_PORT \
    --tensor-parallel-size $ASST_NUM_GPUS \
    --model $ASST_MODEL_PATH \
    --tokenizer $ASST_TOKENIZER_PATH \
    &> asst_server.log &
pid_asst_server=$!
echo "Assistant Server PID: $pid_asst_server"

# Starting the base server
CUDA_VISIBLE_DEVICES=2 \
    python collm/inference/api_server_deferral.py --port $BASE_PORT \
    --tensor-parallel-size $BASE_NUM_GPUS \
    --model $BASE_MODEL_PATH \
    --tokenizer $BASE_TOKENIZER_PATH \
    &> base_server.log &
pid_base_server=$!
echo "Base Server PID: $pid_base_server"

python collm/generate.py deferral_threshold_search \
        --dataset "static_eval-completion/$DATASET/validation" \
        --save_path "checkpoints/$SAVE_FOLDER/$DATASET/$MODEL_NAME_SAVE/" \
        --num_proc 8 \
        --max_tokens $MAX_TOKENS \
        --base_model_port $BASE_PORT \
        --ref_model_port $ASST_PORT \
        --batch_gen_port 8003 \
        --n_samples_for_search 15 \
        --tokens_per_call 1 \
        --tokenizer_name_ref $ASST_TOKENIZER_PATH \
        --tokenizer_name_base $BASE_TOKENIZER_PATH \
        --debug

python collm/eval.py eval_folder \
        --task_name $TASK_NAME \
        --orig_data_path static_eval-completion/$DATASET/validation \
        --pred_data_folder checkpoints/$SAVE_FOLDER/$DATASET/$MODEL_NAME_SAVE/_deferral_search 

python collm/generate.py generate_deferral \
        --dataset "static_eval-completion/$DATASET/test" \
        --save_path "checkpoints/$SAVE_FOLDER/$DATASET/$MODEL_NAME_SAVE" \
        --ref_model_port $ASST_PORT \
        --base_model_port $BASE_PORT \
        --num_proc 8 \
        --max_tokens $MAX_TOKENS \
        --tokens_per_call 1 \
        --tokenizer_name_base $BASE_TOKENIZER_PATH \
        --tokenizer_name_ref $ASST_TOKENIZER_PATH

kill $pid_asst_server
kill $pid_base_server