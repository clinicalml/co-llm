BASE_PORT=${1:-8302}
MODEL_NAME=$2
TOKENIZER_NAME=$3
NUM_GPUS=${4:-2}
DATASET=$5
SAVE_FOLDER=$6
MODEL_NAME_SAVE=$7
STOP_SEQUENCE=$8

# if two gpus, set CUDA_VISIBLE_DEVICES=0,1, else set CUDA_VISIBLE_DEVICES=0
if [ $NUM_GPUS -eq 2 ]; then
    CUDA_DEVICES=0,1
else
    CUDA_DEVICES=0
fi

echo "CUDA_VISIBLE_DEVICES:   $CUDA_DEVICES"
echo "BASE_PORT:              $BASE_PORT"
echo "MODEL_NAME:             $MODEL_NAME"
echo "TOKENIZER_NAME:         $TOKENIZER_NAME"
echo "NUM_GPUS:               $NUM_GPUS"
echo "DATASET:                $DATASET"
echo "SAVE_FOLDER:            $SAVE_FOLDER"
echo "MODEL_NAME_SAVE:        $MODEL_NAME_SAVE"
echo "STOP_SEQUENCE:          $STOP_SEQUENCE"

# Starting the assistant server with Llama-70b. 
# It takes 2 A100 80GB GPUs. # If you are using A6000 with 48GB memory, you shall use 4 GPUs.
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    python collm/inference/api_server_simple.py --port $BASE_PORT \
    --tensor-parallel-size $NUM_GPUS \
    --model $MODEL_NAME \
    --tokenizer $TOKENIZER_NAME \
    &> base_server.log &

pid_base_server=$!
echo "Base Server PID: $pid_base_server"

python collm/generate.py generate_default \
        --dataset "static_eval-completion/$DATASET/test" \
        --save_path "checkpoints/$SAVE_FOLDER/$DATASET/$MODEL_NAME_SAVE" \
        --model_port $BASE_PORT \
        --num_proc 8 \
        --max_tokens 512 \
        --tokenizer_name_base $TOKENIZER_NAME \
        --stop_sequences $STOP_SEQUENCE

kill $pid_base_server