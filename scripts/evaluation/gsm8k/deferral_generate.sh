ASST_PORT=8303
BASE_PORT=8302

# Starting the assistant server with Llama-70b. 
# It takes 2 A100 80GB GPUs. # If you are using A6000 with 48GB memory, you shall use 4 GPUs.
CUDA_VISIBLE_DEVICES=0,1 \
    python collm/inference/api_server_simple.py --port $ASST_PORT \
    --tensor-parallel-size 2 \
    --model ./weights/Llama-2-70b-hf \
    --tokenizer ./weights/Llama-2-70b-hf\
    &> asst_server.log &
pid_asst_server=$!
echo "Assistant Server PID: $pid_asst_server"

# Starting the assistant server with trained Llama-7b. 
# It takes 1 A100 80 GPUs. 
CUDA_VISIBLE_DEVICES=2 \
    python collm/inference/api_server_deferral.py --port $BASE_PORT \
    --tensor-parallel-size 1 \
    --model checkpoints/deferral_finetuned/gsm8k-completion_Llama-2-70b-hf/Llama-2-7b-hf~128bz \
    --tokenizer checkpoints/deferral_finetuned/gsm8k-completion_Llama-2-70b-hf/Llama-2-7b-hf~128bz \
    &> base_server.log &
pid_base_server=$!
echo "Base Server PID: $pid_base_server"

SAVE_FOLDER="generate/gsm8k-finetuned"
MODEL_NAME="gsm8k-finetuned-Llama-2-7b+Llama-2-70b"
DATASET="gsm-0shot"
threshold=0.6

python collm/generate.py generate_deferral \
        --dataset "static_eval-completion/$DATASET/test" \
        --save_path "checkpoints/$SAVE_FOLDER/$DATASET/$MODEL_NAME~$threshold" \
        --ref_model_port $ASST_PORT \
        --base_model_port $BASE_PORT \
        --num_proc 8 \
        --max_tokens 2048 \
        --deferral_threshold $threshold \
        --tokens_per_call 1 \
        --tokenizer_name_base "./checkpoints/deferral_finetuned/gsm8k-completion_Llama-2-70b-hf/Llama-2-7b-hf~128bz" \
        --tokenizer_name_ref "./weights/Llama-2-70b-hf/" 

kill $pid_asst_server
kill $pid_base_server