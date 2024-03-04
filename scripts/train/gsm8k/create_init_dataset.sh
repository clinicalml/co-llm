export PYTHONPATH="${PYTHONPATH}:$(pwd)"

BASE_CHECKPOINT_PATH=checkpoints/dataset/gsm8k-completion
TRAINING_MODEL=Llama-2-7b-hf
DEFERRAL_MODEL=EleutherAI@llemma_7b
INIT_METHOD="a"

python collm/dataset/initialization.py \
    --training_model_ds $BASE_CHECKPOINT_PATH/$TRAINING_MODEL \
    --deferral_model_ds $BASE_CHECKPOINT_PATH/$DEFERRAL_MODEL \
    --init_method $INIT_METHOD \
    --output_dir $BASE_CHECKPOINT_PATH/init-$INIT_METHOD-$TRAINING_MODEL+$DEFERRAL_MODEL \
    --log_level DEBUG 