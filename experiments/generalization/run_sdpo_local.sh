#!/bin/bash

export WANDB_API_KEY="wandb_api_key"
export WANDB_ENTITY="Memory-based-Optimizer"
export BASE_WANDB_PROJECT="sdpo-full-run"
export WANDB_PROJECT="$BASE_WANDB_PROJECT"
export WANDB_MODE="online"

python -m wandb login --relogin "$WANDB_API_KEY"

sanitize_name() {
    local value="$1"

    value="${value%/}"
    value=$(echo "$value" | sed -E 's#[^A-Za-z0-9._-]+#-#g; s#[-]+#-#g; s#[.][.]+#.#g; s#^[-.]+##; s#[-.]+$##')
    if [[ -z "$value" ]]; then
        value="dataset"
    fi

    echo "$value"
}

CONFIG_NAME="sdpo"

DATA_PATHS=(
    "datasets/sciknoweval/biology/"
)

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
HF_PUSH_ENABLED=true
SEED=42
LR=1e-5
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MODEL_PATH="Qwen/Qwen3-1.7B"
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')

for DATA_PATH in "${DATA_PATHS[@]}"; do
    DATA_SLUG=$(sanitize_name "$DATA_PATH")
    export WANDB_PROJECT="${BASE_WANDB_PROJECT}-${DATA_SLUG}"

    EXP_NAME="FINAL-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}"
    export WANDB_RUN="$EXP_NAME"

    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.seed=$SEED \
trainer.project_name=$WANDB_PROJECT \
trainer.group_name=SDPO-generalization \
trainer.n_gpus_per_node=1 \
trainer.hf_push_enabled=$HF_PUSH_ENABLED \
actor_rollout_ref.actor.data_loader_seed=$SEED \
actor_rollout_ref.actor.fsdp_config.seed=$SEED \
actor_rollout_ref.ref.fsdp_config.seed=$SEED \
critic.data_loader_seed=$SEED \
critic.model.fsdp_config.seed=$SEED \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.rollout.logprobs_mode=null \
custom_reward_function.path=$(pwd)/verl/utils/reward_score/feedback/__init__.py"

    echo "=============================="
    echo "Running: $EXP_NAME"
    echo "Data:    $DATA_PATH"
    echo "Seed:    $SEED"
    echo "W&B:     $WANDB_PROJECT"
    echo "=============================="

    bash training/verl_training.sh "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS "$@"
done
