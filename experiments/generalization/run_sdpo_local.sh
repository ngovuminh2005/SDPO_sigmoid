#!/bin/bash

export WANDB_API_KEY="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"
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
    "datasets/sciknoweval/chemistry/"
)

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
HF_PUSH_ENABLED=false
SEED=42
LR=1e-5
WEIGHT_DECAY=0.01
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MODEL_PATH="Qwen/Qwen3-1.7B"
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')

# Optimizer options:
#   AdamW
#   SingleDeviceMuonWithAuxAdam
#   SingleDeviceMemoryMuonWithAuxAdam
OPTIMIZER_NAME="SingleDeviceMemoryMuonWithAuxAdam"
ADAM_AUX_LR="$LR"
ADAM_AUX_EPS=1e-10
MUON_LR=1e-5
MUON_MOMENTUM=0.95
MUON_NESTEROV=True
MUON_NS_STEPS=5
MEMORY_NUM_CENTROIDS=16
MEMORY_CENTROID_DIM=1024
MEMORY_LAMBDA=0.02
MEMORY_INIT_SEED=$SEED
MEMORY_EMA_DECAY=0.99
MEMORY_EMA_EPS=1e-8
MEMORY_STEP_UPDATE=64
MEMORY_EMA_WS_STEPS=256

for DATA_PATH in "${DATA_PATHS[@]}"; do
    DATA_SLUG=$(sanitize_name "$DATA_PATH")
    export WANDB_PROJECT="${BASE_WANDB_PROJECT}-${DATA_SLUG}"

    EXP_NAME="FINAL-${OPTIMIZER_NAME}-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}"
    export WANDB_RUN="$EXP_NAME"

    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE data.seed=$SEED trainer.project_name=$WANDB_PROJECT trainer.group_name=SDPO-generalization trainer.n_gpus_per_node=1 trainer.hf_push_enabled=$HF_PUSH_ENABLED actor_rollout_ref.actor.data_loader_seed=$SEED actor_rollout_ref.actor.fsdp_config.seed=$SEED actor_rollout_ref.ref.fsdp_config.seed=$SEED critic.data_loader_seed=$SEED critic.model.fsdp_config.seed=$SEED actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.model.path=$MODEL_PATH actor_rollout_ref.actor.optim.optimizer=$OPTIMIZER_NAME actor_rollout_ref.actor.optim.lr=$LR actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY actor_rollout_ref.actor.optim.adam_aux_lr=$ADAM_AUX_LR actor_rollout_ref.actor.optim.adam_aux_eps=$ADAM_AUX_EPS actor_rollout_ref.actor.optim.muon_lr=$MUON_LR actor_rollout_ref.actor.optim.muon_momentum=$MUON_MOMENTUM actor_rollout_ref.actor.optim.muon_nesterov=$MUON_NESTEROV actor_rollout_ref.actor.optim.muon_ns_steps=$MUON_NS_STEPS actor_rollout_ref.actor.optim.memory_num_centroids=$MEMORY_NUM_CENTROIDS actor_rollout_ref.actor.optim.memory_centroid_dim=$MEMORY_CENTROID_DIM actor_rollout_ref.actor.optim.memory_lambda=$MEMORY_LAMBDA actor_rollout_ref.actor.optim.memory_init_seed=$MEMORY_INIT_SEED actor_rollout_ref.actor.optim.memory_ema_decay=$MEMORY_EMA_DECAY actor_rollout_ref.actor.optim.memory_ema_eps=$MEMORY_EMA_EPS actor_rollout_ref.actor.optim.memory_step_update=$MEMORY_STEP_UPDATE actor_rollout_ref.actor.optim.memory_ema_ws_steps=$MEMORY_EMA_WS_STEPS actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.self_distillation.distillation_topk=100 algorithm.rollout_correction.rollout_is=token actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} actor_rollout_ref.actor.self_distillation.alpha=$ALPHA actor_rollout_ref.actor.self_distillation.include_environment_feedback=False actor_rollout_ref.actor.optim.lr_warmup_steps=10 actor_rollout_ref.rollout.val_kwargs.n=16 actor_rollout_ref.rollout.logprobs_mode=null custom_reward_function.path=$(pwd)/verl/utils/reward_score/feedback/__init__.py"

    echo "=============================="
    echo "Running:   $EXP_NAME"
    echo "Data:      $DATA_PATH"
    echo "Seed:      $SEED"
    echo "Optimizer: $OPTIMIZER_NAME"
    echo "W&B:       $WANDB_PROJECT"
    echo "=============================="

    bash training/verl_training.sh "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS "$@"
done
