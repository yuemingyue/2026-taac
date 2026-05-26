#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Default paths (can be overridden by pre-exported env vars)
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${SCRIPT_DIR}/data}"
export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-${SCRIPT_DIR}/ckpt}"
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${SCRIPT_DIR}/log}"
export TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-${SCRIPT_DIR}/tf_events}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_flash_attn_varlen \
    --amp --amp_dtype bf16 \
    --fused_adamw \
    --use_cosine_lr --min_lr 1e-6 \
    --torch_compile --compile_mode default \
    --log_every 20 \
    --use_time_feats \
    --user_dense_grouped \
    --use_din \
    --use_aux_loss \
    --aux_loss_weight 0.1 \
    --aux_temperature 0.6 \
    --aux_warmup_steps 0 \
    --aux_candidate_count 64 \
    --aux_positive_weight 2.0 \
    --aux_history_weight 0.25 \
    --aux_history_max_per_sample 32 \
    --aux_history_pos_weight 2.0 \
    --aux_history_domain all \
    --aux_pair_chunk_size 4096 \
    --use_ema \
    --ema_decay 0.999 \
    --ema_start_epoch 2 \
    --use_target_attn_bias \
    --target_attn_bias_dropout 0.0 \
    "$@"