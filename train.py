"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    # ─── New optimizations: time features / user-dense grouping / DIN ───
    parser.add_argument('--use_time_feats', action='store_true', default=False,
                        help='Inject 9-column UTC+8 calendar embeddings into '
                             'user_ns as a residual (does not change NS '
                             'token count).')
    parser.add_argument('--time_feats_dropout', type=float, default=0.1,
                        help='Dropout for the calendar-time projection MLP '
                             '(only effective when --use_time_feats is on).')
    parser.add_argument('--user_dense_grouped', action='store_true', default=False,
                        help='Split user_dense into typed groups (SUM emb 256 '
                             '+ LMF4Ads emb 320 + plain dense) and project '
                             'each independently before fusing into one NS '
                             'token. Falls back to a single Linear+LN when '
                             'user_dense_dim is too small.')
    parser.add_argument('--use_din', action='store_true', default=False,
                        help='Enable target-aware DIN activation that adds a '
                             'candidate-conditioned residual on top of the '
                             'backbone output.')
    parser.add_argument('--din_top_k', type=int, default=32,
                        help='Top-k positions kept by the DIN attention per '
                             'sequence domain (only effective when --use_din).')
    parser.add_argument('--din_hidden_mult', type=int, default=2,
                        help='Hidden-dim multiplier for the DIN gate / delta '
                             'MLPs (only effective when --use_din).')

    # ─── Auxiliary contrastive losses (InfoNCE + SupCon) ───────────────────
    parser.add_argument('--use_aux_loss', action='store_true', default=False,
                        help='Enable candidate-logit InfoNCE as an auxiliary loss. '
                             'It expands the main model scoring path to per-row '
                             'candidate items, with column 0 as the observed pair. '
                             'Required to also enable --use_supcon.')
    parser.add_argument('--aux_proj_dim', type=int, default=64,
                        help='Output dim of the auxiliary projection heads.')
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                        help='Maximum weight on the candidate-logit InfoNCE auxiliary loss.')
    parser.add_argument('--aux_temperature', type=float, default=0.6,
                        help='Softmax temperature for candidate-logit InfoNCE.')
    parser.add_argument('--aux_warmup_steps', type=int, default=0,
                        help='Reserved for compatibility; candidate-logit InfoNCE does not use warmup.')
    parser.add_argument('--aux_pair_chunk_size', type=int, default=4096,
                        help='Compatibility option retained for older configs; lightweight candidate-rank InfoNCE does not replay the backbone in chunks.')
    parser.add_argument('--aux_candidate_count', type=int, default=64,
                        help='Number of per-row candidate items for auxiliary InfoNCE, including the observed item in column 0.')
    parser.add_argument('--aux_positive_weight', type=float, default=1.0,
                        help='Sample weight for label==1 rows inside candidate-rank InfoNCE. 1.0 disables positive reweighting.')
    parser.add_argument('--aux_history_weight', type=float, default=0.0,
                        help='Extra weight of same-user history hard-negative InfoNCE added inside the auxiliary loss. 0 disables it.')
    parser.add_argument('--aux_history_max_per_sample', type=int, default=64,
                        help='Maximum number of same-user sequence tokens sampled as history hard negatives per row.')
    parser.add_argument('--aux_history_pos_weight', type=float, default=2.0,
                        help='Sample weight for label==1 rows inside the history hard-negative InfoNCE branch.')
    parser.add_argument('--aux_history_domain', type=str, default='all',
                        help='Comma-separated sequence domains used by history hard negatives, or all.')
    parser.add_argument('--use_supcon', action='store_true', default=False,
                        help='Additionally apply Supervised Contrastive '
                             'loss on the sample-level representation. '
                             'Requires --use_aux_loss.')
    parser.add_argument('--supcon_weight', type=float, default=0.05,
                        help='Weight on the SupCon auxiliary loss.')
    parser.add_argument('--supcon_temperature', type=float, default=0.1,
                        help='Softmax temperature for SupCon.')

    # ─── EMA (exponential moving average) of dense parameters ──────────────
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Maintain a Polyak EMA of all dense (non-'
                             'embedding) parameters and evaluate / save '
                             'checkpoints with the EMA weights. Embedding '
                             'tables are excluded.')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay (shadow = decay*shadow + '
                             '(1-decay)*raw). Default 0.999.')
    parser.add_argument('--ema_start_epoch', type=int, default=2,
                        help='Epoch (1-indexed) at which the EMA shadow '
                             'starts updating. The first epoch is excluded '
                             'so its high-variance gradients do not '
                             'contaminate the average.')

    # ─── Historical Behavior Importance Weighting ──────────────────────────
    parser.add_argument('--use_importance_weighting', action='store_true', default=False,
                        help='Enable historical behavior importance weighting '
                             'that assigns different weights to user behaviors '
                             'based on their relevance to the target item.')
    parser.add_argument('--importance_weighting_type', type=str, default='cross_attention',
                        choices=['cross_attention', 'bilinear', 'mlp'],
                        help='Type of importance weighting mechanism: '
                             'cross_attention = cross-attention between target and behaviors, '
                             'bilinear = bilinear interaction, '
                             'mlp = MLP-based scoring')
    parser.add_argument('--importance_dropout', type=float, default=0.1,
                        help='Dropout rate for importance weighting layers '
                             '(only effective when --use_importance_weighting is on)')

    # ─── 方案 A: Target-aware Cross-Attention Bias ──────────────────
    # 与 --use_importance_weighting 互斥: 后者是原地改写 seq (与 DIN 信息
    # 冲突且会在 bf16 amp 下 NaN); 本项是把 target prior 加性注入 cross-attn,
    # 与 DIN / cross-attn 互补且初值 alpha=0 安全退化.
    parser.add_argument('--use_target_attn_bias', action='store_true', default=False,
                        help='Inject target-aware additive bias into every '
                             'HyFormer cross-attention layer (Plan A). Replaces '
                             'the destructive sequence reweighting from '
                             '--use_importance_weighting; mutually exclusive in '
                             'practice. alpha is initialized to 0 so the model '
                             'starts as if disabled.')
    parser.add_argument('--target_attn_bias_dropout', type=float, default=0.0,
                        help='Dropout on the target-attn bias scores '
                             '(only effective when --use_target_attn_bias is on).')

    # ─── Acceleration knobs ──────────────────────────────────────────────
    parser.add_argument('--use_flash_attn_varlen', action='store_true', default=False,
                        help='Use Tri Dao flash-attn varlen kernel for '
                             'padding-masked self-attention (requires CUDA + '
                             'fp16/bf16 + `pip install flash-attn`). Auto '
                             'falls back to SDPA when unavailable.')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable mixed-precision training via '
                             'torch.autocast. BF16 skips GradScaler.')
    parser.add_argument('--amp_dtype', type=str, default='bf16',
                        choices=['bf16', 'fp16'],
                        help='Autocast dtype when --amp is on')
    parser.add_argument('--fused_adamw', action='store_true', default=False,
                        help='Use the CUDA fused AdamW implementation. '
                             'Auto-disabled on CPU.')
    parser.add_argument('--use_cosine_lr', action='store_true', default=False,
                        help='Apply CosineAnnealingLR on the dense optimizer '
                             '(per-step). T_max is set to total training steps.')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Eta-min for CosineAnnealingLR when '
                             '--use_cosine_lr is on')
    parser.add_argument('--torch_compile', action='store_true', default=False,
                        help='Wrap the model with torch.compile() for '
                             'kernel fusion. Ignored on CPU-only setups.')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode. `max-autotune` costs a few '
                             'minutes of warmup but gives the best steady-state '
                             'throughput.')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Output train loss / gradient statistics every N steps.')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_flash_attn_varlen": args.use_flash_attn_varlen,
        # New optimizations.
        "use_time_feats": args.use_time_feats,
        "time_feats_dropout": args.time_feats_dropout,
        "user_dense_grouped": args.user_dense_grouped,
        "use_din": args.use_din,
        "din_top_k": args.din_top_k,
        "din_hidden_mult": args.din_hidden_mult,
        # Auxiliary contrastive heads.
        "use_aux_loss": args.use_aux_loss,
        "aux_proj_dim": args.aux_proj_dim,
        # Historical behavior importance weighting
        "use_importance_weighting": args.use_importance_weighting,
        "importance_weighting_type": args.importance_weighting_type,
        "importance_dropout": args.importance_dropout,
        # 方案 A: Target-aware Cross-Attention Bias
        "use_target_attn_bias": args.use_target_attn_bias,
        "target_attn_bias_dropout": args.target_attn_bias_dropout,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # ── torch.compile (optional) ──
    # Wrapped AFTER .to(device) so that the graph tracer sees the final
    # device. We keep fullgraph=False because the model has control flow
    # (e.g. LongerEncoder L>top_k branch) that dynamo should graph-break on.
    if args.torch_compile:
        if args.device == 'cpu':
            logging.info("torch_compile requested but device=cpu; compiling anyway "
                         "(gains are modest on CPU, mainly graph-break diagnostics)")
        logging.info(f"Wrapping model with torch.compile(mode={args.compile_mode})")
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        fused_adamw=args.fused_adamw,
        use_cosine_lr=args.use_cosine_lr,
        min_lr=args.min_lr,
        log_every=args.log_every,
        # Auxiliary contrastive losses.
        use_aux_loss=args.use_aux_loss,
        aux_temperature=args.aux_temperature,
        aux_loss_weight=args.aux_loss_weight,
        aux_warmup_steps=args.aux_warmup_steps,
        aux_pair_chunk_size=args.aux_pair_chunk_size,
        aux_candidate_count=args.aux_candidate_count,
        aux_positive_weight=args.aux_positive_weight,
        aux_history_weight=args.aux_history_weight,
        aux_history_max_per_sample=args.aux_history_max_per_sample,
        aux_history_pos_weight=args.aux_history_pos_weight,
        aux_history_domain=args.aux_history_domain,
        use_supcon=args.use_supcon,
        supcon_temperature=args.supcon_temperature,
        supcon_weight=args.supcon_weight,
        # EMA on dense parameters.
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_start_epoch=args.ema_start_epoch,
        # Historical behavior importance weighting
        use_importance_weighting=args.use_importance_weighting,
        importance_weighting_type=args.importance_weighting_type,
        importance_dropout=args.importance_dropout,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
