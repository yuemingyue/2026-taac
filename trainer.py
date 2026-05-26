"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import (
    sigmoid_focal_loss,
    EarlyStopping,
    option_softmax_loss,
    listwise_rank_infonce_loss,
    supcon_loss,
    EMAModel,
)
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        # ─── Acceleration knobs ──────────────────────────────────
        amp: bool = False,
        amp_dtype: str = 'bf16',
        fused_adamw: bool = False,
        use_cosine_lr: bool = False,
        min_lr: float = 1e-6,
        log_every: int = 20,
        # ─── Auxiliary contrastive losses ───────────────────────
        use_aux_loss: bool = False,
        aux_temperature: float = 0.6,
        aux_loss_weight: float = 0.1,
        aux_warmup_steps: int = 0,
        aux_pair_chunk_size: int = 4096,
        aux_candidate_count: int = 64,
        aux_positive_weight: float = 1.0,
        aux_history_weight: float = 0.0,
        aux_history_max_per_sample: int = 64,
        aux_history_pos_weight: float = 2.0,
        aux_history_domain: str = 'all',
        use_supcon: bool = False,
        supcon_temperature: float = 0.1,
        supcon_weight: float = 0.05,
        # ─── EMA (Polyak averaging on dense params) ───
        # When ``use_ema=True``, a shadow copy of every dense (non-embedding,
        # non-frozen) parameter is maintained with the standard
        # ``s = decay * s + (1-decay) * p`` rule. EMA only starts updating
        # at ``ema_start_epoch`` so the first epoch's high-variance updates
        # don't pollute the average. Validation & checkpointing always run
        # on the EMA weights once they have started; training keeps using
        # the raw weights.
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_start_epoch: int = 2,
        # ─── Historical Behavior Importance Weighting ──────────────────────
        use_importance_weighting: bool = False,
        importance_weighting_type: str = 'cross_attention',
        importance_dropout: float = 0.1,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # ─── Mixed precision ──────────────────────────────────────────────
        self.device_type: str = 'cuda' if str(device).startswith('cuda') else 'cpu'
        self.amp: bool = bool(amp)
        _dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16}
        self.amp_dtype: torch.dtype = _dtype_map.get(amp_dtype, torch.bfloat16)
        # GradScaler only needed for fp16; bf16 has wider dynamic range.
        self.grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.amp and self.device_type == 'cuda' and self.amp_dtype is torch.float16:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        if self.amp:
            logging.info(f"AMP enabled: device_type={self.device_type}, "
                         f"dtype={amp_dtype}, grad_scaler={self.grad_scaler is not None}")

        # ─── Dual optimizer: Adagrad for Embeddings, AdamW for dense ──────
        # Fused AdamW is only available on CUDA.
        adamw_kwargs: Dict[str, Any] = dict(lr=lr, betas=(0.9, 0.98))
        if fused_adamw and self.device_type == 'cuda':
            adamw_kwargs['fused'] = True
            logging.info("Using fused AdamW (CUDA)")
        elif fused_adamw:
            logging.info("fused_adamw requested but device is CPU; using regular AdamW")

        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, **adamw_kwargs
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), **adamw_kwargs
            )

        # ─── Cosine LR scheduler (dense optimizer only, per-step) ─────────
        self.use_cosine_lr: bool = bool(use_cosine_lr)
        self.min_lr: float = float(min_lr)
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        if self.use_cosine_lr:
            try:
                steps_per_epoch = len(train_loader)
            except TypeError:
                steps_per_epoch = 0
            t_max = max(1, steps_per_epoch * num_epochs)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.dense_optimizer, T_max=t_max, eta_min=self.min_lr,
            )
            logging.info(f"CosineAnnealingLR enabled: T_max={t_max} steps, "
                         f"eta_min={self.min_lr}")

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.log_every: int = max(1, int(log_every))

        # ─── Auxiliary contrastive losses ─────────────────────────
        # InfoNCE encodes the backbone once, then a lightweight head scores
        # [B,K] sampled candidate items from cached auxiliary user/item
        # representations without adding any raw item-id embedding shortcut.
        # SupCon remains independently weighted on the sample view.
        model_aux_heads = bool(getattr(
            model,
            'use_aux_loss',
            getattr(getattr(model, '_orig_mod', None), 'use_aux_loss', False),
        ))
        self.use_aux_loss: bool = bool(use_aux_loss) and model_aux_heads
        self.aux_temperature: float = float(aux_temperature)
        self.aux_loss_weight: float = float(aux_loss_weight)
        self.aux_warmup_steps: int = max(0, int(aux_warmup_steps))
        self.aux_pair_chunk_size: int = max(1, int(aux_pair_chunk_size))
        self.aux_candidate_count: int = max(2, int(aux_candidate_count))
        self.aux_positive_weight: float = max(0.0, float(aux_positive_weight))
        self.aux_history_weight: float = max(0.0, float(aux_history_weight))
        self.aux_history_max_per_sample: int = max(0, int(aux_history_max_per_sample))
        self.aux_history_pos_weight: float = max(0.0, float(aux_history_pos_weight))
        self.aux_history_domain: str = str(aux_history_domain)
        self.use_supcon: bool = bool(use_supcon) and model_aux_heads
        self.supcon_temperature: float = float(supcon_temperature)
        self.supcon_weight: float = float(supcon_weight)
        self._aux_global_step: int = 0
        if (use_aux_loss or use_supcon) and not model_aux_heads:
            logging.warning(
                "auxiliary loss requested but projection heads are unavailable; "
                "auxiliary losses are DISABLED. Set --use_aux_loss on the train CLI "
                "to build the projection heads.")
        if self.use_aux_loss or self.use_supcon:
            logging.info(
                f"Auxiliary losses enabled: hybrid_candidate_history_nce={self.use_aux_loss} "
                f"(w={self.aux_loss_weight}, T={self.aux_temperature}, "
                f"candidates={self.aux_candidate_count}, "
                f"positive_weight={self.aux_positive_weight}, "
                f"history_weight={self.aux_history_weight}, "
                f"history_max_per_sample={self.aux_history_max_per_sample}, "
                f"history_pos_weight={self.aux_history_pos_weight}, "
                f"history_domain={self.aux_history_domain}, "
                f"warmup=disabled), "
                f"supcon={self.use_supcon} "
                f"(w={self.supcon_weight}, T={self.supcon_temperature})")

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")

        # ─── Historical Behavior Importance Weighting ──────────────────────
        self.use_importance_weighting: bool = bool(use_importance_weighting) and getattr(model, 'use_importance_weighting', False)
        self.importance_weighting_type: str = importance_weighting_type
        self.importance_dropout: float = float(importance_dropout)
        if use_importance_weighting and not getattr(model, 'use_importance_weighting', False):
            logging.warning(
                "use_importance_weighting requested but model.use_importance_weighting=False; "
                "importance weighting is DISABLED. Set --use_importance_weighting on the train CLI "
                "to actually build the importance weighting module.")
        if self.use_importance_weighting:
            logging.info(
                f"Historical behavior importance weighting enabled: "
                f"type={self.importance_weighting_type}, "
                f"dropout={self.importance_dropout}")

        # ─── EMA setup ────────────────────────────────────────────
        # We build the EMA tracker eagerly so the shadow tensors are
        # allocated on the right device once and survive AMP / cosine LR
        # toggles. Updates are gated by ``self._ema_started`` which flips
        # to True at the start of ``ema_start_epoch``.
        self.use_ema: bool = bool(use_ema)
        self.ema_decay: float = float(ema_decay)
        self.ema_start_epoch: int = max(1, int(ema_start_epoch))
        self.ema: Optional[EMAModel] = None
        self._ema_started: bool = False
        if self.use_ema:
            # Exclude every nn.Embedding.weight (the sparse half) so the
            # EMA only spans dense parameters. Falls back to an empty
            # exclusion when the model lacks ``get_sparse_params``.
            if hasattr(self.model, 'get_sparse_params'):
                excl = {p.data_ptr() for p in self.model.get_sparse_params()}
            else:
                excl = set()
            self.ema = EMAModel(
                named_params=list(self.model.named_parameters()),
                decay=self.ema_decay,
                exclude_param_ids=excl,
            )
            logging.info(
                f"EMA enabled: decay={self.ema_decay}, "
                f"start_epoch={self.ema_start_epoch}, "
                f"tracked_tensors={self.ema.num_tracked_tensors()}, "
                f"tracked_params={self.ema.num_tracked_params():,} "
                f"(embedding tables excluded)")

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        # self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            # Flip the EMA gate at the start of ``ema_start_epoch``.
            # All subsequent ``_train_step`` calls will then update the
            # shadow; before this point the shadow stays at its init copy
            # so we don't pollute the average with high-variance early
            # gradient steps.
            if (self.ema is not None and not self._ema_started
                    and epoch >= self.ema_start_epoch):
                self._ema_started = True
                logging.info(
                    f"[EMA] starting shadow updates at epoch {epoch} "
                    f"(decay={self.ema_decay})")
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            # Accumulate train statistics on-device to defer host sync.
            stats_device = self.device if self.device_type == 'cuda' else 'cpu'
            stat_keys = ('loss', 'loss_main', 'loss_infonce', 'loss_supcon', 'grad_norm')
            stats_running = {
                key: torch.zeros((), dtype=torch.float32, device=stats_device)
                for key in stat_keys
            }
            loss_n = 0           # number of steps accumulated since last flush
            loss_sum_epoch = 0.0  # CPU accumulator over the whole epoch

            for step, batch in train_pbar:
                train_stats = self._train_step(batch)
                total_step += 1
                for key in stat_keys:
                    stats_running[key] = stats_running[key] + train_stats[key].detach().float()
                loss_n += 1

                # Flush logging every `log_every` steps – this is the sole
                # host-device sync point on the hot path.
                if (total_step % self.log_every == 0) or (step + 1 == len(self.train_loader)):
                    avg_stats = {
                        key: (stats_running[key] / loss_n).item()
                        for key in stat_keys
                    }
                    avg_loss = avg_stats['loss']
                    loss_sum_epoch += avg_loss * loss_n
                    for value in stats_running.values():
                        value.zero_()
                    loss_n = 0

                    dense_lr = self.dense_optimizer.param_groups[0]['lr']
                    sparse_lr = (
                        self.sparse_optimizer.param_groups[0]['lr']
                        if self.sparse_optimizer is not None else 0.0
                    )
                    logging.info(
                        f"Train step {total_step} | epoch={epoch} "
                        f"batch={step + 1}/{len(self.train_loader)} | "
                        f"loss={avg_stats['loss']:.6f} "
                        f"main={avg_stats['loss_main']:.6f} "
                        f"infonce={avg_stats['loss_infonce']:.6f} "
                        f"supcon={avg_stats['loss_supcon']:.6f} "
                        f"grad_norm={avg_stats['grad_norm']:.6f} "
                        f"dense_lr={dense_lr:.6g} sparse_lr={sparse_lr:.6g}")
                    if self.writer:
                        self.writer.add_scalar('Loss/train', avg_stats['loss'], total_step)
                        self.writer.add_scalar('Loss/main', avg_stats['loss_main'], total_step)
                        self.writer.add_scalar('Loss/aux_infonce', avg_stats['loss_infonce'], total_step)
                        self.writer.add_scalar('Loss/aux_supcon', avg_stats['loss_supcon'], total_step)
                        self.writer.add_scalar('Grad/global_norm', avg_stats['grad_norm'], total_step)
                        self.writer.add_scalar('LR/dense', dense_lr, total_step)
                        if self.sparse_optimizer is not None:
                            self.writer.add_scalar('LR/sparse', sparse_lr, total_step)
                    train_pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "grad": f"{avg_stats['grad_norm']:.3f}",
                    })

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    # Swap in EMA weights (if started) so both the AUC
                    # estimate and the saved checkpoint reflect the EMA
                    # version. Restore raw weights afterwards so training
                    # keeps using the live trajectory.
                    ema_swapped = False
                    if self.ema is not None and self._ema_started:
                        self.ema.apply_shadow()
                        ema_swapped = True
                    try:
                        val_auc, val_logloss = self.evaluate(epoch=epoch)
                        self.model.train()

                        logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                        if self.writer:
                            self.writer.add_scalar('AUC/valid', val_auc, total_step)
                            self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                        # ``_handle_validation_result`` may save model.pt
                        # via EarlyStopping; checkpoint must be EMA so we
                        # call it BEFORE restoring raw weights.
                        self._handle_validation_result(total_step, val_auc, val_logloss)
                    finally:
                        if ema_swapped:
                            self.ema.restore()

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            # Drain any remaining accumulated loss into the epoch sum. In
            # normal runs this has already been flushed on the last batch.
            if loss_n > 0:
                avg_loss = (stats_running['loss'] / loss_n).item()
                loss_sum_epoch += avg_loss * loss_n
                for value in stats_running.values():
                    value.zero_()
            num_batches = max(1, len(self.train_loader))
            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum_epoch / num_batches}")

            # Epoch-level validation. Same swap-in / restore guard as the
            # step-level path above so checkpoint files always carry EMA
            # weights once the EMA has started.
            ema_swapped = False
            if self.ema is not None and self._ema_started:
                self.ema.apply_shadow()
                ema_swapped = True
            try:
                val_auc, val_logloss = self.evaluate(epoch=epoch)
                self.model.train()

                logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                if self.writer:
                    self.writer.add_scalar('AUC/valid', val_auc, total_step)
                    self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                self._handle_validation_result(total_step, val_auc, val_logloss)
            finally:
                if ema_swapped:
                    self.ema.restore()

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        # Calendar-time ids: dataset always emits them, but legacy batches
        # without the key fall back to a (B, 0) placeholder so the model can
        # still run with ``use_time_feats=False``.
        B0 = device_batch['user_int_feats'].shape[0]
        time_feats = device_batch.get(
            'time_feats',
            torch.zeros(B0, 0, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            time_feats=time_feats,
        )

    def _get_aux_model(self) -> Optional[nn.Module]:
        """Return the underlying model object that exposes auxiliary methods."""
        if hasattr(self.model, 'forward_with_aux') and hasattr(self.model, 'score_aux_candidates'):
            return self.model
        orig_model = getattr(self.model, '_orig_mod', None)
        if (orig_model is not None
                and hasattr(orig_model, 'forward_with_aux')
                and hasattr(orig_model, 'score_aux_candidates')):
            return orig_model
        return None

    def _sample_option_columns(
        self,
        row_ids: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Build per-row item columns with the positive item in column 0."""
        device = row_ids.device
        option_count = min(self.aux_candidate_count, batch_size)
        if option_count <= 1:
            return row_ids.view(-1, 1)

        random_cols = torch.randint(
            0,
            batch_size - 1,
            (row_ids.numel(), option_count - 1),
            device=device,
        )
        random_cols = random_cols + (random_cols >= row_ids.view(-1, 1)).long()
        return torch.cat([row_ids.view(-1, 1), random_cols], dim=1)

    def _gather_option_item_ids(
        self,
        item_ids: Optional[torch.Tensor],
        option_cols: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Gather raw item ids for the sampled ``[B,K]`` candidate columns."""
        if item_ids is None:
            return None
        ids = item_ids.view(-1).to(device=option_cols.device)
        if ids.numel() != option_cols.shape[0]:
            return None
        return ids.index_select(0, option_cols.reshape(-1)).view_as(option_cols)

    def _filter_history_aux(
        self,
        position_aux: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[list], Optional[list]]:
        """Return optional per-domain sequence tensors for history hard negatives."""
        if position_aux is None:
            return None, None
        seq_repr_list = position_aux.get('seq_repr_list')
        seq_mask_list = position_aux.get('seq_mask_list')
        if not seq_repr_list or not seq_mask_list:
            return None, None
        if self.aux_history_domain == 'all':
            return seq_repr_list, seq_mask_list

        keep = {d.strip() for d in self.aux_history_domain.split(',') if d.strip()}
        if not keep:
            return seq_repr_list, seq_mask_list
        domains = getattr(self.model, 'seq_domains', None)
        if domains is None:
            orig_model = getattr(self.model, '_orig_mod', None)
            domains = getattr(orig_model, 'seq_domains', None)
        if domains is None:
            return seq_repr_list, seq_mask_list
        filtered_repr = [t for t, d in zip(seq_repr_list, domains) if d in keep]
        filtered_mask = [m for m, d in zip(seq_mask_list, domains) if d in keep]
        if not filtered_repr or not filtered_mask:
            return None, None
        return filtered_repr, filtered_mask

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run one training step and return detached training statistics.

        The returned tensors stay on-device so the outer loop can aggregate
        them and only materialize scalar values every ``log_every`` steps.
        """
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        # set_to_none=True saves one kernel launch per param vs. filling zeros.
        self.dense_optimizer.zero_grad(set_to_none=True)
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad(set_to_none=True)

        model_input = self._make_model_input(device_batch)

        # Autocast region: covers forward + loss. Embedding lookups are kept
        # in fp32 under autocast (PyTorch default), which is what we want.
        autocast_ctx = torch.autocast(
            device_type=self.device_type,
            dtype=self.amp_dtype,
            enabled=self.amp,
        )
        with autocast_ctx:
            aux_model = self._get_aux_model()
            aux_enabled = (
                (self.use_aux_loss or self.use_supcon)
                and aux_model is not None
            )
            if aux_enabled:
                logits, aux = aux_model.forward_with_aux(model_input)
            else:
                logits = self.model(model_input)
                aux = None
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss_main = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
            else:
                loss_main = F.binary_cross_entropy_with_logits(logits, label)

            base_loss = loss_main
            loss_infonce_val = base_loss.new_zeros(())
            loss_supcon_val = base_loss.new_zeros(())
            if aux is not None:
                if len(aux) >= 4:
                    u_repr, i_repr, s_repr, position_aux = aux[:4]
                else:
                    u_repr, i_repr, s_repr = aux
                    position_aux = None
                if self.use_aux_loss and self.aux_loss_weight > 0:
                    batch_size = u_repr.shape[0]
                    if batch_size > 1:
                        row_ids = torch.arange(batch_size, device=u_repr.device)
                        option_cols = self._sample_option_columns(row_ids, batch_size)
                        option_item_ids = self._gather_option_item_ids(
                            device_batch.get('item_id'),
                            option_cols,
                        )
                        rank_scores = aux_model.score_aux_candidates(
                            u_repr,
                            i_repr,
                            option_cols,
                        )
                        loss_candidate = option_softmax_loss(
                            rank_scores,
                            option_item_ids=option_item_ids,
                            temperature=self.aux_temperature,
                            labels=label,
                            positive_weight=self.aux_positive_weight,
                        )
                        loss_nce = loss_candidate

                        if self.aux_history_weight > 0 and self.aux_history_max_per_sample > 0:
                            seq_repr_list, seq_mask_list = self._filter_history_aux(position_aux)
                            if seq_repr_list is not None and seq_mask_list is not None:
                                loss_history = listwise_rank_infonce_loss(
                                    u_repr,
                                    i_repr,
                                    seq_repr_list=seq_repr_list,
                                    seq_mask_list=seq_mask_list,
                                    labels=label,
                                    temperature=self.aux_temperature,
                                    pos_only=False,
                                    pos_weight=self.aux_history_pos_weight,
                                    max_seq_neg_per_sample=self.aux_history_max_per_sample,
                                    seq_proj=getattr(aux_model, 'aux_item_head', None),
                                )
                                loss_nce = loss_nce + self.aux_history_weight * loss_history

                        base_loss = base_loss + self.aux_loss_weight * loss_nce
                        loss_infonce_val = loss_nce.detach()
                if self.use_supcon and self.supcon_weight > 0:
                    loss_sup = supcon_loss(
                        s_repr, label.long(),
                        temperature=self.supcon_temperature,
                    )
                    base_loss = base_loss + self.supcon_weight * loss_sup
                    loss_supcon_val = loss_sup.detach()

        self._aux_global_step += 1

        # Backward (+ unscale) under GradScaler if active; otherwise vanilla.
        # Candidate-rank NCE is part of the same graph as the main forward and
        # only adds a lightweight [B,K] scoring head instead of replaying the
        # long-sequence backbone for every candidate.
        if self.grad_scaler is not None:
            self.grad_scaler.scale(base_loss).backward()
        else:
            base_loss.backward()

        loss = (loss_main.detach()
                + self.aux_loss_weight * loss_infonce_val.detach()
                + self.supcon_weight * loss_supcon_val.detach())

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.dense_optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.grad_scaler.step(self.dense_optimizer)
            self.grad_scaler.update()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        # Per-step cosine LR schedule for the dense optimizer.
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # EMA shadow update (dense params only). Gated by ``_ema_started``
        # which flips to True at the start of ``ema_start_epoch`` so the
        # first epoch's noisy steps do not pollute the average.
        if self.ema is not None and self._ema_started:
            self.ema.update()

        # Return detached tensors – caller decides when to materialize them.
        return {
            'loss': loss.detach(),
            'loss_main': loss_main.detach(),
            'loss_infonce': loss_infonce_val.detach(),
            'loss_supcon': loss_supcon_val.detach(),
            'grad_norm': grad_norm.detach().float(),
        }

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        autocast_ctx = torch.autocast(
            device_type=self.device_type,
            dtype=self.amp_dtype,
            enabled=self.amp,
        )
        with autocast_ctx:
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)
        # Upcast to fp32 for stable metric accumulation
        logits = logits.float()
        return logits, label
