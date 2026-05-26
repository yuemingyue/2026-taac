import os
import random
import copy
import logging
import time
from datetime import timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogFormatter:
    """Custom ``logging.Formatter`` that prefixes every record with the
    wall-clock timestamp and the elapsed wall-clock time since this
    formatter instance was constructed.

    The prefix format is ``"<locale-date> <locale-time> - H:MM:SS"``, which
    is convenient for tracking long-running training runs where both the
    absolute time and the time-since-start are useful.

    Multi-line messages are re-indented so that continuation lines align
    with the beginning of the message (not the prefix).
    """

    def __init__(self) -> None:
        # Anchor used to compute the elapsed-time part of the log prefix.
        # Can be reset at runtime via ``create_logger(...).reset_time()``.
        self.start_time: float = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s" % (
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        # Indent continuation lines so they line up with the message body,
        # not with the timestamp prefix.
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath: str) -> logging.Logger:
    """Create and configure the root logger for a training/inference run.

    The returned logger has two handlers attached:

    * A ``FileHandler`` bound to ``filepath`` (opened in write mode,
      truncating any previous content) that records ``DEBUG``-level and
      above messages for post-mortem inspection.
    * A ``StreamHandler`` to stderr that only echoes ``INFO``-level and
      above messages, keeping the console output concise.

    Both handlers share a ``LogFormatter`` so the console and the log file
    stay in sync. Any pre-existing handlers on the root logger are removed
    to avoid duplicate lines when this function is called multiple times.

    Args:
        filepath: Destination path of the log file. Opened in ``"w"`` mode,
            so previous contents are overwritten.

    Returns:
        The root ``logging.Logger`` instance. The returned object is
        augmented with a ``reset_time()`` attribute that resets the
        elapsed-time clock used by the log prefix. This is useful when the
        "interesting" phase of a run starts well after process launch
        (e.g. after schema building and data loading).
    """
    log_formatter = LogFormatter()

    file_handler = logging.FileHandler(filepath, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Allow callers to reset the elapsed-time clock shown in the log prefix.
    def reset_time() -> None:
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # type: ignore[attr-defined]

    return logger


class EarlyStopping:
    """Early-stop training when the validation metric plateaus.

    The tracker assumes a *higher-is-better* metric (typical for AUC or
    accuracy). A candidate ``score`` is considered an improvement iff
    ``score > best_score + delta``; otherwise the internal ``counter`` is
    incremented and training is requested to stop once
    ``counter >= patience``.

    On every improvement the current ``model.state_dict()`` is both
    deep-copied in memory (``self.best_model``) and persisted to disk at
    ``checkpoint_path``. The most recent *improving* score is cached in
    ``self.best_saved_score`` so callers can skip redundant IO.

    Attributes:
        checkpoint_path: Destination path for the best ``state_dict``.
        patience: Number of non-improving calls tolerated before
            ``early_stop`` is flipped to ``True``.
        verbose: If ``True``, emit an ``INFO`` line whenever a checkpoint
            is written.
        counter: Number of consecutive non-improving calls seen so far.
        best_score: Best score observed; ``None`` until the first call.
        early_stop: Set to ``True`` once ``counter >= patience``.
        delta: Minimum absolute improvement required to reset ``counter``.
        best_model: In-memory deep copy of the best ``state_dict``.
        best_saved_score: Score associated with the last checkpoint
            actually written to disk.
        best_extra_metrics: Optional auxiliary metrics captured at the
            best-score step (e.g. logloss, other AUCs).
        label: Short prefix (e.g. ``"val"``) prepended to log lines to
            disambiguate multiple trackers running in parallel.
    """

    def __init__(
        self,
        checkpoint_path: str,
        label: str = "",
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path: str = checkpoint_path
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.delta: float = delta
        self.best_model: Optional[Dict[str, torch.Tensor]] = None
        self.best_saved_score: float = 0.0
        self.best_extra_metrics: Optional[Dict[str, Any]] = None
        self.label: str = label
        if self.label != "":
            self.label += " "

    def _is_not_improved(self, score: float) -> bool:
        """Return ``True`` iff ``score`` fails to beat ``best_score + delta``.

        Used as the gating condition for incrementing the patience counter.
        ``best_score`` must have been seeded by a prior ``__call__``.
        """
        assert self.best_score is not None, "call __call__ first to seed best_score"
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Feed a new validation score into the tracker.

        Three branches, in order:

        1. First call (``best_score is None``): seed the tracker, persist a
           checkpoint, and cache the model weights.
        2. Not improved: increment ``counter`` and log the progress; flip
           ``early_stop`` once ``counter >= patience``.
        3. Improved: reset ``counter`` to ``0``, update ``best_score`` and
           ``best_extra_metrics``, refresh the in-memory ``best_model``,
           and write a new checkpoint to disk.

        Args:
            score: Scalar validation metric (higher is better, e.g. AUC).
            model: Model whose ``state_dict`` is snapshotted on
                improvement. Only the parameters are saved, not the
                optimizer state.
            extra_metrics: Optional dict of auxiliary metrics recorded at
                the same step, e.g.
                ``{"best_val_AUC": ..., "best_val_logloss": ...}``. Stored
                verbatim as ``self.best_extra_metrics``; not interpreted
                by ``EarlyStopping`` itself.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            self.counter += 1
            logging.info(f'{self.label}earlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info(f'{self.label}earlyStopping counter reset!')
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        """Persist ``model.state_dict()`` to ``self.checkpoint_path``.

        Creates any missing parent directories, writes atomically via
        ``torch.save``, and records ``score`` as ``self.best_saved_score``
        so subsequent callers can detect "no new improvement since last
        save" without re-reading the checkpoint file.

        Args:
            score: Validation score associated with the weights being
                saved. Exposed to callers via ``best_saved_score`` after
                the write completes.
            model: Model whose parameters are being snapshotted. Only
                ``state_dict()`` is written; optimizer and scheduler state
                are explicitly *not* included.
        """
        if self.verbose:
            logging.info('Validation score increased. Saving model ...')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_saved_score = score


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence training reproducibility.

    Seeds ``random``, the ``PYTHONHASHSEED`` env var, NumPy, the CPU
    PyTorch generator and all CUDA generators, then forces cuDNN into
    deterministic mode.

    Note that full bitwise determinism on GPU also requires disabling
    cuDNN auto-tuning (``torch.backends.cudnn.benchmark = False``) and may
    come with a non-trivial throughput cost; this helper intentionally
    only toggles ``deterministic`` to preserve speed for common use cases.

    Args:
        seed: Non-negative integer seed shared by all RNGs listed above.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (N,) raw logits (before sigmoid).
        targets: (N,) binary labels {0, 1}.
        alpha: positive-class weight in (0, 1). When positives dominate,
            use alpha < 0.5 to downweight the positive class.
        gamma: focusing parameter. gamma=0 degenerates to standard BCE;
            gamma=2 is the standard value.
        reduction: 'mean' | 'sum' | 'none'.
    """
    p = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def sampled_nce_loss(
    query: torch.Tensor,
    pos_key: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 0.6,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Sampled NCE auxiliary loss migrated from the 111 implementation.

    The original in-batch CLIP-style loss builds a ``B x B`` matrix and uses
    other positive items as negatives. This sampled NCE variant follows 111's
    single-direction form:

        loss_i = -pos_i + log(exp(pos_i) + sum_j exp(neg_ij))

    In the pointwise PCVR setting, positive anchors/keys are rows with
    ``label==1`` and the explicit negative key pool is formed by item-side
    representations from rows with ``label==0``. If no labels are supplied,
    all rows are used as anchors and a one-step rolled item view is used as a
    fallback negative pool.

    Args:
        query: (B, D) query/user-side representation.
        pos_key: (B, D) matched item-side positive representation.
        labels: (B,) optional binary labels used to split positive anchors
            and explicit negative keys.
        temperature: softmax temperature. The migration default is 0.6.
        chunk_size: Number of negative keys processed per matmul chunk.

    Returns:
        Scalar loss. Returns 0 when a batch has no positive anchors or no
        available negative keys.
    """
    query = F.normalize(query.float(), dim=-1)
    pos_key = F.normalize(pos_key.float(), dim=-1)

    if query.shape[0] <= 1:
        return query.new_zeros(())

    if labels is not None:
        labels_bool = labels.view(-1).bool()
        pos_mask = labels_bool
        neg_mask = ~labels_bool
        if pos_mask.sum() < 1 or neg_mask.sum() < 1:
            return query.new_zeros(())
        query_embs = query[pos_mask]
        pos_key_embs = pos_key[pos_mask]
        neg_key_embs = pos_key[neg_mask]
    else:
        query_embs = query
        pos_key_embs = pos_key
        neg_key_embs = torch.roll(pos_key, shifts=1, dims=0)

    if neg_key_embs.dim() == 3:
        neg_key_embs = neg_key_embs.reshape(-1, neg_key_embs.shape[-1])
    if neg_key_embs.shape[0] < 1:
        return query.new_zeros(())

    temperature = max(float(temperature), 1e-6)
    chunk_size = max(1, int(chunk_size))

    pos_logits = torch.einsum('bd,bd->b', query_embs, pos_key_embs)

    log_sum_exp_neg_logits = torch.full(
        (query_embs.shape[0],),
        fill_value=-1e9,
        device=query.device,
        dtype=torch.float32,
    )
    for start in range(0, neg_key_embs.shape[0], chunk_size):
        neg_chunk = neg_key_embs[start:start + chunk_size]
        neg_logits_chunk = torch.matmul(query_embs, neg_chunk.t())
        neg_logits_chunk = neg_logits_chunk / temperature
        lse_chunk = torch.logsumexp(neg_logits_chunk, dim=1)
        log_sum_exp_neg_logits = torch.logaddexp(log_sum_exp_neg_logits, lse_chunk)

    pos_logits = pos_logits / temperature
    log_denominator = torch.logaddexp(pos_logits, log_sum_exp_neg_logits)
    loss_per_sample = -pos_logits + log_denominator
    return loss_per_sample.mean()


def option_softmax_loss(
    option_logits: torch.Tensor,
    option_item_ids: Optional[torch.Tensor] = None,
    temperature: float = 0.6,
    labels: Optional[torch.Tensor] = None,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    """Softmax NCE over per-row logits from the main scoring path.

    ``option_logits[:, 0]`` is the observed pair score for each row, while the
    remaining columns are sampled candidate item scores for the same
    user/context side. When item ids are provided, repeated items are removed
    from the denominator except for the first-column positive.

    ``labels`` and ``positive_weight`` inject the old implementation's PCVR
    bias: converted samples (label==1) contribute more to the auxiliary rank
    objective, while non-converted rows still serve as valid queries/negatives.
    """
    if option_logits.dim() != 2:
        raise ValueError(
            f"option_logits must be 2D, got {tuple(option_logits.shape)}")

    batch_rows, option_cols = option_logits.shape
    if batch_rows <= 0 or option_cols <= 1:
        return option_logits.new_zeros(())

    device = option_logits.device
    logits = option_logits.float() / max(float(temperature), 1e-6)

    if option_item_ids is not None:
        ids = option_item_ids.to(device)
        if ids.shape == option_logits.shape:
            same_item = ids.eq(ids[:, :1])
            keep_mask = ~same_item
            keep_mask[:, 0] = True
            logits = logits.masked_fill(~keep_mask, -1e9)

    targets = torch.zeros(batch_rows, dtype=torch.long, device=device)
    loss_per_row = F.cross_entropy(logits, targets, reduction='none')

    if labels is None or float(positive_weight) == 1.0:
        return loss_per_row.mean()

    labels_f = labels.float().view(-1).to(device)
    if labels_f.numel() != batch_rows:
        return loss_per_row.mean()

    pos_w = max(float(positive_weight), 0.0)
    weights = torch.where(
        labels_f > 0.5,
        torch.full_like(labels_f, pos_w),
        torch.ones_like(labels_f),
    )
    denom = weights.sum().clamp(min=1e-6)
    return (loss_per_row * weights).sum() / denom


def pair_block_softmax_loss(
    pair_logits: torch.Tensor,
    row_ids: torch.Tensor,
    item_ids: Optional[torch.Tensor] = None,
    temperature: float = 0.6,
) -> torch.Tensor:
    """In-batch pair-logit NCE for a block of anchor rows.

    ``pair_logits[k, j]`` is the raw score of ``row_ids[k]``'s user/context
    side paired with the j-th item side in the mini-batch. The positive column
    for each row is its original row id. Repeated item ids are excluded from
    the denominator except for the positive column.
    """
    if pair_logits.dim() != 2:
        raise ValueError(f"pair_logits must be 2D, got {tuple(pair_logits.shape)}")

    block_rows, batch_cols = pair_logits.shape
    if block_rows <= 0 or batch_cols <= 1:
        return pair_logits.new_zeros(())

    device = pair_logits.device
    row_ids = row_ids.view(-1).to(device=device, dtype=torch.long)
    if row_ids.numel() != block_rows:
        raise ValueError(
            f"row_ids length {row_ids.numel()} does not match logits rows {block_rows}")

    logits = pair_logits.float() / max(float(temperature), 1e-6)

    if item_ids is not None:
        ids = item_ids.view(-1).to(device)
        if ids.numel() == batch_cols:
            same_item = ids.index_select(0, row_ids).view(block_rows, 1).eq(
                ids.view(1, batch_cols))
            keep_mask = ~same_item
            keep_mask[torch.arange(block_rows, device=device), row_ids] = True
            logits = logits.masked_fill(~keep_mask, -1e9)

    return F.cross_entropy(logits, row_ids)


def batch_pair_softmax_loss(
    pair_logits: torch.Tensor,
    item_ids: Optional[torch.Tensor] = None,
    temperature: float = 0.6,
) -> torch.Tensor:
    """Square-table wrapper for the block-wise pair-logit NCE."""
    if pair_logits.dim() != 2:
        raise ValueError(f"pair_logits must be 2D, got {tuple(pair_logits.shape)}")
    batch_rows, batch_cols = pair_logits.shape
    if batch_rows != batch_cols:
        raise ValueError(
            "batch_pair_softmax_loss expects a square in-batch table, "
            f"got {tuple(pair_logits.shape)}")
    row_ids = torch.arange(batch_rows, device=pair_logits.device)
    return pair_block_softmax_loss(
        pair_logits,
        row_ids=row_ids,
        item_ids=item_ids,
        temperature=temperature,
    )


# Backward-compatible alias for older imports/configs. The implementation is
# intentionally no longer in-batch/symmetric; it now points to the migrated
# sampled NCE form above.
def info_nce_inbatch_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 0.6,
    pos_only: bool = True,
    symmetric: bool = False,
) -> torch.Tensor:
    del pos_only, symmetric
    return sampled_nce_loss(u, v, labels=labels, temperature=temperature)


def listwise_rank_infonce_loss(
    u_repr: torch.Tensor,
    i_repr: torch.Tensor,
    seq_repr_list: Optional[List[torch.Tensor]],
    seq_mask_list: Optional[List[torch.Tensor]],
    labels: Optional[torch.Tensor] = None,
    temperature: float = 0.6,
    pos_only: bool = False,
    pos_weight: float = 2.0,
    max_seq_neg_per_sample: int = 64,
    seq_proj: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Listwise rank InfoNCE with same-user history hard negatives.

    Candidate pool = observed self item + in-batch easy negatives + sampled
    hard negatives from the same user's behavior sequences. This keeps the new
    candidate-rank InfoNCE as the main branch while recovering the old version's
    strongest generalization signal from historical behaviors.
    """
    u_repr = u_repr.float()
    i_repr = i_repr.float()
    B, D = u_repr.shape
    if B <= 1:
        return u_repr.new_zeros(())

    seq_neg: Optional[torch.Tensor] = None
    seq_neg_mask: Optional[torch.Tensor] = None
    if (max_seq_neg_per_sample > 0
            and seq_repr_list is not None
            and seq_mask_list is not None
            and len(seq_repr_list) > 0):
        stacks = []
        masks = []
        for tok, msk in zip(seq_repr_list, seq_mask_list):
            tok_f = tok.float()
            if seq_proj is not None:
                tok_f = seq_proj(tok_f)
            if tok_f.shape[-1] != D:
                continue
            stacks.append(tok_f)
            masks.append(msk.bool())
        if len(stacks) > 0:
            seq_all = torch.cat(stacks, dim=1)
            mask_all = torch.cat(masks, dim=1)
            L_total = seq_all.shape[1]
            N_hard = min(max_seq_neg_per_sample, L_total)
            if N_hard > 0:
                rand_key = torch.rand(B, L_total, device=seq_all.device)
                rand_key = rand_key.masked_fill(mask_all, -1.0)
                top_vals, top_idx = rand_key.topk(N_hard, dim=1)
                gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, D)
                seq_neg = seq_all.gather(1, gather_idx)
                seq_neg_mask = (top_vals < 0)

    u = F.normalize(u_repr, dim=-1)
    i = F.normalize(i_repr, dim=-1)

    pos_logit = (u * i).sum(-1, keepdim=True)
    inbatch_logit = u @ i.t()
    eye = torch.eye(B, device=u.device, dtype=torch.bool)
    inbatch_logit = inbatch_logit.masked_fill(eye, -1e9)

    logits_list = [pos_logit, inbatch_logit]
    if seq_neg is not None and seq_neg_mask is not None:
        seq_neg_norm = F.normalize(seq_neg, dim=-1)
        seq_logit = torch.einsum('bd,bnd->bn', u, seq_neg_norm)
        seq_logit = seq_logit.masked_fill(seq_neg_mask, -1e9)
        logits_list.append(seq_logit)

    logits = torch.cat(logits_list, dim=-1)
    logits = logits / max(float(temperature), 1e-6)
    targets = torch.zeros(B, device=u.device, dtype=torch.long)
    loss_per_q = F.cross_entropy(logits, targets, reduction='none')

    if labels is None:
        return loss_per_q.mean()

    labels_f = labels.float().view(-1).to(u.device)
    if labels_f.numel() != B:
        return loss_per_q.mean()
    if pos_only:
        mask = labels_f > 0.5
        if mask.sum() < 1:
            return u.new_zeros(())
        return loss_per_q[mask].mean()

    pos_w = max(float(pos_weight), 0.0)
    weights = torch.where(
        labels_f > 0.5,
        torch.full_like(labels_f, pos_w),
        torch.ones_like(labels_f),
    )
    denom = weights.sum().clamp(min=1e-6)
    return (loss_per_q * weights).sum() / denom


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS'20).

    Pulls together samples sharing the same label and pushes apart samples
    with different labels in a single embedding space. Acts as a strong
    regularizer for binary classification with class imbalance.

    Args:
        z: (B, D) sample-level representation (will be L2-normed).
        labels: (B,) integer / 0-1 labels.
        temperature: softmax temperature.

    Returns:
        Scalar loss. Returns 0 if no anchor has a positive partner.
    """
    z = z.float()
    z = F.normalize(z, dim=-1)
    B = z.shape[0]
    if B <= 1:
        return z.new_zeros(())

    labels = labels.view(-1, 1)
    # (B, B) bool: same-label pairs (positives), excluding self.
    pos_mask = torch.eq(labels, labels.t()).float()
    eye = torch.eye(B, device=z.device)
    pos_mask = pos_mask - eye  # remove self
    pos_mask = pos_mask.clamp(min=0.0)

    # If no anchor has any positive partner, fall back to 0.
    if pos_mask.sum() < 1:
        return z.new_zeros(())

    sim = z @ z.t() / max(temperature, 1e-6)
    # Numerical stability: subtract row-wise max.
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    # Mask out self in the denominator.
    logits_mask = 1.0 - eye
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # Per-anchor mean log-prob over its positives.
    pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

    # Only anchors with at least one positive contribute.
    valid = (pos_mask.sum(dim=1) > 0).float()
    if valid.sum() < 1:
        return z.new_zeros(())
    loss = -(mean_log_prob_pos * valid).sum() / valid.sum()
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Exponential Moving Average (EMA) of model parameters
# ═══════════════════════════════════════════════════════════════════════════════
class EMAModel:
    """Polyak-style exponential moving average of *dense* model parameters.

    The standard update is:

        shadow_t = decay * shadow_{t-1} + (1 - decay) * raw_t

    Notes & deliberate design choices for this codebase:

    1. **Embedding parameters are excluded.** Embedding tables are huge,
       updated sparsely by Adagrad, and EMA-ing them would (a) explode
       memory and (b) hurt fresh-token learning. The caller passes in
       ``exclude_param_ids`` (typically ``{p.data_ptr() for p in
       model.get_sparse_params()}``) and we skip them.
    2. **Frozen / no-grad parameters are excluded** — they never change,
       so storing a shadow copy is wasted memory.
    3. **Shadow is held in fp32** even when the live model uses bf16/fp16.
       This avoids long-horizon precision drift when ``decay`` is high.
    4. **No bias correction.** PyTorch model averaging conventions skip
       ``ema_hat = ema / (1 - decay**t)`` because the model is then
       evaluated with the raw shadow; we follow the same convention.
    5. ``apply_shadow`` / ``restore`` are paired — ``apply_shadow`` swaps
       the live model parameters out into ``self._backup`` and writes the
       shadow values into ``param.data``; ``restore`` restores from
       ``self._backup``. This matches the common BYOL / MoCo pattern.
    """

    def __init__(
        self,
        named_params: 'list[tuple[str, nn.Parameter]]',
        decay: float = 0.999,
        exclude_param_ids: 'Optional[set[int]]' = None,
    ) -> None:
        self.decay: float = float(decay)
        # Pre-compute the actual list of (name, param) pairs that will be
        # EMA-tracked — embedding & frozen tensors get filtered up-front
        # so the hot ``update`` loop is a tight ``for`` over a small list.
        excl: set = set(exclude_param_ids or [])
        self._tracked: list = []
        self.shadow: dict = {}
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if p.data_ptr() in excl:
                continue
            self._tracked.append((name, p))
            # FP32 shadow on the same device as the live tensor.
            self.shadow[name] = p.detach().clone().float()
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, decay: 'Optional[float]' = None) -> None:
        """Apply one EMA step. Called *after* the live optimizer step."""
        d = self.decay if decay is None else float(decay)
        one_minus_d = 1.0 - d
        for name, p in self._tracked:
            s = self.shadow[name]
            # ``s = d * s + (1-d) * p``  – kept in fp32 regardless of p.dtype
            s.mul_(d).add_(p.detach().float(), alpha=one_minus_d)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        """Swap in EMA weights for evaluation / checkpointing.

        Live raw weights are stashed in ``self._backup`` so ``restore``
        can put them back unchanged.
        """
        for name, p in self._tracked:
            self._backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].to(p.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        """Undo the most recent ``apply_shadow`` and resume training from
        the live raw weights."""
        for name, p in self._tracked:
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup.clear()

    def num_tracked_params(self) -> int:
        return sum(p.numel() for _name, p in self._tracked)

    def num_tracked_tensors(self) -> int:
        return len(self._tracked)


# ═══════════════════════════════════════════════════════════════════════════════
# Behavior Importance Weighting Module
# ═══════════════════════════════════════════════════════════════════════════════


class BehaviorImportanceWeighting(nn.Module):
    """历史行为重要性加权模块，为目标item计算每个历史行为的重要性分数
    
    通过目标item与历史行为的交互，计算每个历史行为的重要性权重，
    然后对序列进行重新加权，使模型更关注与目标相关的历史行为。
    """

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        weighting_type: str = 'cross_attention',
        hidden_mult: int = 2,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sequences = num_sequences
        self.weighting_type = weighting_type
        
        if weighting_type == 'cross_attention':
            # 使用交叉注意力机制计算重要性
            self.importance_attentions = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=max(1, d_model // 16),
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_sequences)
            ])
        elif weighting_type == 'bilinear':
            # 使用双线性交互计算重要性
            self.bilinear_scorers = nn.ModuleList([
                nn.Bilinear(d_model, d_model, 1)
                for _ in range(num_sequences)
            ])
        elif weighting_type == 'mlp':
            # 使用MLP计算重要性
            self.mlp_scorers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * hidden_mult, 1)
                ) for _ in range(num_sequences)
            ])
        else:
            raise ValueError(f"Unsupported weighting_type: {weighting_type}")

    def forward(
        self,
        target_tokens: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_padding_masks: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """计算重要性权重并重新加权序列
        
        Args:
            target_tokens: (B, D) 目标item的特征向量
            seq_tokens_list: List[(B, L_i, D)] 各domain的历史行为序列
            seq_padding_masks: List[(B, L_i)] 各序列的padding mask
            
        Returns:
            List[(B, L_i, D)] 重要性加权后的序列
        """
        weighted_seqs = []
        
        for i in range(self.num_sequences):
            seq_tokens = seq_tokens_list[i]
            padding_mask = seq_padding_masks[i]
            
            if self.weighting_type == 'cross_attention':
                # 使用交叉注意力计算重要性
                target_expanded = target_tokens.unsqueeze(1)  # (B, 1, D)
                attn_output, attn_weights = self.importance_attentions[i](
                    query=target_expanded,
                    key=seq_tokens,
                    value=seq_tokens,
                    key_padding_mask=padding_mask,
                    need_weights=True
                )
                importance_weights = attn_weights.squeeze(1)  # (B, L_i)
                
            elif self.weighting_type == 'bilinear':
                # 使用双线性交互计算重要性
                target_expanded = target_tokens.unsqueeze(1).expand(-1, seq_tokens.shape[1], -1)
                importance_scores = self.bilinear_scorers[i](target_expanded, seq_tokens).squeeze(-1)
                importance_scores = importance_scores.masked_fill(padding_mask, float('-inf'))
                importance_weights = F.softmax(importance_scores, dim=-1)
                
            elif self.weighting_type == 'mlp':
                # 使用MLP计算重要性
                target_expanded = target_expanded.expand(-1, seq_tokens.shape[1], -1)
                interaction_input = torch.cat([target_expanded, seq_tokens], dim=-1)
                importance_scores = self.mlp_scorers[i](interaction_input).squeeze(-1)
                importance_scores = importance_scores.masked_fill(padding_mask, float('-inf'))
                importance_weights = F.softmax(importance_scores, dim=-1)
            
            # 应用重要性权重到序列
            weighted_seq = seq_tokens * importance_weights.unsqueeze(-1)
            weighted_seqs.append(weighted_seq)
        
        return weighted_seqs

    def get_importance_weights(
        self,
        target_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        domain_idx: int
    ) -> torch.Tensor:
        """获取特定domain的重要性权重（用于可视化分析）
        
        Args:
            target_tokens: (B, D) 目标item特征
            seq_tokens: (B, L, D) 历史行为序列
            padding_mask: (B, L) padding mask
            domain_idx: domain索引
            
        Returns:
            (B, L) 重要性权重
        """
        if self.weighting_type == 'cross_attention':
            target_expanded = target_tokens.unsqueeze(1)
            _, attn_weights = self.importance_attentions[domain_idx](
                query=target_expanded,
                key=seq_tokens,
                value=seq_tokens,
                key_padding_mask=padding_mask,
                need_weights=True
            )
            return attn_weights.squeeze(1)
        
        elif self.weighting_type == 'bilinear':
            target_expanded = target_tokens.unsqueeze(1).expand(-1, seq_tokens.shape[1], -1)
            scores = self.bilinear_scorers[domain_idx](target_expanded, seq_tokens).squeeze(-1)
            scores = scores.masked_fill(padding_mask, float('-inf'))
            return F.softmax(scores, dim=-1)
        
        elif self.weighting_type == 'mlp':
            target_expanded = target_expanded.expand(-1, seq_tokens.shape[1], -1)
            interaction_input = torch.cat([target_expanded, seq_tokens], dim=-1)
            scores = self.mlp_scorers[domain_idx](interaction_input).squeeze(-1)
            scores = scores.masked_fill(padding_mask, float('-inf'))
            return F.softmax(scores, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced MultiSeqQueryGenerator with Importance Weighting
# ═══════════════════════════════════════════════════════════════════════════════


class EnhancedMultiSeqQueryGenerator(nn.Module):
    """增强版多序列查询生成器，支持历史行为重要性加权
    
    在原始MultiSeqQueryGenerator的基础上，增加了目标感知的
    历史行为重要性加权功能，使query生成更关注与目标相关的历史行为。
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        use_importance_weighting: bool = False,
        weighting_type: str = 'cross_attention',
        importance_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.use_importance_weighting = use_importance_weighting

        global_info_dim = (num_ns + 1) * d_model
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # 原始query生成FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

        # 重要性加权模块
        if use_importance_weighting:
            self.importance_weighting = BehaviorImportanceWeighting(
                d_model=d_model,
                num_sequences=num_sequences,
                weighting_type=weighting_type,
                hidden_mult=2,
                dropout=importance_dropout
            )
        else:
            self.importance_weighting = None

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        target_tokens: Optional[torch.Tensor] = None
    ) -> list:
        """生成查询token，可选支持重要性加权
        
        Args:
            ns_tokens: (B, M, D) 共享NS token
            seq_tokens_list: List[(B, L_i, D)] 各domain的历史行为序列
            seq_padding_masks: List[(B, L_i)] 各序列的padding mask
            target_tokens: (B, D) 目标item特征（可选，用于重要性加权）
            
        Returns:
            List[(B, Nq, D)] 查询token列表
        """
        B = ns_tokens.shape[0]
        
        # 应用重要性加权（如果启用且有目标item）
        if self.use_importance_weighting and target_tokens is not None:
            seq_tokens_list = self.importance_weighting(
                target_tokens, seq_tokens_list, seq_padding_masks)

        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # 生成N个查询token
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list

    def get_importance_scores(
        self,
        target_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        domain_idx: int
    ) -> Optional[torch.Tensor]:
        """获取特定domain的重要性分数（用于分析和可视化）
        
        Args:
            target_tokens: (B, D) 目标item特征
            seq_tokens: (B, L, D) 历史行为序列
            padding_mask: (B, L) padding mask
            domain_idx: domain索引
            
        Returns:
            (B, L) 重要性分数，如果未启用重要性加权则返回None
        """
        if not self.use_importance_weighting or self.importance_weighting is None:
            return None
        
        return self.importance_weighting.get_importance_weights(
            target_tokens, seq_tokens, padding_mask, domain_idx)
