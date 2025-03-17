import copy

import torch
import torch.nn as nn

from models.base_model import Transformer
from models.gpt import Block
from utils.training_utils import accuracy


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, apply_log_softmax: bool = True):
        super().__init__()
        self.apply_log_softmax = apply_log_softmax

    def forward(self, pred_log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred_log_probs.shape == target.shape, f"{pred_log_probs.shape} != {target.shape}"
        if self.apply_log_softmax:
            pred_log_probs = torch.nn.functional.log_softmax(pred_log_probs, dim=-1)
        keep_mask = target.sum(dim=-1, keepdim=True) > 0  # [B, L, 1] -- ignore index have a 0 distribution as the target
        num_elements_kept = max(int(keep_mask.sum()), 1)  # should normalize by at least by 1
        loss = - (keep_mask * target * pred_log_probs).sum(dim=-1).sum() / num_elements_kept  # B x L x V
        return loss


def compute_targets(input_ids: torch.Tensor, vocab_size: int, head_size: int, ignore_idx: int = -100,
                    boundary_condition: str = "normalize") -> torch.Tensor:
    """
    Boundary condition defines how we deal with the boundary condition as we reach the end of the sequence.
    - normalize: just renormalize the sequence based on the available elements and count
    - ignore: normalize and ignore the mass assigned the sink token (assumes that there is a sink token) -- results in an unnormalized distribution
    - sink: have a separate sink token, similar to the BOS token
    """
    assert boundary_condition in ["normalize", "ignore", "sink"], boundary_condition
    assert head_size >= 1, head_size
    assert len(input_ids.shape) == 2, input_ids.shape  # B x L
    B, L = input_ids.shape

    relative_freq = torch.zeros(B, L, vocab_size).to(input_ids.device)  # B x L x V
    for start_idx in range(L):  # iterate over different positions in the sequence
        last_idx = min(start_idx + head_size, L)
        for b in range(B):  # need to iterate over B only due to the shape of valid_indices when using the ignore_idx
            current_input_chunk = input_ids[b, start_idx:last_idx]
            keep_mask = current_input_chunk != ignore_idx
            valid_indices = current_input_chunk[keep_mask]  # shape [num_valid]
            relative_freq[b, start_idx, :] = torch.scatter_add(relative_freq[b, start_idx, :], dim=0, index=valid_indices,
                                                               src=torch.ones(valid_indices.shape, device=relative_freq.device))

    if boundary_condition == "normalize":
        row_sum = relative_freq.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        relative_freq = relative_freq / row_sum  # normalize by the actual frequency
    else:
        raise NotImplementedError(f"Boundary condition {boundary_condition} not implemented!")
    return relative_freq


def compute_targets_optimized(input_ids: torch.Tensor, vocab_size: int, head_size: int, ignore_idx: int = -100,
                              boundary_condition: str = "normalize") -> torch.Tensor:
    """
    Same as the function 'compute_targets', but just faster due to vectorization
    """
    assert boundary_condition in ["normalize", "ignore", "sink"], boundary_condition
    assert head_size >= 1, head_size
    assert len(input_ids.shape) == 2, input_ids.shape  # B x L
    B, L = input_ids.shape

    device = input_ids.device
    # Generate window indices and masks
    t_indices = torch.arange(L, device=device)  # L
    window_indices = t_indices.view(-1, 1) + torch.arange(head_size, device=device).view(1, -1)  # (L, head_size)
    valid_indices_mask = window_indices < L  # (L, head_size)
    window_indices = torch.where(valid_indices_mask, window_indices, torch.tensor(0, dtype=torch.long, device=device))

    # Expand indices for batch and gather tokens
    window_indices_expanded = window_indices.unsqueeze(0).expand(B, -1, -1)  # (B, L, head_size)
    input_expanded = input_ids.unsqueeze(1).expand(-1, L, -1)  # (B, L, L)
    tokens = torch.gather(input_expanded, 2, window_indices_expanded)  # (B, L, head_size)

    # Create combined mask (valid indices and not ignored)
    ignore_mask = (tokens != ignore_idx)
    combined_mask = valid_indices_mask.unsqueeze(0) & ignore_mask  # (B, L, head_size)

    # Replace ignored tokens with 0 to avoid out-of-bounds (results in an error otherwise)
    tokens_filtered = tokens.clone()
    tokens_filtered[~combined_mask] = 0

    # Compute relative frequencies using scatter_add
    relative_freq = torch.zeros(B, L, vocab_size, device=device)
    relative_freq.scatter_add_(
        dim=2,
        index=tokens_filtered,          # no negative indices now!
        src=combined_mask.float()
    )

    # Normalize if required
    if boundary_condition == "normalize":
        row_sum = relative_freq.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        relative_freq = relative_freq / row_sum
    else:
        raise NotImplementedError(f"Boundary condition {boundary_condition} not implemented!")
    return relative_freq


class MultiheadGPT(Transformer):
    def __init__(self, config):
        super().__init__(config, block=Block)
        # Add positional encoding
        self.pos_encoding = nn.Embedding(config.block_size, config.n_embd)
        # Tie weights
        self.embed_tokens.weight = self.lm_head.weight

        # Set the new args
        self.head_sizes = config.head_sizes
        self.head_weights = config.head_weights
        self.boundary_condition = config.boundary_condition
        assert len(self.head_sizes) == len(self.head_weights), f"{len(self.head_sizes)} != {len(self.head_weights)}"
        assert self.head_sizes[0] == 1, "First head should have a size of 1"

        # Setup the prediction heads
        self.prediction_heads = [Block(config, config.n_layers + layer_idx - 1) for layer_idx in range(len(self.head_sizes))]
        self.prediction_heads = torch.nn.ModuleList(self.prediction_heads)  # wrap into module list to ensure .to(device) works as expected
        del self.layers[-1]  # remove the final layer itself as it now absorbed in prediction_heads

        # Define the loss function
        self.ignore_idx = -1
        self.loss_fn = CrossEntropyLoss()

    def forward(self, idx, targets=None):
        device = idx.device
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only " \
                                                  f"{self.config.block_size}"
        tok_emb = self.embed_tokens(idx)
        start_pos = 0 if self.cache is None or not self.cache.use_caching else self.cache.cur_seq_len[0]
        pos = torch.arange(start_pos, seq_len + start_pos, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_encoding(pos)
        x = tok_emb + pos_emb

        # Base transformer layers
        for block in self.layers:
            x = block(x, self.cache)

        logits, accs = None, None
        total_loss = 0.
        for head_idx, head_size in enumerate(self.head_sizes):
            head_output = self.prediction_heads[head_idx](x, self.cache)
            head_output = self.final_layernorm(head_output)

            if targets is not None:
                targets = targets.to(device)
                head_logits = self.lm_head(head_output)
                vocab_size = head_logits.shape[-1]  # B x L x V
                # Calculate loss with ignore_index=-1, meaning we skip the gradient contributions from those tokens which is basically the prefix tokens
                head_targets = compute_targets_optimized(targets, vocab_size, head_size, self.ignore_idx, boundary_condition=self.boundary_condition)
                head_loss = self.loss_fn(head_logits, head_targets)
                total_loss += head_loss * self.head_weights[head_idx]
                if head_idx == 0:
                    assert head_size == 1, f"head size should be 1. found: {head_size}"
                    acc, token_acc = accuracy(head_logits, targets)
                    accs = {"acc": acc, "token_acc": token_acc}
                    logits = head_logits
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(head_output[:, [-1], :])  # note: using list [-1] to preserve the time dim
                break

        return logits, total_loss, accs


if __name__ == "__main__":
    # Execution: python -m models.multihead_gpt

    import time
    from tqdm import tqdm

    vocab_size: int = 10
    seq_len = 100
    ignore_idx = -1
    prefix_len = int(0.1 * seq_len)

    def get_input():
        input_ids = torch.randint(0, vocab_size, size=(1, seq_len,))
        input_ids[:, :prefix_len] = ignore_idx  # mask the prefix
        return input_ids

    input_ids = get_input()
    print(input_ids.shape)
    print(input_ids)
    targets = compute_targets(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx)
    print(targets.shape)
    print(targets)

    targets_optim = compute_targets_optimized(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx)
    print(targets_optim.shape)
    print(targets_optim)
    assert (targets == targets_optim).all(), f"{torch.where(targets != targets_optim)}"

    n_samples = 100
    for _ in tqdm(range(n_samples)):
        input_ids = get_input()
        targets = compute_targets(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx)
        targets_optim = compute_targets_optimized(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx)
        assert (targets == targets_optim).all()

    start_time = time.time()
    for _ in tqdm(range(n_samples)):
        input_ids = get_input()
        targets = compute_targets(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx)
    elapsed = time.time() - start_time
    print(f"Base implementation elapsed time: {elapsed:.2f} secs")

    start_time = time.time()
    for _ in tqdm(range(n_samples)):
        input_ids = get_input()
        targets = compute_targets_optimized(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx)
    elapsed = time.time() - start_time
    print(f"Optimized implementation elapsed time: {elapsed:.2f} secs")
