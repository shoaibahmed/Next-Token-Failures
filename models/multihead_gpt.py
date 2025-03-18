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
        - consider the last token as the sink token in the vocab -- added to the LM head
    """
    assert boundary_condition in ["normalize", "ignore", "sink"], boundary_condition
    assert head_size >= 1, head_size
    assert len(input_ids.shape) == 2, input_ids.shape  # B x L
    B, L = input_ids.shape

    if boundary_condition == "ignore":
        vocab_size += 1  # inflate the vocab size so that it can be discarded at the end -- sink already has an additional token

    new_L = L
    if boundary_condition in ["ignore", "sink"]:
        # Append sink token to the input_ids so that the model would naturally assign weight to the sink token
        sink_token_idx = vocab_size - 1  # sink token is the last token
        sink_tokens = torch.full(size=(B, head_size), fill_value=sink_token_idx, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, sink_tokens], dim=1)
        new_L = L + head_size

    relative_freq = torch.zeros(B, L, vocab_size).to(input_ids.device)  # B x L x V
    for start_idx in range(L):  # iterate over different positions in the sequence
        last_idx = min(start_idx + head_size, new_L)  # min only applied to the 'normalize' condition
        for b in range(B):  # need to iterate over B only due to the shape of valid_indices when using the ignore_idx
            current_input_chunk = input_ids[b, start_idx:last_idx]
            keep_mask = current_input_chunk != ignore_idx
            valid_indices = current_input_chunk[keep_mask]  # shape [num_valid]
            relative_freq[b, start_idx, :] = torch.scatter_add(relative_freq[b, start_idx, :], dim=0, index=valid_indices,
                                                               src=torch.ones(valid_indices.shape, device=relative_freq.device))

    # Normalize the frequencies
    row_sum = relative_freq.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    relative_freq = relative_freq / row_sum  # normalize by the actual frequency
    if boundary_condition == "ignore":
        relative_freq = relative_freq[:, :, :-1]  # chop the last additional sink token which results in an unnormalized target dist

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

    if boundary_condition == "ignore":
        vocab_size += 1  # inflate the vocab size so that it can be discarded at the end -- sink already has an additional token

    new_L = L
    if boundary_condition in ["ignore", "sink"]:
        # Append sink token to the input_ids so that the model would naturally assign weight to the sink token
        sink_token_idx = vocab_size - 1  # sink token is the last token
        sink_tokens = torch.full(size=(B, head_size), fill_value=sink_token_idx, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, sink_tokens], dim=1)
        new_L = L + head_size

    # Generate window indices
    t_indices = torch.arange(L, device=device)
    window_indices = t_indices.view(-1, 1) + torch.arange(head_size, device=device)
    valid_indices_mask = window_indices < new_L

    # Clamp indices to valid range and replace out-of-bounds with 0
    window_indices = torch.where(valid_indices_mask, window_indices, torch.tensor(0, device=device))
    window_indices = torch.clamp(window_indices, 0, new_L - 1)  # Ensure indices are within bounds

    # Expand indices for batch and gather tokens
    window_indices_expanded = window_indices.unsqueeze(0).expand(B, -1, -1)  # (B, L, head_size)
    input_expanded = input_ids.unsqueeze(1).expand(-1, L, -1)  # (B, L, new_L)
    tokens = torch.gather(input_expanded, 2, window_indices_expanded)  # (B, L, head_size)

    # Create masks for valid and non-ignored tokens
    ignore_mask = (tokens != ignore_idx)
    combined_mask = valid_indices_mask.unsqueeze(0) & ignore_mask

    # Filter invalid/ignored tokens (set to 0 to avoid out-of-bounds in scatter)
    tokens_filtered = torch.where(combined_mask, tokens, torch.tensor(0, device=device))

    # Compute frequencies using scatter_add
    relative_freq = torch.zeros(B, L, vocab_size, device=device)
    relative_freq.scatter_add_(2, tokens_filtered, combined_mask.float())

    # Normalize and handle sink token
    row_sum = relative_freq.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    relative_freq = relative_freq / row_sum  # normalize by the actual frequency
    if boundary_condition == "ignore":
        relative_freq = relative_freq[:, :, :-1]  # chop the last additional sink token which results in an unnormalized target dist

    return relative_freq


class MultiheadGPT(Transformer):
    def __init__(self, config):
        if config.boundary_condition == "sink":
            config.vocab_size += 1  # add the sink token to the model's vocab
            print("Updated vocab size with sink token:", config.vocab_size)

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

    for boundary_condition in ["normalize", "ignore", "sink"]:
        print("="*50)
        print("~"*20, boundary_condition, "~"*20)
        print("="*50)

        input_ids = get_input()
        print(input_ids.shape)
        print(input_ids)
        targets = compute_targets(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx, boundary_condition=boundary_condition)
        print(targets.shape)
        print(targets)

        targets_optim = compute_targets_optimized(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx,
                                                  boundary_condition=boundary_condition)
        print(targets_optim.shape)
        print(targets_optim)
        assert (targets == targets_optim).all(), f"{torch.where(targets != targets_optim)}"

        n_samples = 100
        for _ in tqdm(range(n_samples)):
            input_ids = get_input()
            targets = compute_targets(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx, boundary_condition=boundary_condition)
            targets_optim = compute_targets_optimized(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx,
                                                      boundary_condition=boundary_condition)
            assert (targets == targets_optim).all()

        start_time = time.time()
        for _ in tqdm(range(n_samples)):
            input_ids = get_input()
            targets = compute_targets(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx, boundary_condition=boundary_condition)
        elapsed = time.time() - start_time
        print(f"Base implementation elapsed time: {elapsed:.2f} secs")

        start_time = time.time()
        for _ in tqdm(range(n_samples)):
            input_ids = get_input()
            targets = compute_targets_optimized(input_ids, vocab_size, head_size=10, ignore_idx=ignore_idx,
                                                boundary_condition=boundary_condition)
        elapsed = time.time() - start_time
        print(f"Optimized implementation elapsed time: {elapsed:.2f} secs")
