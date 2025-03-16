import copy
from typing import Dict

import torch
import torch.nn as nn

from models.base_model import Transformer
from models.gpt import Block
from utils.training_utils import accuracy


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, apply_log_softmax: bool = True, ignore_idx: int = -100):
        super().__init__()
        self.apply_log_softmax = apply_log_softmax
        self.ignore_idx = ignore_idx

    def forward(self, pred_log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred_log_probs.shape == target.shape, f"{pred_log_probs.shape} != {target.shape}"
        if self.apply_log_softmax:
            pred_log_probs = torch.nn.functional.log_softmax(pred_log_probs, dim=-1)
        keep_mask = target[:, :, 0:1] != self.ignore_idx  # assumes that all of the indicates are ignored if true (output shape: [B, L, 1])
        num_elements_kept = max(int(keep_mask.sum()), 1)  # should normalize by at least by 1
        loss = - (keep_mask * target * pred_log_probs).sum(dim=-1).sum() / num_elements_kept  # B x L x V
        return loss


def compute_targets(input_ids: torch.Tensor, vocab_size: int, head_size: int, boundary_condition: str = "normalize") -> torch.Tensor:
    """
    Boundary condition defines how we deal with the boundary condition as we reach the end of the sequence.
    - normalize: just renormalize the sequence based on the available elements and count
    - ignore: normalize and ignore the mass assigned the sink token (assumes that there is a sink token) -- results in an unnormalized distribution
    - sink: have a separate sink token, similar to the BOS token
    """
    assert boundary_condition in ["normalize", "ignore", "sink"], boundary_condition
    assert head_size >= 1, head_size
    assert len(input_ids.shape) == 2, input_ids.shape  # B x L
    B, seq_len = input_ids.shape

    relative_freq = torch.zeros(B, seq_len-1, vocab_size).to(input_ids.device)  # B x (L-1) x V
    weights = torch.ones_like(relative_freq[:, 0, :])  # B x V
    for start_idx in range(1, seq_len):  # targets are one token less than the sequence
        last_idx = min(start_idx + head_size, seq_len)
        current_input_chunk = input_ids[:, start_idx:last_idx]
        relative_freq[:, start_idx-1, :] = torch.scatter_add(relative_freq[:, start_idx-1, :], dim=1, index=current_input_chunk, src=weights)
    if boundary_condition == "normalize":
        relative_freq = relative_freq / relative_freq.sum(dim=-1, keepdim=True)  # normalize by the actual frequency
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
        assert len(self.head_sizes) == len(self.head_weights), f"{len(self.head_sizes)} != {len(self.head_weights)}"
        assert self.head_sizes[0] == 1, "First head should have a size of 1"

        # Setup the prediction heads
        self.prediction_heads = [copy.deepcopy(self.layers[-1]) for _ in range(len(self.head_sizes))]
        del self.layers[-1]  # remove the final layer itself

        # Define the loss function
        self.loss_fn = CrossEntropyLoss(ignore_idx=-1)

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
                head_logits = self.lm_head(head_output)
                vocab_size = head_logits.shape[-1]  # B x L x V
                # Calculate loss with ignore_index=-1, meaning we skip the gradient contributions from those tokens which is basically the prefix tokens
                head_targets = compute_targets(targets, vocab_size, head_size)
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
