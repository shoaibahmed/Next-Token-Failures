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
        num_elements_kept = int(keep_mask.sum())
        loss = - (keep_mask * target * pred_log_probs).sum(dim=-1).sum() / num_elements_kept  # B x L x V
        return loss


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
        self.prediction_heads = [copy.deepcopy(self.layers[-1]) for _ in range(len(self.head_horizon))]
        del self.layers[-1]  # remove the final layer itself

        # Define the loss function
        self.loss_fn = CrossEntropyLoss(ignore_idx=-1)

    def forward(self, idx, targets: Dict[int, torch.Tensor] = None):
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

        logits_list = []
        loss_list = []
        accs_list = []
        total_loss = 0.
        for head_idx, head_size in enumerate(self.head_sizes):
            head_output = self.prediction_heads[head_idx](x, self.cache)
            head_output = self.final_layernorm(head_output)

            if targets is not None:
                assert head_size in targets, f"{head_size} not in target keys {targets.keys()}"
                logits = self.lm_head(head_output)
                # Calculate loss with ignore_index=-1, meaning we skip the gradient contributions from those tokens
                # which is basically the prefix tokens
                loss = self.loss_fn(logits, targets[head_size])
                acc, token_acc = accuracy(logits, targets)
                accs = {"acc": acc, "token_acc": token_acc}
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(head_output[:, [-1], :])  # note: using list [-1] to preserve the time dim
                return logits, None, None  # no need to return a list -- makes it compatible with existing generate function

            logits_list.append(logits)
            loss_list.append(loss)
            accs_list.append(accs)

            total_loss += loss * self.head_weights[head_idx]

        return logits_list, total_loss, accs_list  # returns total loss instead of loss list
