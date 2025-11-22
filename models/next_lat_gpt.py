import copy

import torch
import torch.nn as nn

from models.base_model import Transformer
from models.gpt import Block
from utils.training_utils import accuracy


class LatentDynamicsModel(torch.nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()

        # Follow the setup from https://arxiv.org/abs/2511.05963 (appendix C)
        # Note: we are using the representation of the model after the final layer norm
        input_dim = 2 * n_embd  # previous hidden dim and input embedding
        self.latent_dynamics_model = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, n_embd),
            torch.nn.GELU(),
            torch.nn.Linear(n_embd, n_embd),
            torch.nn.GELU(),
            torch.nn.Linear(n_embd, n_embd),
        ) # three layered MLP

    def forward(self, token_embed: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        input_state = torch.cat([token_embed, h], dim=-1)  # concatenate in hidden dim
        pred_h = h + self.latent_dynamics_model(input_state)
        return pred_h


class NextLatGPT(Transformer):
    def __init__(self, config):
        super().__init__(config, block=Block)
        # Add positional encoding
        self.pos_encoding = nn.Embedding(config.block_size, config.n_embd)
        # Tie weights
        self.embed_tokens.weight = self.lm_head.weight

        # Set the new args
        self.pred_horizon = config.pred_horizon

        # Setup the latent dynamics model
        self.latent_dynamics_model = LatentDynamicsModel(config.n_embd)

        # Define the loss functions
        self.ignore_idx = -1
        self.loss_smooth_l1 = torch.nn.SmoothL1Loss()
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)  # Note: input in log space
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_idx)

        # Define the loss weights
        self.next_lat_lambda = config.next_lat_lambda
        self.kl_lambda = config.kl_lambda

    def forward(self, idx, targets=None):
        device = idx.device
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only " \
                                                  f"{self.config.block_size}"
        tok_emb = self.embed_tokens(idx)
        start_pos = 0 if self.cache is None or not self.cache.use_caching else self.cache.cur_seq_len[0]
        pos = torch.arange(start_pos, seq_len + start_pos, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_encoding(pos)
        tokens = tok_emb + pos_emb

        # Base transformer layers
        x = tokens
        for block in self.layers:
            x = block(x, self.cache)
        latents = self.final_layernorm(x)

        # Compute the next token prediction loss
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(latents)
            # Calculate loss with ignore_index=-1, meaning we skip the gradient contributions from those tokens
            # which is basically the prefix tokens
            loss = self.loss_ce(logits.view(-1, logits.size(-1)), targets.view(-1))
            acc, token_acc = accuracy(logits, targets)
            accs = {"acc": acc, "token_acc": token_acc}
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss, accs = None, None

        # Compute the next-latent prediction loss
        total_loss = loss
        if loss is not None:
            regression_loss = 0.
            kl_loss = 0.
            probs = torch.softmax(logits.detach(), dim=-1)
            input_latents = latents[:, :-1, :]  # discard last latent for the target
            for horizon in range(1, self.pred_horizon+1):
                # Get the input and target latents
                target_latents = latents[:, horizon:, :]
                next_token = tokens[:, horizon:]
                target_probs = probs[:, horizon:, :]

                # Roll the dynamics model to predict the next latents
                predicted_latents = self.latent_dynamics_model(input_latents, next_token)

                # Compute the smooth L1 loss on the predicted latents (note: detach is important)
                regression_loss = regression_loss + self.loss_smooth_l1(predicted_latents, target_latents.detach())

                # Compute the KL loss using the output head (with the output head frozen)
                # A computationally bad but visually elegant way to do it would be: copy.deepcopy(self.lm_head)(predicted_latents)
                predicted_logits = torch.nn.functional.linear(
                    predicted_latents, 
                    self.lm_head.weight.detach(), 
                    bias=self.lm_head.bias.detach() if self.lm_head.bias is not None else None
                )
                predicted_log_probs = torch.nn.functional.log_softmax(predicted_logits, dim=-1)
                kl_loss = kl_loss + self.kl_div(predicted_log_probs, target_probs.detach())

                # Use the predicted latents as the input for the next iteration
                input_latents = predicted_latents[:, 1:, :]  # move by a stride of 1

            # Compute the total loss
            total_loss = loss + self.next_lat_lambda * regression_loss + self.kl_lambda * kl_loss

        return logits, total_loss, accs


if __name__ == "__main__":
    # Execution: python -m models.next_lat_gpt

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

    start_time = time.time()
    n_samples = 100
    for _ in tqdm(range(n_samples)):
        input = get_input()  # get the input
        # TODO: forward prop through the model
    elapsed = time.time() - start_time
    print(f"Base implementation elapsed time: {elapsed:.2f} secs")
