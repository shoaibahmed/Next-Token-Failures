import os
import copy
import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import wandb

from tokenizing import get_tokenizer
from utils.training_utils import get_lr, get_run_name, AverageMeter
from data import get_dataset
from evaluate import evaluate, evaluate_forced
from models import get_model


# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction with GRPO")
# Data
parser.add_argument(
        "--model", type=str, default='gpt', help="Learning rate",
    )
parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of layers",
    )
parser.add_argument(
        "--n_embd", type=int, default=240, help="Embedding size",
    )
parser.add_argument(
        "--n_head", type=int, default=6, help="Number of heads",
    )
parser.add_argument(
    "--dataset", default='graph', type=str, help="Choice of dataset"
    )
parser.add_argument(
    "--n_train", default=200000, type=int, help="Number of training samples"
    )
parser.add_argument(
    "--n_test", default=10000, type=int, help="Number of test samples"
    )
parser.add_argument(
    "--num_nodes", default=50, type=int, help="Number of node values in graph"
    )
parser.add_argument(
    "--deg", default=2, type=int, help="Degree of starting node"
    )
parser.add_argument(
    "--path_len", default=5, type=int, help="Path length in star graph"
    )
parser.add_argument(
        "--mate_in", default=2, type=int, help="For chess, number of moves to checkmate"
    )
parser.add_argument(
        "--unrolled", action=argparse.BooleanOptionalAction, default=True, help="For chess, unrolled board state",
    )
parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size",
    )
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate",
    )
parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Strength of weight decay",
    )
parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs",
    )
parser.add_argument(
        "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )
parser.add_argument(
        "--eval_train", action=argparse.BooleanOptionalAction, default=False, help="Eval for training set",
    )
parser.add_argument(
        "--eval_every", type=int, default=1, help="Interval (in steps) to evaluate the model on test",
    )
parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Whether to use wandb",
    )
parser.add_argument(
        "--save_checkpoints", action=argparse.BooleanOptionalAction, default=False, help="Whether to save model checkpoints",
    )
parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb username",
    )
parser.add_argument(
    "--prediction_head_sizes", type=str, default='1', help='Prediction heads to be used for model training'
    )
parser.add_argument(
    "--prediction_head_weights", type=str, default='1', help='Weights to be assigned to the different predictions heads for model training'
    )
parser.add_argument(
    "--multihead_boundary_condition", type=str, default='normalize', help='Boundary condition to be used when computing multi-head targets'
    )
parser.add_argument(
        "--waypoint_len", type=str, default=None, help="Use waypoint task for the graph instead of the endpoint",
    )
parser.add_argument(
        "--grpo-group-size", type=int, default=16, help="GRPO group size",
    )
parser.add_argument(
        "--grpo-kl-beta", type=float, default=0.0, help="Beta value to be the used with the KL-divergence term in order to minimize deviation from the reference policy",
    )
parser.add_argument(
        "--use-grpo-val-set", action=argparse.BooleanOptionalAction, default=False, help="Use validation set for GRPO training",
    )

args = parser.parse_args()

assert 0 <= args.grpo_kl_beta, args.grpo_kl_beta

# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_entity = args.wandb_entity
wandb_log = args.use_wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Basic option validation
if args.waypoint_len is not None:
    if args.waypoint_len.isdigit():
        args.waypoint_len = int(args.waypoint_len)
    else:
        assert args.waypoint_len == "all", args.waypoint_len
    if isinstance(args.waypoint_len, int) and args.waypoint_len < 1:
        args.waypoint_len = None

# Model stuff
top_k = 1

# Evaluation stuff
eval_iters = 1000
eval_interval = 5
log_interval = 10

# Multi-head config
assert len(args.prediction_head_sizes.split(",")) == len(args.prediction_head_weights.split(",")), \
        f"list length mismatch: {args.prediction_head_sizes.split(',')} != {args.prediction_head_weights.split(',')}"

# Optimiser
dtype = 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
beta1 = 0.9
beta2 = 0.999
decay_lr = True
args.compile = False if device == 'cuda' else False
args.use_flash = True if device == 'cuda' else False
warmup_iters = 100
min_lr = 1e-5

run_name = get_run_name(args)

# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)
train_data, test_data = get_dataset(args, tokenizer, device)

val_data = None
if args.use_grpo_val_set:
    # Define a validation set (for GRPO)
    rng = np.random.default_rng(42)
    val_frac = 0.1
    val_len = int(val_frac * len(test_data))
    all_idx = rng.permutation(list(range((len(test_data)))))
    all_idx = [int(x) for x in all_idx]
    val_idx = all_idx[:val_len]
    test_idx = all_idx[val_len:]
    print(f"Selected idx / full test: {len(all_idx)} / val: {len(val_idx)} / test: {len(test_idx)}")

    # Define the dataset and the dataloader
    val_data = torch.utils.data.Subset(test_data, val_idx)
    test_data = torch.utils.data.Subset(test_data, test_idx)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
print(f"train dataset: {len(train_data)} / train loader: {len(train_loader)}")

val_loader = None
if val_data is not None:
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    print(f"val dataset: {len(val_data)} / val loader: {len(val_loader)}")

test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
print(f"test dataset: {len(test_data)} / test loader: {len(test_loader)}")

# Setup the training args
max_iters = len(train_data) * args.epochs
lr_decay_iters = max_iters

args.block_size = train_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None
model = get_model(args)

if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

model.to(device)
model.train()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# Define checkpoint path
checkpoint_dir = "./checkpoints/"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    print("Checkpoint directory created:", checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}.pth")
print("Using checkpoint path:", checkpoint_path)
assert os.path.exists(checkpoint_path), "GRPO training assumes the pretrained model already exists"
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

# Define the new checkpoint file for GRPO
run_name = f"{run_name}_grpo_group_size_{args.grpo_group_size}_kl_beta_{args.grpo_kl_beta}"
run_name += "_val" if args.use_grpo_val_set else ""
grpo_checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}.pth")

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures-grpo', entity=wandb_entity, config=args.__dict__,)
    wandb.run.name = run_name

# Evaluate the base model
print("Evaluating base model before GRPO training...")
results = {}
results = evaluate(model, train_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='base_train')
results = evaluate_forced(model, train_loader, ctx=ctx, results=results, mode='base_train')
if val_loader is not None:
    results = evaluate(model, val_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='base_val')
    results = evaluate_forced(model, val_loader, ctx=ctx, results=results, mode='base_val')
results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='base_test')
results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='base_test')
print(results)
if wandb_log:
    wandb.log(results)

results = {}
num_iters = 0

num_prefix_tokens = train_loader.dataset.num_prefix_tokens
num_target_tokens = train_loader.dataset.num_target_tokens
print(f"Prefix tokens: {num_prefix_tokens} / target tokens: {num_target_tokens}")

ref_model = None
if args.grpo_kl_beta > 0:
    # Clone the base model as reference policy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    print("Model cloned as reference policy for computing KL-divergence...")

for ep in range(args.epochs):
    train_bar = tqdm(val_loader if args.use_grpo_val_set else train_loader)
    total_loss, total_acc = AverageMeter(), AverageMeter()

    for x, y in train_bar:
        # determine and set the learning rate for this iteration
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Generate the output from the model
        model.eval()  # generate in eval mode
        prefix_x = x[:, :num_prefix_tokens]  # teacher forcing input with both the input and output tokens
        prefix_x = prefix_x.repeat_interleave(args.grpo_group_size, dim=0)  # repeat the input prompt for parallel inference
        with torch.no_grad(), ctx:  # output: (batch size * group size, num prefix tokens + num target tokens)
            y_pred = model.generate(prefix_x, num_target_tokens, temperature=0.8, top_k=top_k)

        # Compute log-probs in train mode
        model.train()  # compute the log probs in train mode
        new_generated_tokens = y_pred[:, -num_target_tokens:]  # ignore the prompt length (B, L')
        with ctx:  # (B*grpo_group_size)LV -> (B*grpo_group_size)OV
            targets_for_acc = -torch.ones_like(y_pred)
            targets_for_acc[:, num_prefix_tokens:] = y_pred[:, num_prefix_tokens:]
            logits, _, accs = model(y_pred[:, :-1].contiguous(), targets=targets_for_acc[:, 1:].contiguous())
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, -num_target_tokens:, :]  # -1 adjusted in model inputs
            assert new_generated_tokens.shape[1] == log_probs.shape[1], f"{new_generated_tokens.shape} != {log_probs.shape}"
            target_log_probs = torch.gather(log_probs, dim=2, index=new_generated_tokens.unsqueeze(-1)).squeeze(-1)  # BLV -> BL

        # Define the correct output reward based on exact match (B*grpo_group_size) -> (B, grpo_group_size)
        target_y = y[:, -num_target_tokens:].repeat_interleave(args.grpo_group_size, dim=0)
        rewards = target_y.eq(new_generated_tokens).sum(dim=-1).float()  # (B * grpo_group_size, num_tokens) -> ((B * grpo_group_size,)
        rewards = rewards.reshape(-1, args.grpo_group_size)

        # Compute the advantage score
        mean = rewards.mean(dim=1, keepdims=True)
        std = rewards.std(dim=1, keepdims=True)
        eps = 1e-6
        advantage = (rewards - mean) / (std + eps)
        advantage = advantage.detach().view(-1)  # flatten it out again (B*grpo_group_size)

        # Compute the policy gradient -- similar to REINFORCE
        reinforce_loss = - (target_log_probs * advantage.unsqueeze(-1)).mean()  # (BL, B1) -> scalar (negative log-likelihood)

        kl_div = None
        if args.grpo_kl_beta > 0:
            # Compute the KL-divergence from the GRPO paper: https://arxiv.org/abs/2402.03300
            with torch.no_grad(), ctx:  # BLV -> BOV
                ref_logits, _, _ = ref_model(y_pred[:, :-1].contiguous(), targets=y_pred[:, 1:].contiguous())  # targets required for right output shape
                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)[:, -num_target_tokens:, :]  # -1 adjusted in model inputs
                assert new_generated_tokens.shape[1] == ref_log_probs.shape[1], f"{new_generated_tokens.shape} != {ref_log_probs.shape}"
                ref_target_log_probs = torch.gather(ref_log_probs, dim=2, index=new_generated_tokens.unsqueeze(-1)).squeeze(-1)  # BLV -> BL

            # KL(m || m_ref) = [ m_ref(o|q) / m(o|q) ] - log [m_ref(o|q) / m(o|q)] - 1
            # KL(m || m_ref) = exp[ log m_ref(o|q) - log m(o|q) ] - log m_ref(o|q) + log m(o|q) - 1
            log_ratio = ref_target_log_probs - target_log_probs
            kl_div = (torch.exp(log_ratio) - log_ratio - 1).mean()  # BL -> scalar

            # Compute the final loss
            loss = reinforce_loss + args.grpo_kl_beta * kl_div
        else:
            loss = reinforce_loss

        total_loss.update(loss.item(), x.shape[0] * train_data.num_target_tokens)
        total_acc.update(accs['acc'], x.shape[0])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1
        train_bar.set_description(
            'Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'.format(ep, args.epochs, total_loss.get(),
             total_acc.get(percentage=True))
        )
        if wandb_log:
            output_dict = {"epoch": ep, "loss": float(loss), "acc": float(accs['acc']), "reinforce_loss": float(reinforce_loss),
                           "kl_div": float(kl_div) if kl_div is not None else kl_div}
            wandb.log({f"train/{k}": v for k, v in output_dict.items()})

    # evaluate the loss on train/val sets and write checkpoints
    if ep % args.eval_every == 0:
        # Generate sequences and check accuracies
        if args.eval_train:
            results = evaluate(model, train_loader, temperature=0.8, top_k=top_k, results=results, mode='train')
            results = evaluate_forced(model, train_loader, results=results, mode='train')

        results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test')
        results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')
        if wandb_log:
            wandb.log(results)

if args.save_checkpoints:
    print("Saving model checkpoint to file:", grpo_checkpoint_path)
    torch.save(model.state_dict(), grpo_checkpoint_path)
    print("Model checkpoint saved to file:", grpo_checkpoint_path)
