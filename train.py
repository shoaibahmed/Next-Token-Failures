import os
import argparse
from contextlib import nullcontext
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import wandb

from tokenizing import get_tokenizer
from utils.training_utils import get_lr, get_run_name, AverageMeter
from data import get_dataset
from evaluate import evaluate, evaluate_forced
from models import get_model


# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction")
# Data
parser.add_argument(
        "--model", type=str, default='gpt', help="Learning rate",
    )
parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of layers",
    )
parser.add_argument(
        "--n_embd", type=int, default=384, help="Embedding size",
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
        "--weight_decay", type=float, default=1e-2, help="Strength of weight decay",
    )
parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs",
    )
parser.add_argument(
        "--save_every", type=int, default=5000, help="Interval (in steps) at which to save model",
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
        "--eval_every", type=int, default=5000, help="Interval (in steps) to evaluate the model on test",
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
        "--clip-grad-norm", type=float, default=None, help="Grad norm clipping threshold",
    )

args = parser.parse_args()
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
if args.clip_grad_norm is not None and args.clip_grad_norm <= 0.:
    args.clip_grad_norm = None

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
args.compile = True if device == 'cuda' else False
args.use_flash = True if device == 'cuda' else False
warmup_iters = 100
min_lr = 1e-5

args.pred_horizon = args.path_len - 2  # based on the next-lat paper (input seq: L-1; latents w/ target: L-2)
run_name = get_run_name(args)
run_name += f"_clip_grad_{args.clip_grad_norm}" if args.clip_grad_norm is not None else ""
run_name += f"_n_layers_{args.n_layer}" if args.n_layer != 6 else ""
run_name += f"_n_embed_{args.n_embd}" if args.n_embd != 384 else ""
run_name += f"_n_head_{args.n_head}" if args.n_head != 6 else ""
path = './checkpoints/' + run_name + '.pt'

# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)
train_data, test_data = get_dataset(args, tokenizer, device)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
print(f"train dataset: {len(train_data)} / train loader: {len(train_loader)}")
if isinstance(test_data, dict):
    test_loader = {}
    for k in test_data.keys():
        test_loader[k] = DataLoader(test_data[k], batch_size=args.batch_size, shuffle=False)
        print(f"test dataset {k}: {len(test_data[k])} / test loader: {len(test_loader[k])}")
else:
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    print(f"test dataset: {len(test_data)} / test loader: {len(test_loader)}")

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
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# Define checkpoint path
checkpoint_dir = "./checkpoints/"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    print("Checkpoint directory created:", checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}.pth")
print("Using checkpoint path:", checkpoint_path)

if args.save_checkpoints and os.path.exists(checkpoint_path):
    print("Model checkpoint already exists:", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    results = {}
    if isinstance(test_loader, dict):
        for k in test_loader.keys():
            if k not in results:
                results[k] = {}
            results[k] = evaluate(model, test_loader[k], temperature=0.8, ctx=ctx, top_k=top_k, results=results[k], mode=f'test_{k}')
            results[k] = evaluate_forced(model, test_loader[k], ctx=ctx, results=results[k], mode=f'test_{k}')
            print(results[k])
    else:
        results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test')
        results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')
        print(results)

    if args.model == "multihead_gpt":
        assert not isinstance(test_loader, dict), type(test_loader)
        print("Model vocab size:", model.vocab_size)

        # Containers to accumulate ranks across the entire test set
        rank_histogram = {hs: defaultdict(int) for hs in model.head_sizes}
        total_counts = {hs: 0 for hs in model.head_sizes}
        coverage_lists = {hs: [] for hs in model.head_sizes}

        ignore_idx = -1
        verbose = False
        for x, y in tqdm(test_loader, desc="Ranking eval"):
            # Get logits for all heads (full sequence length – teacher forcing)
            with ctx:
                outputs = model(x, return_all_predictions=True)
            assert isinstance(outputs, (tuple, list)), type(outputs)

            # Identify the start position of the target sequence for every sample
            bs = len(y)
            unmasked_tokens = y != ignore_idx
            first_pos = unmasked_tokens.float().argmax(dim=1)  # (B,)

            for b in range(bs):
                tgt_tokens = y[b, first_pos[b]:]  # 1-D tensor of target tokens for sample b
                if verbose and b == 0:
                    print("~"*10)
                    print("Target tokens:", tgt_tokens)

                for head_idx, head_size in enumerate(model.head_sizes):
                    # Predicted probability distribution at the first target position
                    seq_rep = outputs[head_idx][b, first_pos[b]]  # (V,)

                    # Sort logits once per sample-head to speed things up
                    sorted_vals, sorted_idx = torch.sort(seq_rep, descending=True, stable=True)

                    # Build a rank map: token_id -> rank (1-based)
                    rank_map = torch.empty_like(seq_rep, dtype=torch.long)
                    rank_map[sorted_idx] = torch.arange(1, len(sorted_idx) + 1, device=seq_rep.device)

                    accepted_cnt = 0
                    for token in tgt_tokens:
                        token_int = int(token.item())
                        rank = int(rank_map[token_int].item())
                        rank_histogram[head_size][rank] += 1
                        total_counts[head_size] += 1
                        if (head_size == "separate_bow" and rank <= len(tgt_tokens)) or (head_size != "separate_bow" and rank <= head_size):
                            accepted_cnt += 1  # token visible inside prediction window

                    # Record per-sequence acceptance percentage
                    if len(tgt_tokens) > 0:
                        coverage_lists[head_size].append(accepted_cnt / len(tgt_tokens))

                    if verbose and b == 0:
                        sorted_dict = {int(k): round(float(v), 2) for k, v in zip(sorted_idx, sorted_vals)}
                        print(f"Head size {head_size} raw logits:", seq_rep)
                        print(f"Head size {head_size} sorted dict:", sorted_dict)
                        print("-"*10)

        # ---------------------------------------------------------------------
        # Summarise results
        # ---------------------------------------------------------------------
        import json, datetime, os

        # Save raw histogram data for later plotting
        rank_dump = {
            "rank_histogram": {int(k): {int(r): int(c) for r, c in v.items()} for k, v in rank_histogram.items()},
            "total_counts": {int(k): int(v) for k, v in total_counts.items()},
            "coverage_lists": {int(k): v for k, v in coverage_lists.items()},
            "run_name": run_name,
            "timestamp": datetime.datetime.now().isoformat()
        }
        dump_dir = "rank_eval_outputs"
        os.makedirs(dump_dir, exist_ok=True)
        dump_file = os.path.join(dump_dir, f"{run_name}_rank_hist.json")
        with open(dump_file, "w") as f:
            json.dump(rank_dump, f)
        print(f"Saved rank-histogram data to {dump_file}")

        for head_size, hist in rank_histogram.items():
            if total_counts[head_size] == 0:
                continue

            # Compute average rank and top-k coverage (k = 1, 5, 10)
            avg_rank = sum(r * c for r, c in hist.items()) / total_counts[head_size]
            top1 = sum(c for r, c in hist.items() if r <= 1) / total_counts[head_size] * 100
            top5 = sum(c for r, c in hist.items() if r <= 5) / total_counts[head_size] * 100
            top10 = sum(c for r, c in hist.items() if r <= 10) / total_counts[head_size] * 100

            # Mean acceptance percentage per sequence
            if coverage_lists[head_size]:
                mean_accept = sum(coverage_lists[head_size]) / len(coverage_lists[head_size]) * 100
            else:
                mean_accept = float('nan')

            print("=" * 80)
            print(f"Head size: {head_size}")
            print(f"  Avg. rank of target tokens: {avg_rank:.2f}")
            print(f"  % of target tokens in top 1 logits:  {top1:.2f}%")
            print(f"  % of target tokens in top 5 logits:  {top5:.2f}%")
            print(f"  % of target tokens in top 10 logits: {top10:.2f}%")
            print(f"  Avg. % of target tokens present per sequence: {mean_accept:.2f}%")

    print("Terminating script...")
    exit()

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures-latest', entity=wandb_entity, config=args.__dict__,)
    wandb.run.name = run_name

results = {}
num_iters = 0

for ep in range(args.epochs):
    if ep % args.save_every == 0 and ep > 0:
        torch.save(model.state_dict(), path + "_epoch_" + str(ep))

    train_bar = tqdm(train_loader)
    total_loss, total_acc = AverageMeter(), AverageMeter()

    for x, y in train_bar:
        # determine and set the learning rate for this iteration
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            logits, loss, accs = model(x, y)

        total_loss.update(loss.item(), x.shape[0] * train_data.num_target_tokens)
        total_acc.update(accs['acc'], x.shape[0])
        scaler.scale(loss).backward()
        if args.clip_grad_norm is not None:
            # https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
            scaler.unscale_(optimizer)  # get the gradients in the original scale
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1
        train_bar.set_description(
            'Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'.format(ep, args.epochs, total_loss.get(),
             total_acc.get(percentage=True))
        )
        if wandb_log:
            wandb.log({"train/epoch": ep, "train/loss": total_loss.get(), "train/acc": total_acc.get(percentage=True)})

    # evaluate the loss on train/val sets and write checkpoints
    if ep % args.eval_every == 0:
        # Generate sequences and check accuracies
        if args.eval_train:
            results = evaluate(model, train_loader, temperature=0.8, top_k=top_k, results=results, mode='train')
            results = evaluate_forced(model, train_loader, results=results, mode='train')

        if isinstance(test_loader, dict):
            for k in test_loader.keys():
                if k not in results:
                    results[k] = {}
                results[k] = evaluate(model, test_loader[k], temperature=0.8, ctx=ctx, top_k=top_k, results=results[k], mode=f'test_{k}')
                results[k] = evaluate_forced(model, test_loader[k], ctx=ctx, results=results[k], mode=f'test_{k}')
                if wandb_log:
                    wandb.log(results[k])
        else:
            results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test')
            results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')
            if wandb_log:
                wandb.log(results)

if args.save_checkpoints:
    print("Saving model checkpoint to file:", checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)
    print("Model checkpoint saved to file:", checkpoint_path)
