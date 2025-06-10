import os
import argparse
from contextlib import nullcontext
from tqdm import tqdm
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

    batch_idx = 0
    if args.model == "multihead_gpt":
        assert not isinstance(test_loader, dict), type(test_loader)
        print("Model vocab size:", model.vocab_size)
        for x, y in test_loader:
            with ctx:
                outputs = model(x, return_all_predictions=True)
            assert isinstance(outputs, tuple) or isinstance(outputs, list), type(outputs)
            print(f"x: {x.shape} / y: {y.shape}")
            print(len(outputs), [x.shape for x in outputs])  # BLV
            assert all([x.shape[-1] == model.vocab_size for x in outputs]), [x.shape for x in outputs]
            print(f"data / x: {x[batch_idx]} / y: {y[batch_idx]}")
            print(f"data / output 0: {outputs[0][batch_idx].argmax(dim=-1)}")

            # Visualize the outputs from the different heads
            bs = len(y)
            ignore_idx = -1
            unmasked_tokens = y != ignore_idx
            first_pos = unmasked_tokens.float().argmax(dim=1)    # (B,) – returns the first index in the case of a tie i.e., first index of one

            # Crop out just the target sequence
            target_seq = torch.stack([y[b, first_pos[b]:] for b in range(bs)], dim=0)[batch_idx]  # B(L-1) -> BL'
            print("target seq:", target_seq)

            for head_idx, head_size in enumerate(model.head_sizes):
                seq_rep = outputs[head_idx][torch.arange(bs, device=outputs[0].device), first_pos]  #  BLD -> BD (first unmasked token)
                print("Sequence rep:", seq_rep.shape)
                predicted_prob = seq_rep[batch_idx]
                if head_size == "separate_bow":
                    predicted_prob = torch.sigmoid(predicted_prob)
                sorted_idx = torch.argsort(predicted_prob, descending=True, stable=True)
                token_prob_map = {int(idx): float(predicted_prob[idx]) for idx in sorted_idx}
                print(f"head size: {head_size} / prob: {predicted_prob} / sorted dict: {token_prob_map}")

            break
    print("Terminating script...")
    exit()

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures', entity=wandb_entity, config=args.__dict__,)
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
