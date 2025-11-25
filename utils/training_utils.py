import math
import torch


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.val = 0

    def update(self, val, num):
        self.val += val * num
        self.num += num

    def get(self, percentage=False):
        val = self.val / self.num * 100 if percentage else self.val / self.num
        return val


def accuracy(logits, targets):
    num_prefix_tokens = targets[0].eq(-1).sum().item()
    num_target_tokens = targets.shape[1] - num_prefix_tokens
    targets = targets[:, num_prefix_tokens:]
    logits = logits[:, num_prefix_tokens:, :]
    correct = torch.argmax(logits, dim=-1).eq(targets).to(torch.float)
    seq_correct = torch.sum(correct, dim=1).eq(num_target_tokens).float()
    acc = torch.mean(seq_correct)
    per_token_acc = correct.mean(dim=0)

    return acc, per_token_acc


def get_run_name(args):
    name = args.dataset + "_" + args.model
    if args.n_layer != 6:
        name += f"_l{args.n_layer}"
    if args.n_head != 6:
        name += f"_h{args.n_head}"
    if args.n_embd != 384:
        name += f"_e{args.n_embd}"
    if args.lr != 5e-4:
        name += f"_lr{args.lr}"
    if args.weight_decay != 1e-2:
        name += f"_wd{args.weight_decay}"
    if args.clip_grad_norm is not None:
        name += f"_clip{args.clip_grad_norm}"
    if args.batch_size != 256:
        name += f"_bs{args.batch_size}"
    if args.epochs != 100:
        name += f"_ep{args.epochs}"

    if args.dataset == 'graph':
        waypoint_str = f"_waypoint_len_{args.waypoint_len}" if args.waypoint_len is not None else ""
        name += '_deg' + str(args.deg) + '_path_' + str(args.path_len) + 'num_nodes_' + str(args.num_nodes) + waypoint_str + \
                '_ntrain_' + str(args.n_train) + '_teacherless_' + str(args.teacherless) + '_reverse_' + str(args.reverse)
        if args.model == "multihead_gpt":
            name += f"_heads_{args.prediction_head_sizes}_weights_{args.prediction_head_weights}"
            if args.multihead_boundary_condition is not None:
                name += f"_boundary_{args.multihead_boundary_condition}"
        elif args.model == "next_lat_gpt":
            name += "_v2"
            if args.pred_horizon != args.path_len - 2:
                name += f"_horizon_{args.pred_horizon}"
            if args.next_lat_lambda != 1.0 or args.kl_lambda != 1.0:
                name += "_lambda"
                if args.next_lat_lambda != 1.0:
                    name += f"_nl_{args.next_lat_lambda}"
                if args.kl_lambda != 1.0:
                    name += f"_kl_{args.kl_lambda}"
    elif args.dataset == 'chess':
        assert not args.model == "multihead_gpt", args.model
        assert args.waypoint_len is None
        name += '_mate_in_' + str(args.mate_in) + '_ntrain_' + str(args.n_train) + '_unrolled_' + str(args.unrolled) + \
                '_teacherless_' + str(args.teacherless)

    return name
