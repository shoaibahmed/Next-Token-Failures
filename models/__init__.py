from models.gpt import GPT
from models.multihead_gpt import MultiheadGPT
from models.next_lat_gpt import NextLatGPT
from models.pythia import Pythia
from models.config import GPTConfig


def get_model(args):
    if args.model == 'gpt':
        config = GPTConfig(n_layers=args.n_layer, n_heads=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                           bias=True, vocab_size=args.vocab_size, dropout=0, use_flash=args.use_flash,
                           teacherless_token=args.teacherless_token)
        model = GPT(config)
    elif args.model == 'multihead_gpt':
        head_sizes = [x if x in ["bow", "separate_bow"] else int(x) for x in args.prediction_head_sizes.split(",")]  # keep bow intact
        head_weights = [float(x) for x in args.prediction_head_weights.split(",")]
        config = GPTConfig(n_layers=args.n_layer, n_heads=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                           bias=True, vocab_size=args.vocab_size, dropout=0, use_flash=args.use_flash,
                           teacherless_token=args.teacherless_token, head_sizes=head_sizes, head_weights=head_weights,
                           boundary_condition=args.multihead_boundary_condition)
        model = MultiheadGPT(config)
    elif args.model == 'next_lat_gpt':
        next_latent_pred_layers = [int(x) for x in args.next_latent_pred_layers.split(",")]
        config = GPTConfig(n_layers=args.n_layer, n_heads=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                           bias=True, vocab_size=args.vocab_size, dropout=0, use_flash=args.use_flash,
                           teacherless_token=args.teacherless_token, pred_horizon=args.pred_horizon,
                           next_lat_lambda=args.next_lat_lambda, kl_lambda=args.kl_lambda,
                           num_prev_latents=args.num_prev_latents,
                           next_latent_pred_layers=next_latent_pred_layers,
                           use_last_lat_res_conn=args.use_last_lat_res_conn)
        model = NextLatGPT(config)
    elif args.model.startswith('gpt2'):
        model = GPT.from_pretrained(args.model, teacherless_token=args.teacherless_token)
        if args.block_size < 1024:
            model.crop_block_size(args.block_size)

    elif args.model.startswith('pythia'):
        model = Pythia.from_pretrained(args.model, teacherless_token=args.teacherless_token)

    return model
