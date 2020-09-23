import torch

from configs.args_ppo import get_args as get_base_args

from utils.cli import boolean_argument


def get_args(rest_args):
    base_parser = get_base_args()

    # --- AUP ---
    base_parser.add_argument("--use_aup", type=boolean_argument, default=True,
                             help='Whether to use the auxiliary utility preservation (default: False).')
    base_parser.add_argument("--num_q_aux", type=int, default=1,
                             help='The number of Q_aux functions to use.')
    base_parser.add_argument("--aup_coef_start", type=float, default=0.001,
                             help='Starting coefficient for AUP.')
    base_parser.add_argument("--aup_coef_end", type=float, default=0.01,
                             help='AUP coefficient will be linearly increased over time.')
    base_parser.add_argument("--q_aux_dir", type=str, default="q_aux_dir/coinrun",
                             help='Directory to load the Q_aux model weights from. If --num_q_aux=1 then this is the path to specific weights.')

    args = base_parser.parse_args(rest_args)
    args.cuda = torch.cuda.is_available()

    return args