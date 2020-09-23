import torch

from configs.args_ppo import get_args as get_base_args


def get_args(rest_args):

    base_parser = get_base_args()

    # --- GENERAL ---
    # train parameters
    base_parser.add_argument('--num_frames_r_aux', type=int, default=12e4,
                             help='number of frames to train for training the auxiliary reward function R_aux.')
    base_parser.add_argument('--num_frames_q_aux', type=int, default=12e5,
                             help='number of frames to train for training the Q-function Q_aux.')

    # --- CB-VAE ---
    base_parser.add_argument("--cb_vae_latent_dim", type=int, default=1,
                             help='The size of the latent dimension.')
    base_parser.add_argument("--cb_vae_epochs", type=int, default=100,
                             help='Number of epochs to train the CB-VAE.')
    base_parser.add_argument("--cb_vae_batch_size", type=int, default=2048,
                             help='The size of the latent dimension.')
    base_parser.add_argument("--cb_vae_learning_rate", type=float, default=5e-4,
                             help='The learning rate for the ADAM optimiser.')
    base_parser.add_argument("--cb_vae_num_samples", type=int, default=7,
                             help='The number of reconstruction samples to show from the trained CB-VAE.')

    # --- Q_aux ---
    base_parser.add_argument("--q_aux_path", type=str, default="q_aux_dir/coinrun/0.pt",
                             help='Directory to save the Q_aux model weights.')


    args = base_parser.parse_args(rest_args)
    args.cuda = torch.cuda.is_available()

    return args