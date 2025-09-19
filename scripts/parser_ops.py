import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Zero-Shot Self-Supervised denoising")

    parser.add_argument(
        "-p",
        "--data_path",
        default="./data/noisy_braint2-1654_128x128x14_snr20.npy",
        type=str,
        help="The path to the numpy array input noisy images",
    )

    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        default="./results",
        help="The base directory to which to write evaluation results",
    )

    parser.add_argument(
        "-c", "--n_chan", type=int, default=2, help="number of channels"
    )

    parser.add_argument(
        "-n",
        "--noise_std",
        type=float,
        default=None,
        help="the estimated noise standard deviation. if None the noise standard deviation is estimated automatically by an internal function",
    )

    parser.add_argument(
        "-a",
        "--alpha",
        type=int,
        default=1,
        help="the level of noise to be added to the acquired image (alpha*noise_acquired_image)."
        " The higher is gamma the more smooth are the denoised images",
    )

    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        default=2000,
        help="the maximum number of epochs",
    )

    parser.add_argument(
        "-s",
        "--stop_training",
        type=int,
        default=20,
        help="stop training if a new lowest validation loss hasn't been achieved in xx epochs",
    )

    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="device: cpu or cuda"
    )

    return parser