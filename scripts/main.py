#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('./src/')
from parser_ops import *
from zs_nac.denoising_func import *
from pathlib import Path
from datetime import datetime
import time


parser = get_parser()
args = parser.parse_args()


def main():
    results_path = args.results_dir
    noisy_img = np.load(Path(Path.cwd() / args.data_path))
    start_training_nac = time.time()
    denoised_img = nac(noisy_img, noise_std=args.noise_std, alpha=args.alpha, correction=False, batch_size=noisy_img.shape[-1], max_epochs=args.max_epochs, stop_training=args.stop_training,
                       training_part=False, ratio=0.3, device=args.device)
    end_inference_nac = time.time()
    total_nac_duration = end_inference_nac - start_training_nac
    speed_nac = np.prod(noisy_img.shape) / (total_nac_duration * 1000)
    print("zs_nac speed: {:.3f} kilopixels/s".format(speed_nac))

    Path(args.results_dir).mkdir(exist_ok=True)
    np.save(
        file=Path(results_path)
        / "{}_denoised_data".format(datetime.now().strftime("%Y%m%d%H%M%S")),
        arr=denoised_img,
    )
    print("output saved to ./results")


if __name__ == "__main__":
    main()
