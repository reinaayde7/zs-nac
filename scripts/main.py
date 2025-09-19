#!/usr/bin/env python

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from parser_ops import *
from zs_nac.denoising_func import *
from datetime import datetime
import time

parser = get_parser()
args = parser.parse_args()

def main():
    results_path = args.results_dir
    noisy_img = np.load(os.path.join(project_root, args.data_path))
    start_training_nac = time.time()
    denoised_img = nac(noisy_img, alpha=args.alpha, correction=False, batch_size=noisy_img.shape[-1], max_epochs=args.max_epochs, stop_training=args.stop_training,
                       training_part=False, ratio=0.3, device=args.device)
    end_inference_nac = time.time()
    total_nac_duration = end_inference_nac - start_training_nac
    speed_nac = np.prod(noisy_img.shape) / (total_nac_duration * 1000)
    print("zs_nac speed: {:.3f} kilopixels/s".format(speed_nac))

    results_path = os.path.join(project_root, args.results_dir)
    os.makedirs(results_path, exist_ok=True)
    filename = "{}_denoised_data.npy".format(datetime.now().strftime("%Y%m%d%H%M%S"))
    np.save(os.path.join(results_path, filename), arr=denoised_img)
    print("output saved to /zs-nac/results/")


if __name__ == "__main__":
    main()
