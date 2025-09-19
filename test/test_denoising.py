import pathlib
from zs_nac.denoising_func import *
import skimage
import os
import re
import json
import time

'''
Todo: Test the pipeline with 4D data 
P.S: Pytest run with numpy version < 2
'''

def test_cuda_2channels() -> None:
    data_path = "../data/noisy_braint2-1654_128x128x14_snr20.npy"
    noisy_data = np.load(pathlib.Path(data_path))
    n_chan = 2
    alpha = 1
    max_epochs = 2000
    stop_training = 50
    device = "cuda"
    a = time.time()
    denoised_img = nac(noisy_data, alpha=alpha, correction=False, batch_size=noisy_data.shape[-1], max_epochs=max_epochs, stop_training=stop_training, device=device)
    b = time.time()
    total_nac_duration = b - a
    print(total_nac_duration)
    ref_img = np.load(pathlib.Path("../data/ref_braint2-1654_128x128x14.npy"))
    ref_img_norm = normalize(abs(ref_img))
    background_threshold = 0.1
    # I made ref img 4D to be compatible with create_mask function
    ref_img_norm = np.moveaxis(ref_img_norm, 2, 0)
    ref_img_norm = ref_img_norm[:, None]
    mask_background = 1 - create_mask(ref_img_norm, threshold=background_threshold)
    ref_img_norm_masked = ref_img_norm[:, 0] * mask_background
    ref_img_norm_masked = np.moveaxis(ref_img_norm_masked, 0, 2)
    mask_background = np.moveaxis(mask_background, 0, 2)
    denoised_img_norm_masked = normalize(abs(denoised_img)) * mask_background
    ssim = np.empty(denoised_img.shape[-1])
    psnr = np.empty(denoised_img.shape[-1])
    for i in range(denoised_img.shape[-1]):
        ssim[i] = skimage.metrics.structural_similarity(
            ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i], data_range=1
        )
        psnr[i] = myPSNR(ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i])
    file_name = find_file(
        "test_{}_device{}_alpha{}_nchan{}.json".format(
            data_path[8:-4], device, alpha, n_chan
        ),
        pathlib.Path("../test"),
    )[0]
    f = json.load(open("../test/{}".format(file_name)))

    for i in zip(ssim, [val*0.9 for val in f[0].get("results").get("ssim")]):
        assert i[1] < i[0]
    for j in zip(psnr, [val*0.9 for val in f[0].get("results").get("psnr")]):
        assert j[1] < j[0]
    # I stopped asserting the time on cpu and cuda because I am running the code from different devices and it's likely to generate an error
    # assert ( total_nac_duration < f[0].get("results").get("time (s)")*1.5)

def test_cpu_2channels() -> None:
    data_path = "../data/noisy_braint2-1654_128x128x14_snr20.npy"
    noisy_data = np.load(pathlib.Path(data_path))
    n_chan = 2
    alpha = 1
    max_epochs = 2000
    stop_training = 50
    device = "cpu"
    a = time.time()
    denoised_img = nac(noisy_data, alpha=alpha, correction=False, batch_size=noisy_data.shape[-1], max_epochs=max_epochs, stop_training=stop_training, device=device)
    b = time.time()
    total_nac_duration = b - a
    print(total_nac_duration)
    ref_img = np.load(pathlib.Path("../data/ref_braint2-1654_128x128x14.npy"))
    ref_img_norm = normalize(abs(ref_img))
    background_threshold = 0.1
    ref_img_norm = np.moveaxis(ref_img_norm, 2, 0)
    ref_img_norm = ref_img_norm[:, None]
    mask_background = 1 - create_mask(ref_img_norm, threshold=background_threshold)
    ref_img_norm_masked = ref_img_norm[:, 0] * mask_background
    ref_img_norm_masked = np.moveaxis(ref_img_norm_masked, 0, 2)
    mask_background = np.moveaxis(mask_background, 0, 2)
    denoised_img_norm_masked = normalize(abs(denoised_img)) * mask_background
    ssim = np.empty(denoised_img.shape[-1])
    psnr = np.empty(denoised_img.shape[-1])
    for i in range(denoised_img.shape[-1]):
        ssim[i] = skimage.metrics.structural_similarity(
            ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i], data_range=1
        )
        psnr[i] = myPSNR(ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i])
    file_name = find_file(
        "test_{}_device{}_alpha{}_nchan{}.json".format(
            data_path[8:-4], device, alpha, n_chan
        ),
        pathlib.Path("../test"),
    )[0]
    f = json.load(open("../test/{}".format(file_name)))
    for i in zip(ssim, [val * 0.9 for val in f[0].get("results").get("ssim")]):
        assert i[1] < i[0]
    for j in zip(psnr, [val * 0.9 for val in f[0].get("results").get("psnr")]):
        assert j[1] < j[0]
    # I stopped asserting the time on cpu and cuda because I am running the code from different devices and it's likely to generate an error
    # assert ( total_nac_duration < f[0].get("results").get("time (s)")*1.5)

def test_cuda_1channel() -> None:
    data_path = "../data/noisy_braint2-1654_128x128x14_snr20.npy"
    noisy_data = np.load(pathlib.Path(data_path))
    n_chan = 1
    alpha = 1
    max_epochs = 2000
    stop_training = 50
    device = "cuda"
    a = time.time()
    denoised_img = nac(noisy_data, alpha=alpha, correction=False, batch_size=noisy_data.shape[-1], max_epochs=max_epochs, stop_training=stop_training, device=device)
    b = time.time()
    total_nac_duration = b - a
    print(total_nac_duration)
    ref_img = np.load(pathlib.Path("../data/ref_braint2-1654_128x128x14.npy"))
    ref_img_norm = normalize(abs(ref_img))
    background_threshold = 0.1
    ref_img_norm = np.moveaxis(ref_img_norm, 2, 0)
    ref_img_norm = ref_img_norm[:, None]
    mask_background = 1 - create_mask(ref_img_norm, threshold=background_threshold)
    ref_img_norm_masked = ref_img_norm[:, 0] * mask_background
    ref_img_norm_masked = np.moveaxis(ref_img_norm_masked, 0, 2)
    mask_background = np.moveaxis(mask_background, 0, 2)
    denoised_img_norm_masked = normalize(abs(denoised_img)) * mask_background
    ssim = np.empty(denoised_img.shape[-1])
    psnr = np.empty(denoised_img.shape[-1])
    for i in range(denoised_img.shape[-1]):
        ssim[i] = skimage.metrics.structural_similarity(
            ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i], data_range=1
        )
        psnr[i] = myPSNR(ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i])
    file_name = find_file(
        "test_{}_device{}_alpha{}_nchan{}.json".format(
            data_path[8:-4], device, alpha, n_chan
        ),
        pathlib.Path("../test"),
    )[0]
    f = json.load(open("../test/{}".format(file_name)))

    for i in zip(ssim,  [val*0.9 for val in f[0].get("results").get("ssim")]):
        assert i[1] < i[0]
    for j in zip(psnr,  [val*0.9 for val in f[0].get("results").get("psnr")]):
        assert j[1] < j[0]
    # I stopped asserting the time on cpu and cuda because I am running the code from different devices and it's likely to generate an error
    # assert (total_nac_duration < f[0].get("results").get("time (s)")*1.5)

def test_cpu_1channel() -> None:
    data_path = "../data/noisy_braint2-1654_128x128x14_snr20.npy"
    noisy_data = np.load(pathlib.Path(data_path))
    n_chan = 1
    alpha = 1
    max_epochs = 2000
    stop_training = 50
    device = "cpu"
    a = time.time()
    denoised_img = nac(noisy_data, alpha=alpha, correction=False, batch_size=noisy_data.shape[-1], max_epochs=max_epochs, stop_training=stop_training, device=device)
    b = time.time()
    total_nac_duration = b - a
    print(total_nac_duration)
    ref_img = np.load(pathlib.Path("../data/ref_braint2-1654_128x128x14.npy"))
    ref_img_norm = normalize(abs(ref_img))
    background_threshold = 0.1
    ref_img_norm = np.moveaxis(ref_img_norm, 2, 0)
    ref_img_norm = ref_img_norm[:, None]
    mask_background = 1 - create_mask(ref_img_norm, threshold=background_threshold)
    ref_img_norm_masked = ref_img_norm[:, 0] * mask_background
    ref_img_norm_masked = np.moveaxis(ref_img_norm_masked, 0, 2)
    mask_background = np.moveaxis(mask_background, 0, 2)
    denoised_img_norm_masked = normalize(abs(denoised_img)) * mask_background
    ssim = np.empty(denoised_img.shape[-1])
    psnr = np.empty(denoised_img.shape[-1])
    for i in range(denoised_img.shape[-1]):
        ssim[i] = skimage.metrics.structural_similarity(
            ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i], data_range=1
        )
        psnr[i] = myPSNR(ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i])
    file_name = find_file(
        "test_{}_device{}_alpha{}_nchan{}.json".format(
            data_path[8:-4], device, alpha, n_chan
        ),
        pathlib.Path("../test"),
    )[0]
    f = json.load(open("../test/{}".format(file_name)))

    for i in zip(ssim,  [val*0.9 for val in f[0].get("results").get("ssim")]):
        assert i[1] < i[0]
    for j in zip(psnr,  [val*0.9 for val in f[0].get("results").get("psnr")]):
        assert j[1] < j[0]
    # I stopped asserting the time on cpu and cuda because I am running the code from different devices and it's likely to generate an error
    # assert (total_nac_duration < f[0].get("results").get("time (s)")*1.5)

# def test_nan_slices():
#     data_path = "../data/gz11.npy"
#     noisy_data = np.load(pathlib.Path(data_path))
#     device = "cuda"
#     denoised_img = nac(noisy_data, correction=False, batch_size=16, device=device)
#     assert not np.isnan(denoised_img).any()

def test_loaded_mask():
    data_path = "../data/noisy_braint2-1654_128x128x14_snr20.npy"
    noisy_data = np.load(pathlib.Path(data_path))
    n_chan = 2
    alpha = 1
    mask_path = "../data/background_mask.npy"
    mask = np.load(pathlib.Path(mask_path))
    device = "cuda"
    a = time.time()
    denoised_img = nac(noisy_data, mask=mask, correction=False, batch_size=noisy_data.shape[-1], device=device)
    b = time.time()
    total_nac_duration = b - a
    ref_img = np.load(pathlib.Path("../data/ref_braint2-1654_128x128x14.npy"))
    ref_img_norm = normalize(abs(ref_img))
    background_threshold = 0.1
    ref_img_norm = np.moveaxis(ref_img_norm, 2, 0)
    ref_img_norm = ref_img_norm[:, None]
    mask_background = 1 - create_mask(ref_img_norm, threshold=background_threshold)
    ref_img_norm_masked = ref_img_norm[:, 0] * mask_background
    ref_img_norm_masked = np.moveaxis(ref_img_norm_masked, 0, 2)
    mask_background = np.moveaxis(mask_background, 0, 2)
    denoised_img_norm_masked = normalize(abs(denoised_img)) * mask_background
    ssim = np.empty(denoised_img.shape[-1])
    psnr = np.empty(denoised_img.shape[-1])
    for i in range(denoised_img.shape[-1]):
        ssim[i] = skimage.metrics.structural_similarity(
            ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i], data_range=1
        )
        psnr[i] = myPSNR(ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i])
    file_name = find_file(
                "test_{}_device{}_alpha{}_nchan{}.json".format(
                    data_path[8:-4], device, alpha, n_chan
                ),
        pathlib.Path("../test"),
    )[0]
    f = json.load(open("../test/{}".format(file_name)))

    for i in zip(ssim,[val*0.9 for val in f[0].get("results").get("ssim")]):
        assert i[1] < i[0]
    for j in zip(psnr,[val*0.9 for val in f[0].get("results").get("psnr")]):
        assert j[1] < j[0]
    # I stopped asserting the time on cpu and cuda because I am running the code from different devices and it's likely to generate an error
    # assert (total_nac_duration < f[0].get("results").get("time (s)")*1.5)

def test_noise_std():
    data_path = "../data/noisy_braint2-1654_128x128x14_snr20.npy"
    noisy_data = np.load(pathlib.Path(data_path))
    n_chan = 2
    alpha = 1
    noise_std = 37
    device = "cuda"
    a = time.time()
    denoised_img = nac(noisy_data, noise_std=noise_std, correction=False, batch_size=noisy_data.shape[-1], device=device)
    b = time.time()
    total_nac_duration = b - a
    ref_img = np.load(pathlib.Path("../data/ref_braint2-1654_128x128x14.npy"))
    ref_img_norm = normalize(abs(ref_img))
    background_threshold = 0.1
    ref_img_norm = np.moveaxis(ref_img_norm, 2, 0)
    ref_img_norm = ref_img_norm[:, None]
    mask_background = 1 - create_mask(ref_img_norm, threshold=background_threshold)
    ref_img_norm_masked = ref_img_norm[:, 0] * mask_background
    ref_img_norm_masked = np.moveaxis(ref_img_norm_masked, 0, 2)
    mask_background = np.moveaxis(mask_background, 0, 2)
    denoised_img_norm_masked = normalize(abs(denoised_img)) * mask_background
    ssim = np.empty(denoised_img.shape[-1])
    psnr = np.empty(denoised_img.shape[-1])
    for i in range(denoised_img.shape[-1]):
        ssim[i] = skimage.metrics.structural_similarity(
            ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i], data_range=1
        )
        psnr[i] = myPSNR(ref_img_norm_masked[..., i], denoised_img_norm_masked[..., i])
    file_name = find_file(
        "test_{}_device{}_alpha{}_nchan{}.json".format(
            data_path[8:-4], device, alpha, n_chan
        ),
        pathlib.Path("../test"),
    )[0]
    f = json.load(open("../test/{}".format(file_name)))

    for i in zip(ssim, [val*0.9 for val in f[0].get("results").get("ssim")]):
        assert i[1] < i[0]
    for j in zip(psnr, [val*0.9 for val in f[0].get("results").get("psnr")]):
        assert j[1] < j[0]
    # I stopped asserting the time on cpu and cuda because I am running the code from different devices and it's likely to generate an error
    # assert (total_nac_duration < f[0].get("results").get("time (s)")*1.5)


def find_file(pattern, path):
    result = []
    for _, _, files in os.walk(path):
        for name in files:
            a = re.findall(pattern, name)
            if a != []:
                result.append(name)
    if len(result) != 1:
        print("Error: only one JSON file is accepted for comparison")
        exit()
    else:
        return result

# Todo: Add a test for images with single slice

# # Creating Json file for metrics comparison
# import json
# results = []
# item = {
#     'data': args.data_path,
#     'device': args.device,
#     'number_of_channel': args.n_chan,
#     'alpha': args.alpha,
#     'max_epochs': args.max_epochs,
#     'stop_training': args.stop_training,
#     'results':
#     {
#         'ssim': [i for i in ssim],
#         'psnr': [i for i in psnr],
#         'time (s)': total_nac_duration,
#     }
#     }
# results.append(item)
# with open(Path('./test/test_{}_device{}_alpha{}_nchan{}.json'.format(
#         args.data_path[8:-4], args.device, args.alpha, args.n_chan)), 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)

# # display image
# slice_idx = 7
# fig, ax = plt.subplots(3, 3)
# ax[0, 0].imshow(normalize(abs(noisy_img[..., slice_idx])))
# ax[0, 1].imshow(normalize(abs(denoised_img[..., slice_idx])))
# ax[0, 1].set_title('PSNR{} \n SSIM{}'.format(psnr[slice_idx], ssim[slice_idx]))
# ax[0, 2].imshow(normalize(abs(ref_img[..., slice_idx])))
# ax[1, 0].imshow(abs((normalize(noisy_img)-normalize(ref_img))[..., slice_idx]), cmap='bwr')
# ax[1, 1].imshow(abs((normalize(denoised_img)-normalize(ref_img))[..., slice_idx]), cmap='bwr')
# ax[1, 2].imshow(abs((normalize(ref_img)-normalize(ref_img))[..., slice_idx]), cmap='bwr')
# ax[2, 0].imshow(np.angle(noisy_img[..., slice_idx]))
# ax[2, 1].imshow(np.angle(denoised_img[..., slice_idx]))
# ax[2, 2].imshow(np.angle(ref_img[..., slice_idx]))
# plt.show()
