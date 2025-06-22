from zs_nac.training import *
from zs_nac.load_data import *
from zs_nac.unet_model import *
from zs_nac.utils import *
import numpy as np
import time


def nac(
    noisy_data, mask=None, noise_std=None, alpha=1, correction=False, batch_size=1, max_epochs=2000, stop_training=50, training_part=False, ratio=0.3, device='cuda'
):
    """
    :param noisy_data: is the noisy input images. It can be complex or magnitude images.
    Input should be a numpy array of 2, 3 or 4 dimensions:
    2D: [height (inplane_dimension1), width (inplane_dimension2)]
    3D: [height (inplane_dimension1), width (inplane_dimension2), slice]
    4D: [height (inplane_dimension1), width (inplane_dimension2), slice, 4th dimension]
    If input data are real or imaginary data only, the internal function of mask creation will fail. In this condition the background mask or the std_noise of real or imaginary parts must be given as arguments.
    :param mask: is the background mask and has the same shape of noisy input images. Default: None, which means that it will be automatically created.
    :param noise_std: is the noise standard deviation of either 1- real or imaginary parts of complex input images or 2- of magnitude input images. Default: None, which means that the nosie_std is estimated based on the standard deviation of the background mask.
    :param alpha: α is the added noise factor in σ_B=ασ_A (check the paper). Default: 1
    :param correction: whether to apply a correction at inference stage. Default: False.
    :param batch_size: training batch size. Keep the number low if your computer doesn't have enough RAMs. Default:1
    :param max_epochs: maximum number of epochs. Default: 2000
    :param stop_training: The number of epochs needed to stop the training when validation error is not improving. Default: 50
    :param training_part: whether to train the model on a fewer number of slices (True) or the whole matrix (False).
     recommended when the number of slice or 4th dimension is too big resulting in a significantly slow denoising process. Default: False
    :param ratio: if training_part==True, ratio defines the ratio of slices/4th dimension to train on. Default: 0.3
    :param device: 'cuda' or 'cpu'. Default: 'cuda'

    :return: denoised image: ndarray

    Example: denoised_img = nac(noisy_img);
    """
    # ........................... Loading noisy images
    noisy_im, img_ndim, fourth_dim, n_chan = load_noisy_data(data=noisy_data)
    if training_part:
        noisy_img = noisy_im[int(noisy_im.shape[0]/2 - (ratio*noisy_im.shape[0]/2)):int(noisy_im.shape[0]/2 + (ratio*noisy_im.shape[0]/2))]
    else:
        noisy_img = noisy_im

    # ........................... Generating background mask
    if noise_std is None:
        if mask is None:
            background_threshold = 0.1
            mask_background = create_mask(noisy_img, threshold=background_threshold)
        else:
            if mask.ndim == 2:
                mask_background = mask[np.newaxis]
            elif mask.ndim == 3:
                mask_background = mask
            elif mask.ndim == 4:
                mask_background = np.reshape(
                    mask,
                    (
                        mask.shape[0],
                        mask.shape[1],
                        mask.shape[2] * mask.shape[3],
                    ),
                )
            else:
                print("Mask matrix size is neither 2D, 3D nor 4D. Please check your input dimension")
                exit()
    # ........................... Noise std estimation
        noise_std, noise_mean = estimate_background_noise(noisy_img, mask_background, n_chan=n_chan)
    else:
        noise_std = noise_std
        noise_mean = 0 # this is true for complex MRI data, not for magnitude todo make it more general
    # ........................... Initializing the network
    network = unet_res(n_channels=n_chan, n_classes=2)

    # ........................... Loading into device
    if torch.cuda.is_available() != 1:
        device = 'cpu'
        print('!!Warning!! CUDA is not available, training is run on CPU')
    else:
        device = device
    network = network.to(device)
    noisy_img = torch.from_numpy(noisy_img).type(torch.FloatTensor).to(device)

    # ........................... Training
    print("------ starting denoising ------")
    # learning rate
    lr = 0.01
    # number of epochs at which learning rate decays
    step_size = 10
    # factor by which learning rate decays
    factor = 0.9
    start_training_nac = time.time()
    trained_network_nac, doubly_noise_img, epoch_nac = training(
        network=network,
        gamma=alpha,
        noisy_img=noisy_img,
        noise_std=noise_std,
        noise_mean=noise_mean,
        n_chan=n_chan,
        lr=lr,
        batch_size=batch_size,
        step_size=step_size,
        factor=factor,
        max_epochs=max_epochs,
        stop_training=stop_training,
        device=device,
    )

    # ........................... Inference
    noisy_im = torch.from_numpy(noisy_im).type(torch.FloatTensor).to(device)
    denoised_img = trained_network_nac(noisy_im)
    if correction:
        denoised_img = (((1 + alpha ** 2) * denoised_img) - noisy_im) / (alpha ** 2)
    else:
        denoised_img = denoised_img
    pre_output = denoised_img.permute(2, 3, 0, 1).cpu().detach().numpy()
    if n_chan == 1:
        output = pre_output[:, :, :, 0]
    else:
        output = dualchannels2complex(pre_output)

    end_inference_nac = time.time()
    total_nac_duration = end_inference_nac - start_training_nac

    print("training completed in {:.3f} seconds on {}".format(total_nac_duration, device))

    if img_ndim == 2:
        final_output = np.squeeze(output)
    elif img_ndim == 3:
        final_output = output
    elif img_ndim == 4:
        final_output = np.reshape(
            output,
            (
                output.shape[0],
                output.shape[1],
                int(output.shape[2] / fourth_dim),
                fourth_dim,
            ),
        )
    else:
        print("Error: image dimensions are wrong, please check your input dimensions")
        exit()
    return final_output

