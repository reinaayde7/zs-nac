from zs_nac.training import *
from zs_nac.load_data import *
from zs_nac.unet_model import *
from zs_nac.utils import *
import numpy as np
import time


def nac(
    noisy_data, mask=None, alpha=1, correction=False, batch_size=1, max_epochs=2000, stop_training=50, training_part=False, ratio=0.3, inplane=False, device='cuda'
):
    """
    :param noisy_data: ndarray. noisy_data is the noisy image.
        Image shape can be 2D, 3D or 4D:
        2D: [height/inplane_dim1, width/inplane_dim2]:
        3D: [height/inplane_dim1, width/inplane_dim2, slice/phase2]
        4D: [height/inplane_dim1, width/inplane_dim2, slice/phase2, 4th dim]
    :param mask: bool ndarray. mask is the background mask and has the same shape of the noisy input images. Default: None, which means that it will be automatically created based on an internal function.
    :param alpha: float. α is the added noise factor in σ_B=ασ_A (check the paper). Default: 1
    :param correction: bool. whether to apply the correction after inference (check the paper). Default: False
    :param batch_size: int. training batch size. Keep the number low if your computer doesn't have enough RAMs. Default:1
    :param max_epochs: integer. max_epochs is the maximum number of epochs. Default: 2000
    :param stop_training: integer. The training stops when validation error is not improving after 'stop_training' epochs. Default: 50
    :param training_part: bool. whether to train the model on a fewer number of slices (True) or the whole batch (False).
     recommended when the number of slice or 4th dimension is too big resulting in a significantly slow denoising process. Default: False
    :param ratio: float. when training part==True, the ratio defines the ratio of training slice/4th dimension. Default: 0.3
    :param device: 'cuda' or 'cpu'. Default: 'cpu'
    :return: denoised image: ndarray


    Example: denoised_img = nac(noisy_img, alpha=1, n_chan=2);
    """
    # ........................... Loading noisy images
    noisy_im, img_ndim, fourth_dim, n_chan = load_noisy_data(data=noisy_data)
    if training_part and ratio != 1.0:
        if inplane == False:
            noisy_img = noisy_im[int(noisy_im.shape[0]/2 - (ratio*noisy_im.shape[0]/2)):int(noisy_im.shape[0]/2 + (ratio*noisy_im.shape[0]/2))]
        else:
            noisy_img = noisy_im[..., int(noisy_im.shape[2]/2 - (noisy_im.shape[2]/(2/(ratio)))):int(noisy_im.shape[2]/2 + (noisy_im.shape[2]/(2/(ratio)))),
                        int(noisy_im.shape[3]/2 - (noisy_im.shape[3]/(2/(ratio)))):int(noisy_im.shape[3]/2 + (noisy_im.shape[3]/(2/(ratio))))
                        ]
    else:
        noisy_img = noisy_im

    # ........................... Generating background mask
    if mask is None:
        background_threshold = 0.2
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

