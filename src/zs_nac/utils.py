import numpy as np
from skimage import morphology
from torch.utils.data import DataLoader

def normalize(img_data):
    img_data_min = img_data - img_data.min()
    img_data_norm = img_data_min / img_data_min.max()
    return img_data_norm


def dualchannels2complex(double_channel_array):
    """
    This function transform a dual channel real/imaginary array to complex.
    It detects which dimension has the real/imaginary data and turns it to complex.
    If it happens that the array has more than one dimension with a shape of 2, this function won't work.
    :param double_channel_array: array with dual channel for real and imaginary
    :return: complex array
    """
    idx = []
    i = 0
    for l in double_channel_array.shape:
        if l == 2:
            idx.append(i)
        i += 1
    if len(idx) > 1:
        print(
            "your array has more than one dimension with size 2, this function cannot identify which dimension "
            "corresponds to real and imaginary dimension"
        )
    else:
        double_channel_reshaped = np.moveaxis(double_channel_array, idx[0], 0)
        complex_array = double_channel_reshaped[0] + 1j * double_channel_reshaped[1]
    return complex_array


def complex2dualchannel(complex_array, dual_channel_dim=1):
    """
    :param complex_array
    :param dual_channel_dim: where to add the additional channel
    :return: dual channel real/imaginary array
    """
    array_2ch = np.empty(complex_array.shape + (2,))
    array_2ch[..., 0] = np.real(complex_array)
    array_2ch[..., 1] = np.imag(complex_array)
    array_2ch_reashaped = np.moveaxis(array_2ch, -1, dual_channel_dim)
    return array_2ch_reashaped


def create_mask(img_ch, threshold=0.1):
    """
    Create a background mask
    :param img_ch: 4D data [slice, channel, height, width]
    :param threshold: float < 1, default = 0.1
    :return mask: 3D mask
    """

    mask = np.empty(img_ch[:, 0].shape)
    for k in range(img_ch.shape[0]):
        if img_ch.shape[1] == 2:
            noisy_img_abs_norm = normalize(abs(dualchannels2complex(img_ch))[k])
        else:
            noisy_img_abs_norm = normalize(abs(img_ch[k, 0]))
        orig_imgs_norm_threshold = np.where((noisy_img_abs_norm) > threshold, 1, 0)
        mask_background_0 = morphology.remove_small_objects(
            orig_imgs_norm_threshold.astype(bool), min_size=10, connectivity=2
        ).astype(int)
        mask_background_1 = morphology.area_closing(
            mask_background_0,
            area_threshold=64,
            connectivity=1,
            parent=None,
            tree_traverser=None,
        )

        if mask_background_1.sum() > 500: # to avoid a null mask in case the slice is noise everywhere
            mask_background_2 = morphology.convex_hull_image(mask_background_1)
        else:
            mask_background_2 = np.empty(mask_background_1.shape)
            mask_background_2[:] = np.nan
        mask[k] = 1 - mask_background_2

    return mask


def estimate_background_noise(noisy_img, mask_background, n_chan=2):
    """
    Estimate the noise standard deviation in real and imaginary channels of the noisy data based on image background
    :param noisy_img: noisy data shape: [slice, channel, height, width]
    :param mask_background:
    :return: noise standard deviation
    """
    if n_chan == 2:
        a_times_real_img = mask_background * noisy_img[:, 0]
        a_times_imag_img = mask_background * noisy_img[:, 1]
        noise_std = np.nanstd(
            [
                a_times_real_img[np.abs(a_times_real_img) != 0],
                a_times_imag_img[np.abs(a_times_imag_img) != 0],
            ]
        )
        noise_mean = 0
    else:
        a_times_img = mask_background * noisy_img[:, 0]
        noise_std = np.nanstd(
            [
                a_times_img[np.abs(a_times_img) != 0],
            ]
        )
        noise_mean = np.nanmean(
            [
                a_times_img[np.abs(a_times_img) != 0],
            ]
        )

    return noise_std, noise_mean


def myPSNR(ref, denoised):
    """This function calculates PSNR between the reference and
    the denoised image"""
    mse = np.sum(np.square(np.abs(ref - denoised))) / ref.size
    psnr = 20 * np.log10(ref.max() / (np.sqrt(mse) + 1e-10))
    return psnr


def create_loaders(
    noisy_input: np.ndarray[np.complex64],
    # doubly_noisy_input: np.ndarray[np.complex64],
    batch_size
):
    """
    Returns:
        train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader
    """
    train_data = Dataset(noisy_input) #, doubly_noisy_input)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # 15s faster that num_worker=6 without pin_memory
    )

    return train_loader

class Dataset():
    def __init__(self, trn_noisy):#, trn_doublynoisy):
        self.trn_noisy = trn_noisy.clone().detach().squeeze(
            dim=tuple(range(2, trn_noisy.ndim))
        )
        # self.trn_doublynoisy = trn_doublynoisy.clone().detach().squeeze(
        #     dim=tuple(range(2, trn_doublynoisy.ndim))
        # )
    def __len__(self):
        return len(self.trn_noisy)
    def __getitem__(self, idx):
        nw_input = self.trn_noisy[idx]
        # nw_input_doublynoisy = self.trn_doublynoisy[idx]

        return nw_input#, nw_input_doublynoisy

