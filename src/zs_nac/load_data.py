from zs_nac.utils import *


def load_noisy_data(data):
    """
    :param data: Images should be in numpy format
    Images can be complex, in this case use n_chan=2.
    Images can be magnitude (real data), in this case use n_chan=1.
    Image shape can be 2D, 3D or 4D:
    2D: [height/inplane_dim1, width/inplane_dim2]:
    3D: [height/inplane_dim1, width/inplane_dim2, slice/phase2]
    4D: [height/inplane_dim1, width/inplane_dim2, slice/phase2, 4th dim]
    :return: preprocessed noisy images in a 4D matrix: [slice, channel, height, width]
    """

    if data.ndim == 2:
        img_ndim = 2
        fourth_dim = 1
        data = data[..., None]
    elif data.ndim == 3:
        img_ndim = 3
        fourth_dim = 1
        data = data
    elif data.ndim == 4:
        img_ndim = 4
        fourth_dim = data.shape[-1]
        data = np.reshape(
            data,
            (
                data.shape[0],
                data.shape[1],
                data.shape[2] * data.shape[3],
            ),
        )
    else:
        print("Matrix size is neither 2D, 3D nor 4D. Please check your input dimension")
        exit()
    noisy_data_norm = data / data.std() * 100
    noisy_data_norm_reshaped = np.moveaxis(noisy_data_norm, -1, 0)
    if noisy_data_norm_reshaped.dtype == complex or noisy_data_norm_reshaped.dtype == 'complex64' or noisy_data_norm_reshaped.dtype == 'complex128':
        noisy_data = complex2dualchannel(noisy_data_norm_reshaped, 1)
        n_chan = 2
    else:
        noisy_data = np.expand_dims(noisy_data_norm_reshaped, 1)
        n_chan = 1
    # else:
    #     print('Error: number of channels n_chan should be either 1 or 2')
    #     exit()

    return noisy_data, img_ndim, fourth_dim, n_chan
