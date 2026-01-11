# Zero-Shot self-supervised denoising for MRI data
## Description
This Python scripts is a fast zero-shot self-supervised denoising method for MRI data.  
Paper: Fast zero-shot deep learning-based denoising method for low-field MR images (https://www.doi.org/10.1007/s10334-025-01311-w).
## Package installation and execution
First, clone the repository:
```bash
git clone https://github.com/reinaayde7/zs-nac.git
```
Then, go to zs-nac directory and install the package by writing in terminal:
```bash
pip install .
```
To use the package, import nac function from the installed package:
```bash
from zs_nac.denoising_func import nac
```
Then use the function:
```bash
denoised_image = nac(noisy_image, device='cuda')
```
To further accelerate the training process:
```bash
denoised_image = nac(noisy_image, training_part=True, ratio=0.3, device='cuda')
```
## Function arguments
nac function has several arguments:
- noisy_data: is the noisy input images. It can be complex or magnitude images.
Input should be a numpy array of 2, 3 or 4 dimensions:   
2D: [height (inplane_dimension1), width (inplane_dimension2)]  
3D: [height (inplane_dimension1), width (inplane_dimension2), slice]  
4D: [height (inplane_dimension1), width (inplane_dimension2), slice, 4th dimension]
If input data is only real or imaginary data, the internal function of mask creation will fail. In this case, the background mask must be given as an input.
- mask: is the background mask and has the same shape of noisy input images. Default: None, which means that it will be automatically created. 
- alpha: α is the added noise factor in σ_B=ασ_A (check the paper). Default: 1  
- correction: whether to apply a correction at inference stage (check the paper). Default: False.
- batch_size: training batch size. Keep the number low if your computer doesn't have enough RAMs. Default:1
- max_epochs: maximum number of epochs. Default: 2000  
- stop_training: The number of epochs needed to stop the training when validation error is not improving. Default: 50 
- training_part: whether to train the model on a fewer number of slices (True) or the whole matrix (False).
     recommended when the number of slice or 4th dimension is too big resulting in a significantly slow denoising process. Default: False
- ratio: when training_part==True, ratio defines the ratio of slices/4th dimension to train on. Default: 0.3
- device: 'cuda' or 'cpu'. Default: 'cuda'

The output is a numpy array with the same shape and type of the input array.

## Denoising function (nac) workflow details
1. Loading the numpy MRI complex or magnitude images  
2. Creating the background mask  
3. Estimating the noise standard deviation  
4. training the model  
5. Inference

## Notes
- This method could work on MR images provided that the noise distribution in those images follows a Gaussian distribution or close enough.
- This method is not recommended for non-cartesian or zero-filled k-spaces as noise in corresponding MR images won't follow a Gaussian distribution.

## Citation
If you use this software in your work, please consider citing our paper:

Ayde R, Zihlmann G, Salameh N, Sarracanie M. Fast zero-shot deep learning-based denoising method for low-field MR images. MAGMA. Published online December 22, 2025. doi:10.1007/s10334-025-01311-w
https://www.doi.org/10.1007/s10334-025-01311-w