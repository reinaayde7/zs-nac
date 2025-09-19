import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
from zs_nac.utils import *

# torch.manual_seed(1)


def training(
    network,
    gamma,
    noisy_img,
    noise_std,
    noise_mean,
    n_chan,
    lr,
    batch_size,
    step_size,
    factor,
    max_epochs,
    stop_training,
    device,
):
    optimizer_noisier2noise = optim.Adam(network.parameters(), lr=lr)
    scheduler_noisier2noise = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_noisier2noise, mode="min", factor=factor, patience=step_size
    )
    val_loss_tracker = 0
    min_val_loss = np.inf
    epoch = 0
    ################################################################################################################
    # Creating train loaders
    ################################################################################################################
    # doubly_noisy_img = add_noise(noisy_img, gamma, noise_std, noise_mean, n_chan, device)
    train_loader = create_loaders(
        noisy_img,
        # doubly_noisy_img,
        batch_size,
    )
    ################################################################################################################
    # Creating validation set
    ################################################################################################################
    doubly_noise_img_val = add_noise(noisy_img, gamma, noise_std, noise_mean, n_chan, device)
    noisy1 = single_downsampler(noisy_img).to(device)
    doubly_noisy1 = single_downsampler(doubly_noise_img_val).to(device)
    ################################################################################################################
    # Start the training
    ################################################################################################################
    while epoch <= max_epochs and val_loss_tracker < stop_training:
        # a = time.time()
        if n_chan == 1:
            training_loss, lr_noisier2noise = train(
                network,
                train_loader,
                gamma,
                noise_std,
                noise_mean,
                n_chan,
                optimizer_noisier2noise,
                scheduler_noisier2noise,
                device
            )

            new_val_loss = loss_func(noisy1, network(doubly_noisy1))
            if new_val_loss >= min_val_loss and epoch >= 50:
                val_loss_tracker += 1  # reset the val loss tracker each time a new lowest val error is achieved
            else:
                min_val_loss = new_val_loss
                val_loss_tracker = 0
        else:
            training_loss, doubly_noisy_img, lr_noisier2noise = train(
                network,
                train_loader,
                gamma,
                noise_std,
                noise_mean,
                n_chan,
                optimizer_noisier2noise,
                scheduler_noisier2noise,
                device,
            )

            # validation data
            new_val_loss = loss_func(noisy1, network(doubly_noisy1))
            if new_val_loss >= min_val_loss and epoch >= 50:
                val_loss_tracker += 1  # reset the val loss tracker each time a new lowest val error is achieved
            else:
                min_val_loss = new_val_loss
                val_loss_tracker = 0

        # writer.add_scalar("val_loss_noisier2noise", new_val_loss, epoch)
        # writer.add_scalar("training_loss_noisier2noise", training_loss, epoch)
        # writer.add_scalar("Lr_noisier2noise", lr_noisier2noise, epoch)

        epoch += 1
        # b= time.time()
        # print('epoch time {}'.format(b-a))

    return network, doubly_noisy_img, epoch


def single_downsampler(img):
    # img has shape [Batch Channel Height Width]
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 1], [0, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    return output1


def add_noise(x, gamma, std_noise, mean_noise, n_chan, device):
    if n_chan == 1:
        added_noise = torch.normal(mean_noise, std_noise * gamma, size=x.shape).to(device)
        noisy = x + added_noise
    else:
        noisy = torch.empty(x.shape).to(device)
        added_real_noise = torch.normal(
            mean_noise, std_noise * gamma, size=x[:, 0, :, :].shape
        ).to(device)
        added_imag_noise = torch.normal(
            mean_noise, std_noise * gamma, size=x[:, 0, :, :].shape
        ).to(device)
        noisy[:, 0, :, :] = x[:, 0, :, :] + added_real_noise
        noisy[:, 1, :, :] = x[:, 1, :, :] + added_imag_noise

    return noisy


def train(network,
          train_loader,
          gamma,
          noise_std,
          noise_mean,
          n_chan,
          optimizer,
          scheduler,
          device):

    avg_trn_cost = 0
    network.train(True)
    for ii, batch in enumerate(train_loader):
        # (
        #     noisy_input # doubly_noisy_input,
        # ) = from_batch_to_devide(batch, device)
        noisy_input = batch
        doubly_noisy_input = add_noise(noisy_input, gamma, noise_std, noise_mean, n_chan, device)
        """Forward Path"""
        output = network(
            doubly_noisy_input
        )

        """Loss"""
        loss = loss_func(noisy_input, output)
        avg_trn_cost += loss.item() / len(train_loader)

        """Backpropagation"""
        network.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        scheduler.step(metrics=loss)
        lr = get_lr(optimizer)

    return avg_trn_cost, doubly_noisy_input, lr
    # loss = loss_func(noisy_img, doubly_noisy_img, network)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # scheduler.step(metrics=loss)
    # lr = get_lr(optimizer)
    # return loss.item(), lr
def from_batch_to_devide(batch, device):
    return (ele.to(device).float() for ele in iter(batch[:]))


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)


def loss_func(noisy_img, output):
    loss = mse(noisy_img, output)
    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
