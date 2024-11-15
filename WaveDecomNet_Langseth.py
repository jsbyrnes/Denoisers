import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
from WaveDecompNet import autoencoder_1D_models_torch as wdn
import copy
import warnings
import LangsethData
import os
import argparse

import time

######
#This is a reproduction of the code from this paper, but I redid it to figure out why its so fucking slow. I think its the lstm.
#Not a very complicated architeture. 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_fn=nn.ELU, dropout_prob=0.0, use_relu=True):
        super(ConvBlock, self).__init__()
        if stride>1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size // 2))
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.use_relu = use_relu
        self.activation = activation_fn() if use_relu else None
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_fn=nn.ELU, use_relu=True):
        super(ConvTransBlock, self).__init__()
        self.convtrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=(kernel_size // 2), output_padding=stride - 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.use_relu = use_relu
        self.activation = activation_fn() if use_relu else None

    def forward(self, x):
        x = self.convtrans(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.activation(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, fac, activation_fn, dropout_prob):
        super(Encoder, self).__init__()
        # 7 layers of convolutions with filters multiplied by fac
        self.enc1 = ConvBlock(in_channels, int(8 * fac), kernel_size=9, stride=1, activation_fn=activation_fn, dropout_prob=dropout_prob)
        self.enc2 = ConvBlock(int(8 * fac), int(8 * fac), kernel_size=9, stride=2, activation_fn=activation_fn, dropout_prob=dropout_prob)
        self.enc3 = ConvBlock(int(8 * fac), int(16 * fac), kernel_size=7, stride=1, activation_fn=activation_fn, dropout_prob=dropout_prob)  # Skip connection
        self.enc4 = ConvBlock(int(16 * fac), int(16 * fac), kernel_size=7, stride=2, activation_fn=activation_fn, dropout_prob=dropout_prob)
        self.enc5 = ConvBlock(int(16 * fac), int(32 * fac), kernel_size=5, stride=1, activation_fn=activation_fn, dropout_prob=dropout_prob)  # Skip connection
        self.enc6 = ConvBlock(int(32 * fac), int(32 * fac), kernel_size=5, stride=2, activation_fn=activation_fn, dropout_prob=dropout_prob)
        self.enc7 = ConvBlock(int(32 * fac), int(64 * fac), kernel_size=3, stride=1, activation_fn=activation_fn, dropout_prob=dropout_prob)  # Skip connection

    def forward(self, x):
        # Forward pass through the encoder, returning skip connections
        skip_connections = []
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        skip_connections.append(x)  # Skip connection from layer 3
        x = self.enc4(x)
        x = self.enc5(x)
        skip_connections.append(x)  # Skip connection from layer 5
        x = self.enc6(x)
        x = self.enc7(x)
        skip_connections.append(x)  # Skip connection from layer 7
        return x, skip_connections

class Decoder(nn.Module):
    def __init__(self, out_channels, fac, activation_fn, use_attention_on_skips=False):
        super(Decoder, self).__init__()
        self.use_attention_on_skips = use_attention_on_skips
        
        # Transpose convolutions for upsampling, with batch norm and activations
        self.dec8 = ConvTransBlock(int(64 * fac), int(64 * fac), kernel_size=3, stride=1, activation_fn=activation_fn)
        self.dec7 = ConvTransBlock(int(64 * fac), int(32 * fac), kernel_size=3, stride=2, activation_fn=activation_fn)
        self.dec6 = ConvTransBlock(int(32 * fac), int(32 * fac), kernel_size=5, stride=1, activation_fn=activation_fn)
        self.dec5 = ConvTransBlock(int(32 * fac), int(16 * fac), kernel_size=5, stride=2, activation_fn=activation_fn)
        self.dec4 = ConvTransBlock(int(16 * fac), int(16 * fac), kernel_size=7, stride=1, activation_fn=activation_fn)
        self.dec3 = ConvTransBlock(int(16 * fac), int(8 * fac), kernel_size=7, stride=2, activation_fn=activation_fn)
        self.dec2 = ConvTransBlock(int(8 * fac), int(8 * fac), kernel_size=9, stride=1, activation_fn=activation_fn)
        self.dec1 = ConvTransBlock(int(8 * fac), out_channels, kernel_size=9, stride=1, activation_fn=activation_fn, use_relu=False)
        
        # Define individual attention layers for each skip connection if enabled
        if self.use_attention_on_skips:
            self.attention_layer_1 = nn.MultiheadAttention(embed_dim=int(64 * fac), num_heads=4, batch_first=True)
            self.attention_layer_2 = nn.MultiheadAttention(embed_dim=int(32 * fac), num_heads=4, batch_first=True)
            self.attention_layer_3 = nn.MultiheadAttention(embed_dim=int(16 * fac), num_heads=4, batch_first=True)

    def forward(self, x, skip_connections):
        # Apply attention to skip connections if use_attention_on_skips is True
        if self.use_attention_on_skips:
            skip_connections[2] = self.apply_attention(skip_connections[2], self.attention_layer_1)
            skip_connections[1] = self.apply_attention(skip_connections[1], self.attention_layer_2)
            skip_connections[0] = self.apply_attention(skip_connections[0], self.attention_layer_3)

        # Forward pass through the decoder with skip connections
        x = self.dec8(x) + skip_connections[2]  # Skip connection from layer 7
        x = self.dec7(x)
        x = self.dec6(x) + skip_connections[1]  # Skip connection from layer 5
        x = self.dec5(x)
        x = self.dec4(x) + skip_connections[0]  # Skip connection from layer 3
        x = self.dec3(x)
        x = self.dec2(x)
        x = F.tanh(self.dec1(x))
        return x

    def apply_attention(self, skip, attention_layer):
        # Apply attention to a skip connection with the specified attention layer
        skip = skip.transpose(1, 2)  # Transpose for attention compatibility
        attended_skip, _ = attention_layer(skip, skip, skip)
        return attended_skip.transpose(1, 2)  # Transpose back

class UNetWithBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, fac=1, bottleneck_type=None, activation_fn=nn.ReLU, dropout_prob=0.0, single_decoder=False, use_attention_on_skips=False):
        super(UNetWithBottleneck, self).__init__()
        self.encoder = Encoder(in_channels, fac, activation_fn, dropout_prob)
        self.use_attention_on_skips = use_attention_on_skips

        # Bottleneck layers for each decoder
        self.bottleneck_type = bottleneck_type
        self.dropout_prob = dropout_prob
        self.single_decoder = single_decoder

        if bottleneck_type == 'linear':
            self.bottleneck1 = nn.Sequential(
                nn.Linear(int(64 * fac), int(64 * fac)),
                nn.Dropout(dropout_prob)
            )
            if not single_decoder:
                self.bottleneck2 = nn.Sequential(
                    nn.Linear(int(64 * fac), int(64 * fac)),
                    nn.Dropout(dropout_prob)
                )
        elif bottleneck_type == 'lstm':
            self.bottleneck1 = nn.LSTM(int(64 * fac), int(32 * fac), 2, batch_first=True, bidirectional=True)
            self.dropout1 = nn.Dropout(dropout_prob)
            if not single_decoder:
                self.bottleneck2 = nn.LSTM(int(64 * fac), int(32 * fac), 2, batch_first=True, bidirectional=True)
                self.dropout2 = nn.Dropout(dropout_prob)
        elif bottleneck_type == 'gru':
            self.bottleneck1 = nn.GRU(int(64 * fac), int(32 * fac), 2, batch_first=True, bidirectional=True)
            self.dropout1 = nn.Dropout(dropout_prob)
            if not single_decoder:
                self.bottleneck2 = nn.GRU(int(64 * fac), int(32 * fac), 2, batch_first=True, bidirectional=True)
                self.dropout2 = nn.Dropout(dropout_prob)
        elif bottleneck_type == 'attention':
            self.bottleneck1 = nn.MultiheadAttention(embed_dim=int(64 * fac), num_heads=int(4 * fac), batch_first=True)
            self.dropout1 = nn.Dropout(dropout_prob)
            if not single_decoder:
                self.bottleneck2 = nn.MultiheadAttention(embed_dim=int(64 * fac), num_heads=int(4 * fac), batch_first=True)
                self.dropout2 = nn.Dropout(dropout_prob)
        else:
            self.bottleneck1 = None
            self.bottleneck2 = None if not single_decoder else None

        # One or two decoders based on the single_decoder flag
        self.decoder1 = Decoder(out_channels, fac, activation_fn, use_attention_on_skips=self.use_attention_on_skips)
        if not single_decoder:
            self.decoder2 = Decoder(out_channels, fac, activation_fn, use_attention_on_skips=self.use_attention_on_skips)

    def forward(self, x):
        # Encoder pass
        x, skip_connections = self.encoder(x)

        # Bottleneck processing for decoder1
        if self.bottleneck1 is not None:
            if isinstance(self.bottleneck1, nn.Sequential):
                x1 = self.bottleneck1(x)
            elif isinstance(self.bottleneck1, nn.LSTM):
                x1, _ = self.bottleneck1(x.transpose(1, 2))  # LSTM expects [batch, seq, features]
                x1 = self.dropout1(x1).transpose(1, 2)
            elif isinstance(self.bottleneck1, nn.GRU):
                x1, _ = self.bottleneck1(x.transpose(1, 2))
                x1 = self.dropout1(x1).transpose(1, 2)
            elif isinstance(self.bottleneck1, nn.MultiheadAttention):
                x1, _ = self.bottleneck1(x.transpose(1, 2), x.transpose(1, 2), x.transpose(1, 2))
                x1 = self.dropout1(x1).transpose(1, 2)
        else:
            x1 = x

        # Decoder1 pass
        output1 = self.decoder1(x1, skip_connections)

        # If only one decoder, return output1
        if self.single_decoder:
            return (output1,)

        # Bottleneck processing for decoder2
        if self.bottleneck2 is not None:
            if isinstance(self.bottleneck2, nn.Sequential):
                x2 = self.bottleneck2(x)
            elif isinstance(self.bottleneck2, nn.LSTM):
                x2, _ = self.bottleneck2(x.transpose(1, 2))  # LSTM expects [batch, seq, features]
                x2 = self.dropout2(x2).transpose(1, 2)
            elif isinstance(self.bottleneck2, nn.GRU):
                x2, _ = self.bottleneck2(x.transpose(1, 2))
                x2 = self.dropout2(x2).transpose(1, 2)
            elif isinstance(self.bottleneck2, nn.MultiheadAttention):
                x2, _ = self.bottleneck2(x.transpose(1, 2), x.transpose(1, 2), x.transpose(1, 2))
                x2 = self.dropout2(x2).transpose(1, 2)
        else:
            x2 = x

        # Decoder2 pass
        output2 = self.decoder2(x2, skip_connections)

        return output1, output2
        
class NormalizedMSELoss(nn.Module):
    def __init__(self):
        super(NormalizedMSELoss, self).__init__()

    def forward(self, input, target):
        # Compute mean squared error
        mse_loss = torch.mean((input - target) ** 2, dim=-1, keepdim=True)

        # Compute the variance of the target
        var_target = torch.var(target, dim=-1, unbiased=False, keepdim=True)

        # Normalize the MSE loss by the variance of the target
        normalized_loss = mse_loss / (var_target + 1e-9)  # Small epsilon to prevent division by zero

        # Take the mean over batch and channel dimensions
        loss = normalized_loss.mean()

        return loss
#####

def plot_validation(valbatch, out, tag):
    fig, axes = plt.subplots(10, 2, figsize=(10, 20))

    for k in range(10):
        # First column: Target vs Denoised Signal
        a1 = np.max(np.abs(valbatch[0][k, 0, :].detach().numpy()))
        a2 = np.max(np.abs(out[0][k, 0, :].detach().numpy()))
        a = np.max([a1, a2])
        denoised_signal = 0.5 * out[0][k, 0, :].detach().numpy() / a
        target_signal = 0.5 * valbatch[0][k, 0, :].detach().numpy() / a
        axes[k, 0].plot(target_signal, 'k', linewidth=0.75)
        axes[k, 0].plot(denoised_signal, 'r')
        axes[k, 0].set_title('Target vs Denoised', fontsize=8)
        axes[k, 0].set_ylim(-1, 1)
        axes[k, 0].axis('off')

        # Second column: Input Combined Signal vs Denoised Signal
        a1 = np.max(np.abs(valbatch[1][k, 0, :].detach().numpy()))
        a2 = np.max(np.abs(out[0][k, 0, :].detach().numpy()))
        a = np.max([a1, a2])
        denoised_signal = 0.5 * out[0][k, 0, :].detach().numpy() / a
        combined_signal = 0.5 * valbatch[1][k, 0, :].detach().numpy() / a
        axes[k, 1].plot(combined_signal, 'k', linewidth=0.75)
        axes[k, 1].plot(denoised_signal, 'r')
        axes[k, 1].set_title('Input vs Denoised', fontsize=8)
        axes[k, 1].set_ylim(-1, 1)
        axes[k, 1].axis('off')

    fig.tight_layout()
    plt.savefig('Validation_WDN' + tag + '.pdf')
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def iterate_denoiser(mini_labels, noise_samples, model0):
    # First, for the original set, get the scaling of the noise
    scales = torch.zeros((mini_labels.shape[0], mini_labels.shape[1])).to(mini_labels.device)
    for ix1 in range(mini_labels.shape[0]):  # batch
        for ix2 in range(mini_labels.shape[1]):  # channel
            sig_amp = torch.max(torch.abs(mini_labels[ix1, ix2, :]))
            scales[ix1, ix2] = torch.max(torch.abs(noise_samples[ix1, ix2, :])) / sig_amp

    # Clean the labeled signal
    new_labels = model0(mini_labels)[0]
    # Clean the noise
    new_noise = model0(noise_samples)[1]

    # Renormalize by first resetting the signals to 1 and rescaling the noise
    normalized_new_labels = new_labels.clone()
    normalized_new_noise = new_noise.clone()

    for ch in range(new_labels.shape[1]):
        max_amp = torch.max(torch.abs(new_labels[:, ch, :]), dim=-1).values  # Shape: (128,)
        max_amp_expanded = max_amp.unsqueeze(1)  # Shape: (128, 1)

        non_zero_mask = max_amp > 0
        normalized_new_labels[:, ch, :] = torch.where(
            non_zero_mask.unsqueeze(1),
            new_labels[:, ch, :] / max_amp_expanded,
            new_labels[:, ch, :]
        )

        max_amp = torch.max(torch.abs(new_noise[:, ch, :]), dim=-1).values  # Shape: (128,)
        non_zero_mask = max_amp > 0
        sca = scales[:, ch].unsqueeze(1)
        normalized_new_noise[:, ch, :] = torch.where(
            non_zero_mask.unsqueeze(1),
            sca * new_noise[:, ch, :] / max_amp.unsqueeze(1),
            new_noise[:, ch, :]
        )

    #Combine the normalized signals
    new_inputs = normalized_new_labels + noise_samples

    # Renormalize new_inputs and new_labels
    final_new_inputs = new_inputs.clone()
    final_new_labels = normalized_new_labels.clone()

    for ch in range(new_labels.shape[1]):
        max_amp = torch.max(torch.abs(new_inputs[:, ch, :]), dim=-1).values  # Shape: (128,)
        max_amp_expanded = max_amp.unsqueeze(1)  # Shape: (128, 1)

        non_zero_mask = max_amp > 0
        final_new_inputs[:, ch, :] = torch.where(
            non_zero_mask.unsqueeze(1),
            new_inputs[:, ch, :] / max_amp_expanded,
            new_inputs[:, ch, :]
        )
        final_new_labels[:, ch, :] = torch.where(
            non_zero_mask.unsqueeze(1),
            normalized_new_labels[:, ch, :] / max_amp_expanded,
            normalized_new_labels[:, ch, :]
        )

    return final_new_inputs, final_new_labels, (final_new_inputs - final_new_labels)

def plot_and_save_training_history(training_hist, val_hist, model_name, lr_history, warmup_epochs, test_loss=None):
    """
    Plots and saves training and validation history by epoch, with annotations for learning rate changes.
    
    Parameters:
        training_hist (list or array): List of training loss (or metric) values, one per epoch.
        val_hist (list or array): List of validation loss (or metric) values, one per epoch.
        model_name (str): Name of the model, used to save the figure.
        lr_history (list or array): List of learning rate values for each epoch.
        warmup_epochs (int): Number of epochs in the warm-up phase.
        test_loss (float, optional): Test loss to plot as a blue star at the end of training.
    """
    epochs = range(1, len(training_hist) + 1)  # X-axis: epoch numbers
    
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation history
    plt.plot(epochs, training_hist, label='Training', color='darkblue', marker='o')
    plt.plot(epochs, val_hist, label='Validation', color='black', marker='o')
    
    # Ensure warmup_lr has a default value in case warmup_epochs > len(lr_history)
    warmup_lr = lr_history[0] if len(lr_history) > 0 else 0.0  # Default to the first lr or 0 if empty
    if warmup_epochs+1 <= len(lr_history):
        warmup_lr = lr_history[warmup_epochs]
        plt.annotate(f'LR: {warmup_lr:.1e}; end of warm-up',
                     (warmup_epochs+1, max(val_hist[warmup_epochs], training_hist[warmup_epochs]) * 1.1),
                     textcoords="offset points",
                     xytext=(0, 10), ha='center',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
                     color='black')

    # Annotate subsequent learning rate changes after warm-up
    prev_lr = warmup_lr
    for epoch in range(warmup_epochs + 1, len(lr_history)):
        lr = lr_history[epoch]
        if lr != prev_lr:  # Detect a change in the learning rate
            plt.annotate(f'LR: {lr:.1e}', 
                         (epoch, max(val_hist[epoch-1], training_hist[epoch-1]) * 1.1),  # +1 to match epoch range in plot
                         textcoords="offset points", 
                         xytext=(0, 10), ha='center',
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'),
                         color='black')
            prev_lr = lr  # Update the previous learning rate to the current one

    # Plot the test loss as a blue star at the end if provided
    if test_loss is not None:
        plt.plot(len(epochs), test_loss, 'b*', markersize=12, label='Test Loss')
    
    # Labeling
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation History for {model_name}')
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.yscale('log')  # Set y-axis to log scale
    
    # Save the plot as a PDF file with the model name
    plt.savefig(f'{model_name}_training_history.pdf')
    plt.close()

def training_loop_with_warmup(
    model, 
    optimizer, 
    criterion, 
    scheduler, 
    dataloader_train, 
    dataloader_valid, 
    n_epochs, 
    mini_batch_size, 
    device, 
    val_sweeps, 
    tag, 
    lr, 
    warmup_epochs=5, 
    warmup_factor=0.1,
    model0=None
):
    # Initialize training history lists
    training_hist = np.array([])
    val_hist = np.array([])
    lr_hist = np.array([])

    # Measure batch fetch time for validation
    print("Fetching a batch from validation DataLoader for plotting.")
    for batch in dataloader_valid:
        valbatch = batch  # Adjust depending on data structure
        break

    if model0:
        val_input  = valbatch[1].to(device)
        val_labels = valbatch[0].to(device)
        noise_samples = val_input - val_labels
        valbatch[1], valbatch[0], _ = iterate_denoiser(val_input, noise_samples, model0)

        valbatch[0].cpu()
        valbatch[1].cpu()

    # Training loop
    for epoch in range(n_epochs):
        model.train()

        # Initialize timing accumulators
        total_load_time = 0.0
        total_transfer_time = 0.0
        total_runtime = 0.0

        rl = 0.0
        cc = 0.0
        vrl = 0.0
        vcc = 0.0

        # Adjust learning rate for warm-up
        if epoch <= warmup_epochs:
            warmup_lr = lr * (warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs))
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Print current epoch and learning rate
        print(f"-----> Epoch# {epoch + 1}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        start_time = time.time()

        for large_batch in dataloader_train:

            # Track load time
            loadtime = time.time() - start_time
            total_load_time += loadtime
            start_time = time.time()

            # Transfer large batch to GPU
            inputs_large = large_batch[1].to(device, non_blocking=True)
            labels_large = large_batch[0].to(device, non_blocking=True)

            transtime = time.time() - start_time
            total_transfer_time += transtime
            start_time = time.time()

            # Divide large batch into mini-batches
            for i in range(0, inputs_large.shape[0], mini_batch_size):
                # Skip last if irregularly sized at the end
                if inputs_large.shape[0] < i + mini_batch_size:
                    break

                mini_inputs = inputs_large[i:i + mini_batch_size]
                mini_labels = labels_large[i:i + mini_batch_size]
                noise_samples = mini_inputs - mini_labels

                if model0:
                    mini_inputs, mini_labels, noise_samples = iterate_denoiser(mini_labels, noise_samples, model0)

                # Forward pass and optimization
                optimizer.zero_grad()
                out = model(mini_inputs)

                loss = criterion(out[0], mini_labels)

                # If the model has more than one decoder, calculate additional loss
                if not model.single_decoder:
                    loss2 = criterion(out[1], noise_samples)
                    loss = loss + loss2

                loss.backward()
                optimizer.step()

                rl += loss.item()
                cc += 1

                # Track runtime
                runtime = time.time() - start_time
                total_runtime += runtime
                start_time = time.time()

        avg_training_loss = math.sqrt(rl / cc)
        
        # Epoch timing summary
        epoch_total_time = total_load_time + total_transfer_time + total_runtime
        load_time_percent = (total_load_time / epoch_total_time) * 100
        transfer_time_percent = (total_transfer_time / epoch_total_time) * 100
        runtime_percent = (total_runtime / epoch_total_time) * 100

        print(f"Epoch {epoch + 1}/{n_epochs} - Total Time: {epoch_total_time:.2f} seconds")
        print(f"  Load Time: {total_load_time:.2f} seconds ({load_time_percent:.2f}%)")
        print(f"  Transfer Time: {total_transfer_time:.2f} seconds ({transfer_time_percent:.2f}%)")
        print(f"  Runtime (model training): {total_runtime:.2f} seconds ({runtime_percent:.2f}%)")
        print(f"  Average Running Loss: {avg_training_loss:.4f}")
        if device.type == 'cuda':
            print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for _ in range(val_sweeps):
                for large_batch in dataloader_valid:
                    # Transfer large batch to GPU
                    inputs_large = large_batch[1].to(device, non_blocking=True)
                    labels_large = large_batch[0].to(device, non_blocking=True)

                    # Divide the large batch into mini-batches and perform validation
                    for i in range(0, inputs_large.shape[0], mini_batch_size):
                        # Skip last if irregularly sized at the end
                        if inputs_large.shape[0] < i + mini_batch_size:
                            break

                        mini_inputs = inputs_large[i:i + mini_batch_size]
                        mini_labels = labels_large[i:i + mini_batch_size]
                        noise_samples = mini_inputs - mini_labels

                        if model0:
                            mini_inputs, mini_labels, noise_samples = iterate_denoiser(mini_labels, noise_samples, model0)

                        # Forward pass and loss calculation
                        out = model(mini_inputs)
                        loss = criterion(out[0], mini_labels)

                        # If the model has more than one decoder, calculate additional loss
                        if not model.single_decoder:
                            loss2 = criterion(out[1], noise_samples)
                            loss += loss2

                        val_loss += loss.item()
                        val_batch_count += 1

        avg_val_loss = math.sqrt(val_loss / val_batch_count)
        print(f'--> Validation loss: {avg_val_loss:.4f}')

        # Append losses and learning rate to history
        training_hist = np.append(training_hist, avg_training_loss)
        val_hist = np.append(val_hist, avg_val_loss)
        lr_hist = np.append(lr_hist, optimizer.param_groups[0]['lr'])

        # Plot training history
        plot_and_save_training_history(
            training_hist, 
            val_hist, 
            "model_WaveDecomp" + tag, 
            lr_hist, 
            warmup_epochs=warmup_epochs
        )

        # Save current model
        torch.save(model.state_dict(), 'model_WaveDecomp' + tag + '.pt')

        # Plot validation samples
        model.to("cpu")
        out = model(valbatch[1])
        model.to(device)
        plot_validation(valbatch, out, tag)

        torch.cuda.empty_cache()  # Clear CUDA memory after each epoch

        # Early stopping condition
        if epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)

            if optimizer.param_groups[0]["lr"] <= lr / 128:
                print('Learning rate reduced below minimum')
                break

    return model, training_hist, val_hist, lr_hist

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Script takes an optional argument 'fac' and 'bottleneck' ")

    # Add the 'fac' argument with a default value
    parser.add_argument("--fac", type=str, default=1)
    parser.add_argument("--bottleneck", type=str, default='attention')
    parser.add_argument("--tag", type=str, default='WDN')
    parser.add_argument("--channels", type=str, default='vertical', help="What data to include:all, vertical (default), 3C, pressure (which is pressure and vertical together, horizontals")

    # Add the --test flag, which sets test_mode to True if passed
    parser.add_argument("--test", action='store_true', help="Enable test mode")
    parser.add_argument("--single_decoder", action='store_true', default=False, help="Single or multitask")
    parser.add_argument("--skip_attention", action='store_true', default=False, help="Apply attention to skip connections (use this) or not (don't use this)")

    # Parse the arguments
    args = parser.parse_args()

    fac = int(args.fac)
    bottleneck = args.bottleneck
    test_mode = args.test  # True if --test is passed, otherwise False
    tag = args.tag  # True if --test is passed, otherwise False
    channels = args.channels  # True if --test is passed, otherwise False
    single_decoder = args.single_decoder
    use_attention_on_skips = args.skip_attention

    num_cores = os.cpu_count()
    num_workers = int(min(22, num_cores-1))

    n_epochs = 200 #max
    
    nlen = 2048 # window length
    print_interval = 1  # Print every x batches

    if channels == 'all':
        nchan = 4
    elif channels == 'vertical':
        nchan = 1
    elif channels == 'pressure' or channels == 'horizontals':
        nchan = 2
    elif channels == '3C':
        nchan = 3

    # Define large and mini batch sizes
    large_batch_size = 1  # number of blocks to get, each 256 in size
    mini_batch_size = 128    # Size of each mini-batch within a large batch, this is what is processed
    val_sweeps = 5
    test_sweeps = 25
    # Warm-up parameters
    warmup_epochs = 5  # Number of epochs for warm-up
    warmup_factor = 0.1  # Initial factor for learning rate warm-up

    # Dataset creation
    print('Constructing the dataset')
    start_time = time.time()
    dataloader_train, dataloader_valid, dataloader_test = LangsethData.create_dataloaders(
        '../LangsethDatasetSignal_v4', '../LangsethDatasetNoise_v4', nlen, large_batch_size, mode='1d', channels=channels, 
        num_workers=num_workers, test_mode=test_mode, normalization_type="3C"
        )

    dataset_time = time.time() - start_time
    print(f"Dataset initialization time: {dataset_time:.2f} seconds")

    # Model initialization
    model_structure = "Branch_Encoder_Decoder"
    bottleneck_name = bottleneck
    model_name = model_structure + tag

    print("#" * 12 + f" building model {model_name} " + "#" * 12)
    start_time = time.time()
    model = UNetWithBottleneck(
        in_channels=nchan, out_channels=nchan, bottleneck_type=bottleneck_name, 
        activation_fn=nn.ELU, fac=fac, dropout_prob=0.0,
        single_decoder=single_decoder, use_attention_on_skips=use_attention_on_skips
    )
    model_init_time = time.time() - start_time
    print(f"Model initialization time: {model_init_time:.2f} seconds")

    # Model parameters and optimizer
    n = count_parameters(model)
    print("Parameters: " + str(n))
    lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1/4, patience=10, eps=1e-8, threshold=0.01)

    # Loss function
    criterion = nn.MSELoss()

    # Initialize history arrays
    training_hist = np.array([])
    val_hist = np.array([])
    lr_hist = np.array([])

    ######load a network for preprocessing for iterative denoising
    model0 = UNetWithBottleneck(
        in_channels=nchan, out_channels=nchan, bottleneck_type=bottleneck, 
        activation_fn=nn.ELU, fac=2, dropout_prob=0.0,
    )

    model0.load_state_dict(torch.load('./110324/model_WaveDecompY22_110324_fac2.pt', map_location=torch.device('cpu')))
    model0 = model0.float()
    model0.eval() #probably don't need but still 

    # Check if multiple GPUs are available and wrap the models with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model0 = nn.DataParallel(model0)
        model = nn.DataParallel(model)

    #define the models first
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")

    model.to(device)
    model0.to(device)

    # You can also add more detailed information about the device
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    elif device.type == 'mps':
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        print("Using CPU")

    # Training loop
    np.random.seed(10)
    torch.manual_seed(10)

    torch.backends.cudnn.benchmark = True

    model, training_hist, val_hist, lr_hist = training_loop_with_warmup(
        model, 
        optimizer, 
        criterion, 
        scheduler, 
        dataloader_train, 
        dataloader_valid, 
        n_epochs, 
        mini_batch_size, 
        device, 
        val_sweeps, 
        tag,
        lr, 
        warmup_epochs=warmup_epochs,
        warmup_factor=warmup_factor,
        model0=model0
    )
        
    # Test phase
    model.eval()
    test_loss = 0.0
    test_batch_count = 0

    with torch.no_grad():

        for _ in range(test_sweeps):

            for large_batch in dataloader_test:
                # Transfer large batch to GPU
                inputs_large = large_batch[1].to(device, non_blocking=True)
                labels_large = large_batch[0].to(device, non_blocking=True)

                # Divide the large batch into mini-batches and perform validation
                for i in range(0, inputs_large.shape[0], mini_batch_size):
                    
                    # Skip last if irregularly sized at the end
                    if inputs_large.shape[0] < i + mini_batch_size:
                        break

                    mini_inputs = inputs_large[i:i + mini_batch_size]
                    mini_labels = labels_large[i:i + mini_batch_size]
                    noise_samples = mini_inputs - mini_labels

                    if model0:
                        mini_inputs, mini_labels, noise_samples = iterate_denoiser(mini_labels, noise_samples, model0)

                    # Forward pass and loss calculation
                    out = model(mini_inputs)
                    loss = criterion(out[0], mini_labels)

                    # If the model has more than one decoder, calculate additional loss
                    if not model.single_decoder:
                        loss2 = criterion(out[1], noise_samples)
                        loss += loss2

                    test_loss += loss.item()
                    test_batch_count += 1

    avg_test_loss = math.sqrt(test_loss / test_batch_count)
    print(f'--> Test loss: {avg_test_loss:.4f}')
    # Plot training history
    plot_and_save_training_history(
        training_hist, 
        val_hist, 
        "model_WaveDecomp" + tag, 
        lr_hist, 
        warmup_epochs=warmup_epochs, 
        test_loss = avg_test_loss
    )

    np.save('WDN_fac' + str(fac) + tag, np.vstack( (training_hist, val_hist) ))