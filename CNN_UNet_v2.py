import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
#import pywt
import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
from scipy import signal
import pandas as pd
import h5py
import os
import LangsethData
import argparse
import time
import warnings
import math

warnings.filterwarnings('ignore')

class FullyConnectedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(FullyConnectedAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: [batch, channels, freq, time]
        b, c, f, t = x.shape

        # Flatten spatial dimensions (freq x time) into one sequence dimension
        x = x.reshape(b, c, f * t).permute(0, 2, 1)  # Shape: [batch, seq, channels]

        # Apply multi-head attention on the flattened sequence
        x, _ = self.attention(x, x, x)

        # Reshape back to the original spatial dimensions
        x = x.permute(0, 2, 1).reshape(b, c, f, t)  # Shape: [batch, channels, freq, time]
        return x

class UNetv1(nn.Module):
    def __init__(self, drop=0.1, ncomp=1, fac=1, use_bottleneck_attention=False, use_skip_attention=False):
        super(UNetv1, self).__init__()
        
        self.use_bottleneck_attention = use_bottleneck_attention
        self.use_skip_attention = use_skip_attention

        if ncomp == 1:
            in_channels = 2
        elif ncomp == 3:
            in_channels = 6

        # Downsampling layers (convolutions)
        self.conv1 = nn.Conv2d(in_channels, 8 * fac, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8 * fac)
        
        self.conv1b = nn.Conv2d(8 * fac, 8 * fac, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(8 * fac)
        self.drop1b = nn.Dropout(drop)
        
        self.conv2 = nn.Conv2d(8 * fac, 8 * fac, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(8 * fac)
        
        self.conv2b = nn.Conv2d(8 * fac, 16 * fac, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(16 * fac)
        self.drop2b = nn.Dropout(drop)
        
        self.conv3 = nn.Conv2d(16 * fac, 16 * fac, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16 * fac)
        
        self.conv3b = nn.Conv2d(16 * fac, 32 * fac, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(32 * fac)
        self.drop3b = nn.Dropout(drop)
        
        self.conv4 = nn.Conv2d(32 * fac, 32 * fac, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32 * fac)
        
        self.conv4b = nn.Conv2d(32 * fac, 64 * fac, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(64 * fac)
        self.drop4b = nn.Dropout(drop)
        
        self.conv5 = nn.Conv2d(64 * fac, 64 * fac, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64 * fac)
        
        self.conv5b = nn.Conv2d(64 * fac, 128 * fac, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm2d(128 * fac)
        self.drop5b = nn.Dropout(drop)

        # Bottleneck attention layer
        if self.use_bottleneck_attention:
            self.bottleneck_attention = FullyConnectedAttention(embed_dim=128 * fac, num_heads=4*fac)

        # Attention for skip connections
        if self.use_skip_attention:
            self.skip_attention_layers = nn.ModuleList([
                FullyConnectedAttention(embed_dim=8 * fac, num_heads=2*fac),
                FullyConnectedAttention(embed_dim=16 * fac, num_heads=2*fac),
                FullyConnectedAttention(embed_dim=32 * fac, num_heads=2*fac),
                FullyConnectedAttention(embed_dim=64 * fac, num_heads=4*fac)
            ])

        # Upsampling layers (transposed convolutions)
        self.upconv4 = nn.ConvTranspose2d(128 * fac, 64 * fac, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4u = nn.BatchNorm2d(64 * fac)

        self.conv4u = nn.Conv2d(128 * fac, 64 * fac, kernel_size=3, padding=1)
        self.bn4u2 = nn.BatchNorm2d(64 * fac)

        self.upconv3 = nn.ConvTranspose2d(64 * fac, 32 * fac, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3u = nn.BatchNorm2d(32 * fac)

        self.conv3u = nn.Conv2d(64 * fac, 32 * fac, kernel_size=3, padding=1)
        self.bn3u2 = nn.BatchNorm2d(32 * fac)

        self.upconv2 = nn.ConvTranspose2d(32 * fac, 16 * fac, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2u = nn.BatchNorm2d(16 * fac)

        self.conv2u = nn.Conv2d(32 * fac, 16 * fac, kernel_size=3, padding=1)
        self.bn2u2 = nn.BatchNorm2d(16 * fac)

        self.upconv1 = nn.ConvTranspose2d(16 * fac, 8 * fac, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1u = nn.BatchNorm2d(8 * fac)

        self.conv1u = nn.Conv2d(16 * fac, 8 * fac, kernel_size=3, padding=1)
        self.bn1u2 = nn.BatchNorm2d(8 * fac)

        if ncomp == 1:
            self.final = nn.Conv2d(8 * fac, 2, kernel_size=1)
        else:
            self.final = nn.Conv2d(8 * fac, 6, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoding path
        enc1 = F.elu(self.bn1(self.conv1(x)))
        enc1b = self.drop1b(F.elu(self.bn1b(self.conv1b(enc1))))
        
        enc2 = F.elu(self.bn2(self.conv2(enc1b)))
        enc2b = self.drop2b(F.elu(self.bn2b(self.conv2b(enc2))))
        
        enc3 = F.elu(self.bn3(self.conv3(enc2b)))
        enc3b = self.drop3b(F.elu(self.bn3b(self.conv3b(enc3))))
        
        enc4 = F.elu(self.bn4(self.conv4(enc3b)))
        enc4b = self.drop4b(F.elu(self.bn4b(self.conv4b(enc4))))
        
        enc5 = F.elu(self.bn5(self.conv5(enc4b)))
        enc5b = self.drop5b(F.elu(self.bn5b(self.conv5b(enc5))))

        # Apply attention at the bottleneck if enabled
        if self.use_bottleneck_attention:
            enc5b = self.bottleneck_attention(enc5b)

        # Decoding path with optional skip connection attention
        dec4 = F.elu(self.bn4u(self.upconv4(enc5b)))
        if self.use_skip_attention:
            enc4b = self.skip_attention_layers[3](enc4b)
        dec4 = self.crop_and_concat(dec4, enc4b)
        dec4 = F.elu(self.bn4u2(self.conv4u(dec4)))
        
        dec3 = F.elu(self.bn3u(self.upconv3(dec4)))
        if self.use_skip_attention:
            enc3b = self.skip_attention_layers[2](enc3b)
        dec3 = self.crop_and_concat(dec3, enc3b)
        dec3 = F.elu(self.bn3u2(self.conv3u(dec3)))
        
        dec2 = F.elu(self.bn2u(self.upconv2(dec3)))
        if self.use_skip_attention:
            enc2b = self.skip_attention_layers[1](enc2b)
        dec2 = self.crop_and_concat(dec2, enc2b)
        dec2 = F.elu(self.bn2u2(self.conv2u(dec2)))
        
        dec1 = F.elu(self.bn1u(self.upconv1(dec2)))
        if self.use_skip_attention:
            enc1b = self.skip_attention_layers[0](enc1b)
        dec1 = self.crop_and_concat(dec1, enc1b)
        dec1 = F.elu(self.bn1u2(self.conv1u(dec1)))

        out = self.final(dec1)
        return out

    def crop_and_concat(self, upsampled, bypass):
        diffY = bypass.size(2) - upsampled.size(2)
        diffX = bypass.size(3) - upsampled.size(3)

        if diffY > 0 or diffX > 0:
            bypass = F.pad(bypass, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        elif diffY < 0 or diffX < 0:
            upsampled = F.pad(upsampled, [diffX // 2, (diffX - diffX // 2), diffY // 2, (diffY - diffY // 2)])

        return torch.cat([upsampled, bypass], dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_validation_samples(valbatch, model, nperseg, hop_length, tag, sample_rate=200, cmap='PuRd'):
    num_examples = 10  # Number of examples to plot

    # Frequency values for STFT
    f = np.fft.rfftfreq(nperseg, d=1./sample_rate)  # Frequency values for STFT

    # Time Domain Figure
    plt.figure(figsize=(12, num_examples * 3))
    for i in range(num_examples):
        # Get input data and mask for this example
        inputs = valbatch[1][i].unsqueeze(0)  # Shape: (1, 2*num_channels, freq, seq)
        mask = valbatch[0][i].unsqueeze(0)  # Shape: (1, num_channels, freq, seq)

        # Calculate masked and unmasked traces using the function
        trace_masked_stft, trace_masked, trace_unmasked = calculate_signal_from_mask(inputs, mask, nperseg, hop_length)

        if len(trace_masked.shape) < 3:
            trace_masked = trace_masked.unsqueeze(0)
            trace_unmasked = trace_unmasked.unsqueeze(0)

        # Model output
        out = model(inputs).squeeze().detach()
        stft_pred = out[0, :, :] + out[1, :, :] * 1j
        trace_pred = torch.istft(stft_pred, n_fft=nperseg, hop_length=hop_length, window=torch.hann_window(nperseg))

        trace_unmasked = trace_unmasked[0, 0, :].squeeze()
        trace_masked = trace_masked[0, 0, :].squeeze()
        trace_pred = trace_pred.squeeze()

        # Generate time arrays for plotting
        t_trace = np.arange(0, trace_unmasked.shape[-1] / sample_rate, 1 / sample_rate)
        max_amplitude = torch.max(torch.abs(trace_unmasked)).item()
        trace_masked = trace_masked / max_amplitude
        trace_unmasked = trace_unmasked / max_amplitude
        trace_pred = trace_pred / max_amplitude

        # Plot Input Trace Alone
        plt.subplot(num_examples, 2, 2 * i + 1)
        plt.plot(t_trace, trace_unmasked)  # Unmasked trace
        plt.ylim([-1.1, 1.1])
        if i == 0:
            plt.title("Input Data")
        plt.xlabel('Time (s)')
        plt.gca().axes.yaxis.set_visible(False)

        # Plot Recovered Trace vs Target Trace
        plt.subplot(num_examples, 2, 2 * i + 2)
        plt.plot(t_trace, trace_masked, 'k', label='Target')
        plt.plot(t_trace, trace_pred, 'r', label='Recovered')
        plt.ylim([-1.1, 1.1])
        if i == 0:
            plt.title("Recovered vs Target Data")
            plt.legend()
        plt.xlabel('Time (s)')
        plt.gca().axes.yaxis.set_visible(False)

    # Save and close time-domain figure
    plt.tight_layout()
    plt.savefig('T23_Validation_samples_time_domain' + tag + '.pdf')
    plt.close()

    # STFT Figure
    plt.figure(figsize=(18, num_examples * 3))
    for i in range(num_examples):
        # Normalize the spectrograms for clear plotting

        # Get input data and mask for this example
        inputs = valbatch[1][i].unsqueeze(0)  # Shape: (1, 2*num_channels, freq, seq)
        mask = valbatch[0][i].unsqueeze(0)  # Shape: (1, num_channels, freq, seq)

        # Calculate masked and unmasked traces using the function
        trace_masked_stft, _, _ = calculate_signal_from_mask(inputs, mask)

        # Model output
        out = model(inputs).squeeze().detach()

        norm_trace_in_stft = torch.sqrt(inputs[0][0, :, :]**2 + inputs[0][1, :, :]**2)
        norm_trace_in_stft = norm_trace_in_stft/norm_trace_in_stft.max()

        norm_trace_true_stft = torch.sqrt(trace_masked_stft[0, 0, :, :]**2 + trace_masked_stft[0, 1, :, :]**2)
        norm_trace_true_stft = norm_trace_true_stft/norm_trace_true_stft.max()

        norm_trace_pred_stft = torch.sqrt(out[0, :, :]**2 + out[1, :, :]**2)
        norm_trace_pred_stft = norm_trace_pred_stft/norm_trace_pred_stft.max()

        # Generate time array for STFT
        t_stft = np.linspace(0, len(t_trace) / sample_rate, num=norm_trace_in_stft.shape[-1], endpoint=True)

        # Plot Input STFT Amplitude
        plt.subplot(num_examples, 3, 3 * i + 1)
        plt.pcolormesh(t_stft, f, torch.log(norm_trace_in_stft), cmap=cmap, shading='auto', vmin=-3, vmax=0)
        if i == 0:
            plt.title("Input STFT")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # Plot True STFT Amplitude
        plt.subplot(num_examples, 3, 3 * i + 2)
        plt.pcolormesh(t_stft, f, torch.log(norm_trace_true_stft), cmap=cmap, shading='auto', vmin=-3, vmax=0)
        if i == 0:
            plt.title("True STFT")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # Plot Recovered STFT Amplitude
        plt.subplot(num_examples, 3, 3 * i + 3)
        plt.pcolormesh(t_stft, f, torch.log(norm_trace_pred_stft), cmap=cmap, shading='auto', vmin=-3, vmax=0)
        if i == 0:
            plt.title("Recovered STFT")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

    # Save and close STFT figure
    plt.tight_layout()
    plt.savefig('T23_Validation_samples_stft' + tag + '.png') #pdf renders badly
    plt.close()

def calculate_signal_from_mask(batch, mask, nperseg=-1, hop_length=-1):
    """
    Function to calculate the masked STFT and ISTFT of each trace in the batch using a provided mask.
    If nperseg > 0, also returns the unmasked ISTFT traces.

    Parameters:
    - batch: complex STFT of input data with real and imaginary parts split. Shape is (batch x 2*num_channels x freq x seq).
    - mask: real-valued mask to apply to the batch. Shape is (batch x num_channels x freq x seq).
    - nperseg: length of each segment in the STFT.
    - hop_length: hop length for the ISTFT.

    Returns:
    - trace_masked_stft: Tensor of masked STFTs for the batch, with real and imaginary parts separated (batch x 2*num_channels x freq x seq).
    - trace_masked: Tensor of masked ISTFTs for the batch (batch x num_channels x seq), if nperseg > 0; otherwise, None.
    - trace_unmasked: Tensor of unmasked ISTFTs for the batch (batch x num_channels x seq), if nperseg > 0; otherwise, None.
    """
    num_channels = mask.shape[1]  # Determine the number of input channels from the mask
    masked_stfts = []
    masked_traces = []
    unmasked_traces = []

    # Iterate through each channel to apply the mask to real and imaginary parts
    for ch in range(num_channels):
        # Get real and imaginary parts for this channel

        real_part = batch[:, 2*ch, :, :]
        imag_part = batch[:, 2*ch + 1, :, :]
        
        # Combine real and imaginary into a complex tensor
        trace_stft = real_part + imag_part * 1j  # (batch x freq x seq)
        
        # Apply the mask to the complex STFT
        trace_masked_stft = trace_stft * mask[:, ch, :, :]  # (batch x freq x seq)
        
        # Separate masked STFT into real and imaginary parts and add to the list
        masked_stfts.append(trace_masked_stft.real)
        masked_stfts.append(trace_masked_stft.imag)

        if nperseg > 0:

            # Compute ISTFT for masked trace
            trace_masked = torch.stack([
                torch.istft(trace_masked_stft[i].cpu(), n_fft=nperseg, hop_length=hop_length, 
                            window=torch.hann_window(nperseg)).detach()
                for i in range(batch.shape[0])
            ], dim=0)  # (batch x seq)
            masked_traces.append(trace_masked)
            
            # Compute ISTFT for unmasked trace
            trace_unmasked = torch.stack([
                torch.istft(trace_stft[i].cpu(), n_fft=nperseg, hop_length=hop_length, 
                            window=torch.hann_window(nperseg)).detach()
                for i in range(batch.shape[0])
            ], dim=0)  # (batch x seq)
            unmasked_traces.append(trace_unmasked)

    # Stack all masked STFTs along the channel dimension (batch x 2*num_channels x freq x seq)
    trace_masked_stft = torch.stack(masked_stfts, dim=1)
    
    # Stack ISTFT results if computed (batch x num_channels x seq)
    trace_masked = torch.stack(masked_traces, dim=1) if nperseg > 0 else None
    trace_unmasked = torch.stack(unmasked_traces, dim=1) if nperseg > 0 else None
    
    return trace_masked_stft, trace_masked, trace_unmasked

def time_domain_output(out, nperseg=64, hop_length=16):

    traces = []

    num_channels = out.shape[1]

    for ch in range(int(num_channels/2)):

        real_part = out[:, 2*ch, :, :]
        imag_part = out[:, 2*ch + 1, :, :]
        
        # Combine real and imaginary into a complex tensor
        trace_stft = real_part + imag_part * 1j  # (batch x freq x seq)

        # Compute ISTFT for masked trace
        tr = torch.stack([
            torch.istft(trace_stft[i], n_fft=nperseg, hop_length=hop_length, 
                        window=torch.hann_window(nperseg)).detach()
            for i in range(out.shape[0])
        ], dim=0)  # (batch x seq)
        traces.append(tr)

    return torch.stack(traces, dim=1)

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
    num_epochs, 
    mini_batch_size, 
    device, 
    val_sweeps, 
    tag, 
    nperseg, 
    hop_length, 
    learning_rate, 
    warmup_epochs=5, 
    warmup_factor=0.1
):
    # Initialize training history lists
    training_hist = np.array([])
    val_hist = np.array([])
    lr_hist = np.array([])

    for batch in dataloader_valid:  # for plotting
        valbatch = batch
        break

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0  # Counter for batches

        # Timing accumulators
        total_load_time = 0.0
        total_transfer_time = 0.0
        total_runtime = 0.0

        # Adjust learning rate for warm-up
        if epoch <= warmup_epochs:
            warmup_lr = learning_rate * (warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs))
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        print(f"-----> Epoch# {epoch + 1}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        model.train()
        start_time = time.time()

        for large_batch in dataloader_train:
            # Track load time
            loadtime = time.time() - start_time
            total_load_time += loadtime
            start_time = time.time()

            # Transfer large batch to GPU
            inputs_large = large_batch[1].to(device, non_blocking=True)
            mask_large = large_batch[0].to(device, non_blocking=True)

            # Calculate target for large batch
            target_large = calculate_signal_from_mask(inputs_large, mask_large)[0]

            # Track transfer time
            transtime = time.time() - start_time
            total_transfer_time += transtime
            start_time = time.time()

            # Divide the large batch into mini-batches and train
            for i in range(0, inputs_large.shape[0], mini_batch_size):
                # Skip last if irregularly sized at the end
                if inputs_large.shape[0] < i + mini_batch_size:
                    break

                mini_inputs = inputs_large[i:i + mini_batch_size]
                mini_target = target_large[i:i + mini_batch_size]

                # Forward pass and optimization
                optimizer.zero_grad()
                out = model(mini_inputs)
                loss = criterion(out, mini_target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

                # Track runtime
                runtime = time.time() - start_time
                total_runtime += runtime
                start_time = time.time()

        avg_training_loss = math.sqrt(running_loss / batch_count)

        # Epoch timing summary
        epoch_total_time = total_load_time + total_transfer_time + total_runtime
        load_time_percent = (total_load_time / epoch_total_time) * 100
        transfer_time_percent = (total_transfer_time / epoch_total_time) * 100
        runtime_percent = (total_runtime / epoch_total_time) * 100

        print(f"Epoch {epoch + 1}/{num_epochs} - Total Time: {epoch_total_time:.2f} seconds")
        print(f"  Load Time: {total_load_time:.2f} seconds ({load_time_percent:.2f}%)")
        print(f"  Transfer Time: {total_transfer_time:.2f} seconds ({transfer_time_percent:.2f}%)")
        print(f"  Runtime (model training): {total_runtime:.2f} seconds ({runtime_percent:.2f}%)")
        print(f"  Average Training Loss: {avg_training_loss:.5f}")
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
                    mask_large = large_batch[0].to(device, non_blocking=True)

                    # Calculate target for large batch
                    target_large = calculate_signal_from_mask(inputs_large, mask_large)[0]

                    # Divide the large batch into mini-batches and perform validation
                    for i in range(0, inputs_large.shape[0], mini_batch_size):
                        # Skip last if irregularly sized at the end
                        if inputs_large.shape[0] < i + mini_batch_size:
                            break

                        mini_inputs = inputs_large[i:i + mini_batch_size]
                        mini_target = target_large[i:i + mini_batch_size]

                        # Forward pass and loss calculation
                        out = model(mini_inputs)
                        loss = criterion(out, mini_target)
                        val_loss += loss.item()
                        val_batch_count += 1

        avg_val_loss = math.sqrt(val_loss / val_batch_count)
        print(f'--> Validation loss: {avg_val_loss:.5f}')

        # Plot validation samples
        model.to("cpu")
        plot_validation_samples(valbatch, model, nperseg, hop_length, tag)
        model.to(device)

        # Append losses to history
        training_hist = np.append(training_hist, avg_training_loss)
        val_hist = np.append(val_hist, avg_val_loss)
        lr_hist = np.append(lr_hist, optimizer.param_groups[0]['lr'])

        # Plot training history
        plot_and_save_training_history(training_hist, val_hist, "model" + tag, lr_hist, warmup_epochs)

        # Save current model
        torch.save(model.state_dict(), 'model_' + tag + '.pt')
        torch.cuda.empty_cache()  # Clear CUDA memory

        # Step the scheduler based on validation loss
        if epoch >= warmup_epochs:
            # Early stopping condition
            scheduler.step(avg_val_loss)
            if optimizer.param_groups[0]['lr'] <= learning_rate / 128:
                print('Learning rate reduced below minimum')
                break

    return model, training_hist, val_hist, lr_hist

if __name__ == "__main__":

    # Hyperparameters
    model_type = 'UNet'
    model_v = 'v2'

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Training script for the denoiser model")

    # Add existing arguments
    parser.add_argument("--fac", type=str, default=1, help="Factor for the model")
    parser.add_argument("--tag", type=str, default='ThomasDenoiser', help="Tag for the model run")
    parser.add_argument("--channels", type=str, default='vertical', help="What data to include:all, vertical (default), 3C, pressure (which is pressure and vertical together, horizontals")
    # Add the --test flag, which sets test_mode to True if passed
    parser.add_argument("--test", action='store_true', help="Enable test mode")
    parser.add_argument("--bottleneck_attention", action='store_true', help="Enable attention at the bottleneck")
    parser.add_argument("--skip_attention", action='store_true', help="Enable attention on the skip connections")

    # Parse the arguments
    args = parser.parse_args()

    # Assign parsed values to variables
    fac = int(args.fac)
    tag = args.tag
    test_mode = args.test  # True if --test is passed, otherwise False
    channels = args.channels  # True if --test is passed, otherwise False
    use_bottleneck_attention = args.bottleneck_attention
    use_skip_attention = args.skip_attention

    # Define large and mini batch sizes
    large_batch_size = 1
    mini_batch_size = 128  # Number of mini-batches within each large batch
    learning_rate = 0.001
    num_epochs = 200  # Default to 3 epochs
    sr = 200
    #fac = 2# model size, now passed on command line. 
    eps = 1e-9 # epsilon
    drop=0.0 # model drop rate
    nlen=2048# window length
    nperseg=64 # param for STFT
    hop_length=16
    norm_input=True # leave this alone
    cmap='PuRd'

    val_sweeps = 5
    test_sweeps = 25

    # Warm-up parameters
    warmup_epochs = 5  # Number of epochs for warm-up
    warmup_factor = 0.1  # Initial factor for learning rate warm-up

    np.random.seed(5) #same seed as used during training for an in-house check
    torch.manual_seed(5)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device("cpu")

    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    num_cores = os.cpu_count()
    num_workers = int(min(22, num_cores-1))
    
    if channels == 'all':
        model = UNebtv1(drop = drop, ncomp = 4, fac = fac, use_skip_attention=use_skip_attention, use_bottleneck_attention=use_bottleneck_attention) #pytorch version of amanda's tensorflow UNet
    elif channels == 'vertical':
        model = UNetv1(drop = drop, ncomp = 1, fac = fac, use_skip_attention=use_skip_attention, use_bottleneck_attention=use_bottleneck_attention) #pytorch version of amanda's tensorflow UNet
    elif channels == 'pressure':
        model = UNetv1(drop = drop, ncomp = 2, fac = fac, use_skip_attention=use_skip_attention, use_bottleneck_attention=use_bottleneck_attention) #pytorch version of amanda's tensorflow UNet
    elif channels == 'horizontals':
        model = UNetv1(drop = drop, ncomp = 2, fac = fac, use_skip_attention=use_skip_attention, use_bottleneck_attention=use_bottleneck_attention) #pytorch version of amanda's tensorflow UNet
    elif channels == '3C':
        model = UNetv1(drop = drop, ncomp = 3, fac = fac, use_skip_attention=use_skip_attention, use_bottleneck_attention=use_bottleneck_attention) #pytorch version of amanda's tensorflow UNet

    print("CNN-UNet" + model_v + " initialized with " + str(count_parameters(model)) + " trainable parameters")
    model = model.float()
    model.to(device)

    print('Constructing the dataset')
    dataloader_train, dataloader_valid, dataloader_test = LangsethData.create_dataloaders(
        '../LangsethDatasetSignal_v4', '../LangsethDatasetNoise_v4', nlen, large_batch_size, mode='stft', channels=channels, 
        num_workers=num_workers, nperseg=nperseg, hop_length=hop_length, test_mode=test_mode, normalization_type="1C")

    criterion = nn.MSELoss()  # You could also use nn.L1Loss

    # Initialize history arrays and define optimizer and scheduler
    training_hist = np.array([])
    val_hist = np.array([])
    lr_hist = np.array([])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.25, patience=10, eps=1e-8, threshold=0.01
    )

    torch.backends.cudnn.benchmark = True

    # Call the training loop
    model, training_hist, val_hist, lr_hist = training_loop_with_warmup(
        model, 
        optimizer, 
        criterion, 
        scheduler, 
        dataloader_train, 
        dataloader_valid, 
        num_epochs, 
        mini_batch_size, 
        device, 
        val_sweeps, 
        tag, 
        nperseg, 
        hop_length, 
        learning_rate, 
        warmup_epochs=warmup_epochs,
        warmup_factor=warmup_factor
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
                mask_large = large_batch[0].to(device, non_blocking=True)
                
                # Calculate target for large batch
                target_large = calculate_signal_from_mask(inputs_large, mask_large)[0]

                # Divide the large batch into mini-batches and perform validation
                for i in range(0, inputs_large.shape[0], mini_batch_size):
                    
                    # Skip last if irregularly sized at the end
                    if inputs_large.shape[0] < i + mini_batch_size:
                        break

                    mini_inputs = inputs_large[i:i + mini_batch_size]
                    mini_target = target_large[i:i + mini_batch_size]

                    # Forward pass and loss calculation
                    out = model(mini_inputs)
                    loss = criterion(out, mini_target)
                    test_loss += loss.item()
                    test_batch_count += 1

    avg_test_loss = math.sqrt(test_loss / test_batch_count)
    print(f'--> Test loss: {avg_test_loss:.4f}')
    # Plot training history
    plot_and_save_training_history(
        training_hist, 
        val_hist, 
        "model" + tag, 
        lr_hist, 
        warmup_epochs=warmup_epochs, 
        test_loss = avg_test_loss
    )

    # Save training and validation history
    print('Finished Training')
    np.save('CNN_Unet_' + tag, np.vstack((training_hist, val_hist)))