####compare two python denoisers

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import LangsethData
import CNN_UNet_v2
import WaveDecomNet_Langseth as wdn
from torch import stft, istft

def normalize_and_extract_amplitudes(traces):
    # Compute the amplitude of each channel (max absolute value)
    amplitudes = torch.max(torch.abs(traces), dim=2, keepdim=True)[0]  # Shape: (batch, channel, 1)
    normalized_traces = traces / amplitudes  # Normalize to unit amplitude
    return normalized_traces, amplitudes.squeeze()

def stft_transform_and_normalize(traces, n_fft=64, hop_length=16):
    # Use a Hanning window
    window = torch.hann_window(n_fft, device=traces.device)
    
    # Compute the Short-Time Fourier Transform (STFT) for each channel
    stft_results = []
    amplitude_spectrograms = []

    for ch in range(traces.shape[1]):
        stft_output = torch.stft(
            traces[:, ch, :], 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window, 
            return_complex=True
        )
        
        # Separate real and imaginary parts
        real_part = stft_output.real
        imag_part = stft_output.imag

        # Compute amplitude for normalization separately for real and imaginary components
        amplitude_real = torch.amax(torch.abs(real_part), dim=(1, 2), keepdim=True)
        amplitude_imag = torch.amax(torch.abs(imag_part), dim=(1, 2), keepdim=True)

        # Normalize each component separately
        normalized_real = real_part / amplitude_real
        normalized_imag = imag_part / amplitude_imag

        # Combine the normalized components back into a complex tensor
        normalized_stft = torch.complex(normalized_real, normalized_imag)

        # Append the normalized STFT output and the amplitudes for the current channel
        stft_results.append(normalized_stft)
        amplitude_spectrograms.extend([amplitude_real.squeeze(), amplitude_imag.squeeze()])

    # Stack the real and imaginary parts along the channel dimension
    stft_collated = torch.cat([torch.stack([s.real, s.imag], dim=1) for s in stft_results], dim=1)

    return stft_collated, amplitude_spectrograms

def istft_reconstruct_and_rescale(stft_collated, amplitude_spectrograms, n_fft=64, hop_length=16):
    # Use a Hanning window
    window = torch.hann_window(n_fft, device=stft_collated.device)
    
    # Initialize a list to store reconstructed traces
    reconstructed_traces = []
    batch_size, _, freq_bins, time_frames = stft_collated.shape

    # Iterate over each original channel (there are 3 channels, each with separate real and imaginary parts)
    for ch in range(3):
        # Extract the real and imaginary parts for this channel
        real_part = stft_collated[:, ch * 2, :, :]
        imag_part = stft_collated[:, ch * 2 + 1, :, :]

        # Retrieve and reapply the corresponding amplitudes, using broadcasting
        amplitude_real = amplitude_spectrograms[ch * 2].view(batch_size, 1, 1)  # Reshape to (batch, 1, 1) for broadcasting
        amplitude_imag = amplitude_spectrograms[ch * 2 + 1].view(batch_size, 1, 1)  # Reshape to (batch, 1, 1) for broadcasting

        # Reapply amplitudes to the real and imaginary parts
        real_part *= amplitude_real  # Broadcasted multiplication
        imag_part *= amplitude_imag  # Broadcasted multiplication

        # Recombine the real and imaginary parts into a complex STFT tensor
        complex_stft = torch.complex(real_part, imag_part)

        # Perform the inverse STFT
        trace_reconstructed = torch.istft(
            complex_stft, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window
        )
        reconstructed_traces.append(trace_reconstructed)

    # Stack back into (batch, channel, length) to return the reconstructed traces
    reconstructed_traces = torch.stack(reconstructed_traces, dim=1)
    return reconstructed_traces

def normalize_batch(batch_tensor):
    """
    Demeans and normalizes each individual tensor in a batch to have a mean of 0 and 
    standard deviation of 1 along the last dimension (length).
    
    Args:
        batch_tensor (torch.Tensor): Tensor of shape (batch, channel, length)
    
    Returns:
        torch.Tensor: Normalized tensor with mean 0 and std 1 along the last dimension.
    """
    # Calculate mean and std along the last dimension (length)
    mean = batch_tensor.mean(dim=-1, keepdim=True)
    std = batch_tensor.std(dim=-1, keepdim=True)
    
    # Avoid division by zero by setting a minimum std value
    std[std == 0] = 1
    
    # Normalize by subtracting mean and dividing by std
    normalized_tensor = (batch_tensor - mean) / std
    
    return normalized_tensor

def plot_traces(trace_combined, trace_clean, denoised_1, denoised_2, original_snr, save_path, idx):
    """
    Function to plot the traces for each test sample in the batch, separately for each channel.

    Args:
        trace_combined (torch.Tensor): Tensor of shape (batch, channel, length) for combined noisy traces.
        trace_clean (torch.Tensor): Tensor of shape (batch, channel, length) for clean traces.
        denoised_1 (torch.Tensor): Tensor of shape (batch, channel, length) for denoised traces from the first model.
        denoised_2 (torch.Tensor): Tensor of shape (batch, channel, length) for denoised traces from the second model.
        original_snr (torch.Tensor): Tensor of shape (batch, channel) for original SNR values.
        save_path (str): Path to save the plots.
        idx (int): Batch index to use as a prefix for filenames.
    
    Note:
        Assumes a sample rate of 200 Hz and plots each channel separately.
    """
    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    
    sample_rate = 200
    time = torch.arange(trace_combined.shape[2]) / sample_rate  # Time axis in seconds

    # Model names (replace with the actual names you provide)
    model_name_1 = "T23, 16 filters, 3C"
    model_name_2 = "Y22, 16 filter, 3C"

    # Loop over each sample in the batch
    for i in range(trace_combined.shape[0]):
        for ch in range(trace_combined.shape[1]):
            # Create a new figure for each channel
            plt.figure(figsize=(12, 10))
            
            # Extract the traces for the current sample and channel
            trace_comb = trace_combined[i, ch, :].cpu().numpy()
            trace_cln = trace_clean[i, ch, :].cpu().numpy()
            denoise_1 = denoised_1[i, ch, :].cpu().numpy()
            denoise_2 = denoised_2[i, ch, :].cpu().numpy()
            snr_value = original_snr[i, ch].item()

            # Calculate the explained variance for both models using PyTorch
            misfit_1 = trace_cln - denoise_1
            misfit_2 = trace_cln - denoise_2
            var_clean = np.var(trace_cln)

            explained_variance_1 = 1 - (np.var(misfit_1) / var_clean).item()
            explained_variance_2 = 1 - (np.var(misfit_2) / var_clean).item()
            
            # Define channel name
            channel_name = "Vertical" if ch == 0 else f"Horizontal {ch}"

            # Top panel: Combined noisy trace with SNR in the title
            plt.subplot(3, 1, 1)
            plt.plot(time.numpy(), trace_comb, color='blue')
            plt.title(f"{channel_name} - Combined Noisy Trace (Original SNR: {snr_value:.2f} dB)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")

            # Middle panel: Denoised trace from first model
            plt.subplot(3, 1, 2)
            plt.plot(time.numpy(), trace_cln, color='black', label="True Clean Signal")
            plt.plot(time.numpy(), denoise_1, color='red', label=model_name_1)
            plt.title(f"{channel_name} - {model_name_1} Denoised Trace (Explained Variance: {explained_variance_1:.2f})")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()

            # Bottom panel: Denoised trace from second model
            plt.subplot(3, 1, 3)
            plt.plot(time.numpy(), trace_cln, color='black', label="True Clean Signal")
            plt.plot(time.numpy(), denoise_2, color='red', label=model_name_2)
            plt.title(f"{channel_name} - {model_name_2} Denoised Trace (Explained Variance: {explained_variance_2:.2f})")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()

            plt.tight_layout()
            # Save the plot with the format idx_channel_i (e.g., 0_Vertical_1.png, 0_Horizontal_2_1.png)
            plt.savefig(f"{save_path}/trace_plot_{idx}_{channel_name}_{i + 1}.png")
            plt.close()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate 3-panel plots using two denoising models")
    parser.add_argument("--fac", type=int, default=1, help="Factor for the model")
    parser.add_argument("--tag1", type=str, required=True, help="Tag for the first model")
    parser.add_argument("--tag2", type=str, required=True, help="Tag for the second model")
    parser.add_argument("--channels", type=str, default="vertical", help="Channels: vertical (default), 3C, etc.")
    parser.add_argument("--folder", type=str, default="./", help="Folder for model checkpoints")
    parser.add_argument("--nplot", type=int, default=5, help="How many sets of 256 to plot")
    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_cores = os.cpu_count()
    num_workers = int(min(22, num_cores-1))

    # Load the dataset
    nlen, large_batch_size, nperseg, hop_length = 2048, 1, 64, 16
    _, _, dataloader_test_stft = LangsethData.create_dataloaders(
        '../LangsethDatasetSignal_v4', '../LangsethDatasetNoise_v4', nlen, large_batch_size,
        mode='stft', channels=args.channels, num_workers=num_workers, nperseg=nperseg, hop_length=hop_length,
        normalization_type="1C"
    )

    # Set up Model 1
    model_1 = CNN_UNet_v2.UNetv1(
        drop=0.0, ncomp=(1 if args.channels == 'vertical' else 3), fac=args.fac,
        use_skip_attention=False, use_bottleneck_attention=False
    )

    model_1.load_state_dict(torch.load(f"{args.folder}/{args.tag1}.pt", map_location=device))
    model_1.to(device).eval()

    # Set up Model 2
    model_2 = wdn.UNetWithBottleneck(
        in_channels=(1 if args.channels == 'vertical' else 3), out_channels=(1 if args.channels == 'vertical' else 3),
        bottleneck_type='attention', fac=args.fac, dropout_prob=0.0, use_attention_on_skips=False, activation_fn=nn.ELU
    )
    model_2.load_state_dict(torch.load(f"{args.folder}/{args.tag2}.pt", map_location=device))
    model_2.to(device).eval()

    # Directory to save plots
    save_path = "./comparison_plots"
    os.makedirs(save_path, exist_ok=True)

    count = 0

    # Iterate over the test set and generate plots
    with torch.no_grad():
        for idx, batch in enumerate(dataloader_test_stft):
            snr = batch[3]
            trace_combined = batch[5].to(device)  # (batch, channel, length)

            # Normalize time-domain traces and extract amplitudes
            normalized_traces_time, amplitudes_time = normalize_and_extract_amplitudes(trace_combined)

            # Prepare STFT input, normalize, and extract spectrogram amplitudes
            stft_input, amplitudes_spectrogram = stft_transform_and_normalize(trace_combined)

            # Run both models
            denoised_1 = model_1(stft_input)
            denoised_2 = model_2(normalized_traces_time)[0]

            # Reapply time-domain amplitudes to model 2 output
            denoised_2 *= amplitudes_time.unsqueeze(2)  # Model 2 works entirely in the time domain

            # For model 1, reconstruct the STFT back to the time domain and reapply amplitudes
            denoised_1_reconstructed = istft_reconstruct_and_rescale(denoised_1, amplitudes_spectrogram)

            # Extract clean and noisy traces for plotting
            trace_clean = batch[4].squeeze()

            # Plot traces
            plot_traces(trace_combined, trace_clean, denoised_1_reconstructed, denoised_2, snr, save_path, idx)

            count += 1
            if count == args.nplot:
                break

    print("Completed generating plots.")
