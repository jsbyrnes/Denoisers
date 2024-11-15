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
import WaveDecomNet_Langseth as wdn

warnings.filterwarnings('ignore')

def compute_metrics_windowed(y, y_tilde, pick_index, window_size=200):
    """
    Compute metrics within a specified window around the pick index for each signal.

    Args:
        y: 1D numpy array of shape (length=2048) representing the original signal.
        y_tilde: 1D numpy array of shape (length=2048) representing the denoised signal.
        pick_index: Integer representing the index of the pick within the signal.
        window_size: Integer specifying the length of the window for computation.

    Returns:
        A tuple of cross-correlation coefficient, L2 norm, and expected variance.
    """
    # Define window range around the pick index
    start = max(0, pick_index - window_size)
    end = min(len(y), pick_index + window_size)

    # Extract the windowed segments
    y_window = y[start:end]
    y_tilde_window = y_tilde[start:end]

    # Compute metrics for the current window
    cross_correlation_coefficient = np.corrcoef(y_window, y_tilde_window)[0, 1]
    l2_norm = np.linalg.norm(y_window - y_tilde_window)
    variance_y = np.var(y_window)
    variance_difference = np.var(y_window - y_tilde_window)
    expected_variance = 1 - (variance_difference / variance_y) if variance_y != 0 else np.nan

    return cross_correlation_coefficient, l2_norm, expected_variance

# Save the results using the provided tag
def save_results(tag, vertical_array, horizontal_array):
    # Create a directory to save the results if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the vertical and horizontal results as numpy files
    np.save(f'results/{tag}_vertical_results.npy', vertical_array)
    np.save(f'results/{tag}_horizontal_results.npy', horizontal_array)

# Function to calculate the mean and 95% confidence interval
def calculate_statistics(data):
    mean = np.mean(data)
    lower_bound = np.percentile(data, 2.5)
    upper_bound = np.percentile(data, 97.5)
    return mean, lower_bound, upper_bound

# Plotting function
def plot_metrics_against_snr(tag, vertical_array, horizontal_array, db_bins):
    # Combine vertical and horizontal results for plotting separately

    bin_edges = np.linspace(-30, 20, 21)
    db_bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]

    # Extract and filter out inf values for vertical metrics
    valid_indices_vertical = np.isfinite(vertical_array).all(axis=1)
    vertical_array = vertical_array[valid_indices_vertical]

    old_snr_vertical = vertical_array[:, 0]
    new_snr_vertical = vertical_array[:, 1]
    ev_vertical = vertical_array[:, 4]  # Only keep the relevant metrics
    #ev_vertical[ev_vertical<0] = 0
    # Extract and filter out inf values for horizontal metrics, if available
    if horizontal_array is not None:
        valid_indices_horizontal = np.isfinite(horizontal_array).all(axis=1)
        horizontal_array = horizontal_array[valid_indices_horizontal]

        old_snr_horizontal = horizontal_array[:, 0]
        new_snr_horizontal = horizontal_array[:, 1]
        ev_horizontal = horizontal_array[:, 4]  # Only keep the relevant metrics
        #ev_horizontal[ev_horizontal<0] = 0

    # Function to bin data and prepare statistics
    def prepare_binned_data(old_snr, ev, new_snr):
        binned_data = {bin_range: [] for bin_range in db_bins}
        bin_means = []

        for i in range(len(old_snr)):
            for bin_range in db_bins:
                if bin_range[0] <= old_snr[i] < bin_range[1]:
                    binned_data[bin_range].append([ev[i], new_snr[i]])
                    break

        # Prepare data for box plots
        ev_data, new_snr_data = [], []
        labels = []

        for bin_range, data in binned_data.items():
            if len(data) > 0:
                data = np.array(data)
                bin_mean = np.mean([bin_range[0], bin_range[1]])
                bin_means.append(bin_mean)
                labels.append(f'{bin_mean:.1f} dB')  # Use the mean value of the bin as the label
                
                # Calculate statistics for each metric
                mean_ev, lower_ev, upper_ev = calculate_statistics(data[:, 0])
                mean_snr, lower_snr, upper_snr = calculate_statistics(data[:, 1])

                # Append data for plotting
                ev_data.append([mean_ev, lower_ev, upper_ev])
                new_snr_data.append([mean_snr, lower_snr, upper_snr])

        return ev_data, new_snr_data, labels

    # Prepare binned data for vertical and horizontal
    ev_data_vertical, new_snr_data_vertical, labels_vertical = prepare_binned_data(
        old_snr_vertical, ev_vertical, new_snr_vertical
    )

    if horizontal_array is not None:
        ev_data_horizontal, new_snr_data_horizontal, labels_horizontal = prepare_binned_data(
            old_snr_horizontal, ev_horizontal, new_snr_horizontal
        )

    # Plotting for vertical
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    metrics_vertical = [ev_data_vertical, new_snr_data_vertical]
    titles_vertical = ['Expected Variance (Vertical)', 'Denoised SNR (Vertical)']

    for i, ax in enumerate(axes):
        data = metrics_vertical[i]
        means = [d[0] for d in data]
        lower_bounds = [d[1] for d in data]
        upper_bounds = [d[2] for d in data]
        
        # Create a box plot-like representation of the mean and 95% CI
        ax.errorbar(range(len(data)), means, yerr=[np.array(means) - np.array(lower_bounds), np.array(upper_bounds) - np.array(means)], fmt='o')
        ax.set_title(titles_vertical[i])
        ax.set_xticks(range(len(labels_vertical)))
        ax.set_xticklabels(labels_vertical, rotation=45)
        ax.set_xlabel('Starting SNR (dB)')
        ax.grid()

    axes[0].set_ylabel('Metric Value')
    plt.tight_layout()
    plt.savefig(f'results/{tag}_vertical_metrics_box_plot.png')
    plt.show()

    # Plotting for horizontal (if available)
    if horizontal_array is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        metrics_horizontal = [ev_data_horizontal, new_snr_data_horizontal]
        titles_horizontal = ['Expected Variance (Horizontal)', 'Denoised SNR (Horizontal)']

        for i, ax in enumerate(axes):
            data = metrics_horizontal[i]
            means = [d[0] for d in data]
            lower_bounds = [d[1] for d in data]
            upper_bounds = [d[2] for d in data]
            
            # Create a box plot-like representation of the mean and 95% CI
            ax.errorbar(range(len(data)), means, yerr=[np.array(means) - np.array(lower_bounds), np.array(upper_bounds) - np.array(means)], fmt='o')
            ax.set_title(titles_horizontal[i])
            ax.set_xticks(range(len(labels_horizontal)))
            ax.set_xticklabels(labels_horizontal, rotation=45)
            ax.set_xlabel('Starting SNR (dB)')
            ax.grid()

        axes[0].set_ylabel('Metric Value')
        plt.tight_layout()
        plt.savefig(f'results/{tag}_horizontal_metrics_box_plot.png')
        plt.show()

# Function to compute SNR entirely from the denoised trace
def compute_snr_from_denoised(trace, pick_index, sample_rate):
    """
    Compute the SNR entirely from the denoised trace, using the same trace for power before and after the pick.

    Args:
        trace: 3D numpy array of shape (batch x channel x length).
        pick_index: 1D numpy array of shape (batch) with the index of the pick for each trace in the batch.
        sample_rate: The sample rate of the signal.

    Returns:
        A numpy array of SNR values in dB with shape (batch x channel).
    """
    # Initialize an array to hold SNR values
    snr_values = np.full((trace.shape[0], trace.shape[1]), -np.inf)
    max_length = trace.shape[2]  # Length of the signals

    for batch in range(trace.shape[0]):
        pick = pick_index[batch]

        post_window_size = min(int(2 * sample_rate), pick - 50, max_length - pick)

        for channel in range(trace.shape[1]):
            signal = trace[batch, channel]

            # Compute power before the pick from the same denoised signal
            power_before = np.sum(
                signal[max(0, pick - 50 - post_window_size):(pick - 50)]**2
            ) / post_window_size

            # Compute power after the pick from the same denoised signal
            power_after = np.sum(
                signal[pick:pick + post_window_size]**2
            ) / post_window_size

            # Compute SNR in dB
            if power_before != 0 and power_after != 0:
                snr_values[batch, channel] = 10 * np.log10(power_after / power_before)

    return snr_values

if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Training script for the denoiser model")

    # Add existing arguments
    parser.add_argument("--fac", type=str, default=1, help="Factor for the model")
    parser.add_argument("--tag", type=str, default='ThomasDenoiser', help="Tag for the model run")
    parser.add_argument("--channels", type=str, default='vertical', help="What data to include:all, vertical (default), 3C, pressure (which is pressure and vertical together, horizontals")
    parser.add_argument("--folder", type=str, default='./', help="Where to find the model you are loading")
    parser.add_argument("--bottleneck", type=str, default='attention', help="Bottleneck to use")

    # Add the --test flag, which sets test_mode to True if passed
    parser.add_argument("--test", action='store_true', help="Enable test mode")
    parser.add_argument("--skip_attention", action='store_true', help="Enable attention on the skip connections")

    # Parse the arguments
    args = parser.parse_args()

    # Assign parsed values to variables
    fac = int(args.fac)
    tag = args.tag
    folder = args.folder
    test_mode = args.test  # True if --test is passed, otherwise False
    channels = args.channels  # True if --test is passed, otherwise False
    bottleneck = args.bottleneck
    use_skip_attention = args.skip_attention

    # Define large and mini batch sizes
    large_batch_size = 1
    mini_batch_size = 128  # Number of mini-batches within each large batch
    learning_rate = 0.001
    num_epochs = 200  # Default to 3 epochs
    sample_rate = 200
    #fac = 2# model size, now passed on command line. 
    eps = 1e-9 # epsilon
    drop=0.0 # model drop rate
    nlen=2048# window length
    nperseg=64 # param for STFT
    hop_length=16
    norm_input=True # leave this alone
    cmap='PuRd'

    db_bins = 10

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
        nchan = 4
    elif channels == 'vertical':
        nchan = 1
    elif channels == 'pressure' or channels == 'horizontals':
        nchan = 2
    elif channels == '3C':
        nchan = 3

    model = wdn.UNetWithBottleneck(
        in_channels=nchan, out_channels=nchan, bottleneck_type=bottleneck, 
        activation_fn=nn.ELU, fac=fac, dropout_prob=0.0,
        use_attention_on_skips=use_skip_attention
    )

    model.load_state_dict(torch.load(folder + '/' + tag + '.pt', map_location=torch.device('cpu')))

    model = model.float()
    model.to(device)

    num_cores = os.cpu_count()
    num_workers = int(min(22, num_cores-1))
    print('Constructing the dataset')
    _, _, dataloader_test = LangsethData.create_dataloaders(
        '../LangsethDatasetSignal_v4', '../LangsethDatasetNoise_v4', nlen, large_batch_size, mode='1d', channels=channels, 
        num_workers=num_workers, nperseg=nperseg, hop_length=hop_length, test_mode=test_mode, normalization_type="1C")

    # Main loop
    with torch.no_grad():
        vertical_results = []
        horizontal_results = []

        for idx in range(test_sweeps):
            for large_batch in dataloader_test:

                trace_clean = large_batch[0].to(device, non_blocking=True)
                trace_combined = large_batch[1].to(device, non_blocking=True)
                pick_index = large_batch[2].numpy()  # Ensure pick_index is in numpy format
                old_snr = large_batch[3].numpy()  # Old SNR values, shape (batch x channel)

                denoised = model(trace_combined)[0].cpu().numpy()

                # Compute SNR and metrics for the vertical channel (assuming channel 0)
                vertical_snr = compute_snr_from_denoised(
                    trace=denoised[:, 0:1, :],  # Vertical channel
                    pick_index=pick_index,
                    sample_rate=sample_rate
                )

                vertical_metrics = []
                for batch in range(denoised.shape[0]):
                    y = trace_clean[batch, 0, :].cpu().numpy()
                    y_tilde = denoised[batch, 0, :]
                    correlation, l2, ev = compute_metrics_windowed(y, y_tilde, pick_index[batch])
                    vertical_metrics.append([
                        old_snr[batch, 0],       # Old SNR for the vertical channel
                        vertical_snr[batch, 0],  # New SNR for the vertical channel
                        correlation, l2, ev
                    ])
                vertical_results.append(np.array(vertical_metrics))

                # Process horizontal channels only if the "channels" flag is "all" or "3C"
                if channels in ["all", "3C"]:
                    horizontal_metrics = []

                    for channel in [1, 2]:  # Horizontal channels
                        snr_horizontal = compute_snr_from_denoised(
                            trace=denoised[:, channel:channel + 1, :],
                            pick_index=pick_index,
                            sample_rate=sample_rate
                        )

                        # Compute additional metrics
                        for batch in range(denoised.shape[0]):
                            y = trace_clean[batch, channel, :].cpu().numpy()
                            y_tilde = denoised[batch, channel, :]
                            correlation, l2, ev = compute_metrics_windowed(y, y_tilde, pick_index[batch])
                            horizontal_metrics.append([
                                old_snr[batch, channel],  # Old SNR for the horizontal channel
                                snr_horizontal[batch, 0],  # New SNR for the horizontal channel
                                correlation, l2, ev
                            ])

                    horizontal_results.append(np.array(horizontal_metrics))

            print(f'{idx + 1} of {test_sweeps} sweeps')

        # Final numpy arrays
        vertical_array = np.concatenate(vertical_results, axis=0)  # Concatenate along the batch dimension
        horizontal_array = np.concatenate(horizontal_results, axis=0) if horizontal_results else None

    # Example usage
    save_results(tag, vertical_array, horizontal_array)
    plot_metrics_against_snr(tag, vertical_array, horizontal_array, db_bins)

"""
# Plotting function
def plot_metrics_against_snr(tag, vertical_array, horizontal_array, db_bins):
    # Combine vertical and horizontal results for plotting separately

    bin_edges = np.linspace(-20, 20, 11)
    db_bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]

    # Extract and filter out inf values for vertical metrics
    valid_indices_vertical = np.isfinite(vertical_array).all(axis=1)
    vertical_array = vertical_array[valid_indices_vertical]

    old_snr_vertical = vertical_array[:, 0]
    new_snr_vertical = vertical_array[:, 1]
    cc_vertical = vertical_array[:, 2]
    ev_vertical = vertical_array[:, 4]
    #snr_improvement_vertical = new_snr_vertical - old_snr_vertical

    # Extract and filter out inf values for horizontal metrics, if available
    if horizontal_array is not None:
        valid_indices_horizontal = np.isfinite(horizontal_array).all(axis=1)
        horizontal_array = horizontal_array[valid_indices_horizontal]

        old_snr_horizontal = horizontal_array[:, 0]
        new_snr_horizontal = horizontal_array[:, 1]
        cc_horizontal = horizontal_array[:, 2]
        ev_horizontal = horizontal_array[:, 4]
        #snr_improvement_horizontal = new_snr_horizontal - old_snr_horizontal

    # Function to bin data and prepare statistics
    def prepare_binned_data(old_snr, cc, ev, snr_improvement):
        binned_data = {bin_range: [] for bin_range in db_bins}
        bin_means = []

        for i in range(len(old_snr)):
            for bin_range in db_bins:
                if bin_range[0] <= old_snr[i] < bin_range[1]:
                    binned_data[bin_range].append([cc[i], ev[i], snr_improvement[i]])
                    break

        # Prepare data for box plots
        cc_data, ev_data, snr_improvement_data = [], [], []
        labels = []

        for bin_range, data in binned_data.items():
            if len(data) > 0:
                data = np.array(data)
                bin_mean = np.mean([bin_range[0], bin_range[1]])
                bin_means.append(bin_mean)
                labels.append(f'{bin_mean:.1f} dB')  # Use the mean value of the bin as the label
                
                # Calculate statistics for each metric
                mean_cc, lower_cc, upper_cc = calculate_statistics(data[:, 0])
                mean_ev, lower_ev, upper_ev = calculate_statistics(data[:, 1])
                mean_snr, lower_snr, upper_snr = calculate_statistics(data[:, 2])

                # Append data for plotting
                cc_data.append([mean_cc, lower_cc, upper_cc])
                ev_data.append([mean_ev, lower_ev, upper_ev])
                snr_improvement_data.append([mean_snr, lower_snr, upper_snr])

        return cc_data, ev_data, snr_improvement_data, labels

    # Prepare binned data for vertical and horizontal
    cc_data_vertical, ev_data_vertical, snr_improvement_data_vertical, labels_vertical = prepare_binned_data(
        old_snr_vertical, cc_vertical, ev_vertical, new_snr_vertical
    )

    if horizontal_array is not None:
        cc_data_horizontal, ev_data_horizontal, snr_improvement_data_horizontal, labels_horizontal = prepare_binned_data(
            old_snr_horizontal, cc_horizontal, ev_horizontal, new_snr_horizontal
        )

    # Plotting for vertical
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_vertical = [cc_data_vertical, ev_data_vertical, snr_improvement_data_vertical]
    titles_vertical = ['Cross-Correlation (Vertical)', 'Expected Variance (Vertical)', 'Denoised SNR (Vertical)']

    for i, ax in enumerate(axes):
        data = metrics_vertical[i]
        means = [d[0] for d in data]
        lower_bounds = [d[1] for d in data]
        upper_bounds = [d[2] for d in data]
        
        # Create a box plot-like representation of the mean and 95% CI
        ax.errorbar(range(len(data)), means, yerr=[np.array(means) - np.array(lower_bounds), np.array(upper_bounds) - np.array(means)], fmt='o')
        ax.set_title(titles_vertical[i])
        ax.set_xticks(range(len(labels_vertical)))
        ax.set_xticklabels(labels_vertical, rotation=45)
        ax.set_xlabel('Starting SNR (dB)')
        ax.grid()

    axes[0].set_ylabel('Metric Value')
    plt.tight_layout()
    plt.savefig(f'results/{tag}_vertical_metrics_box_plot.png')
    plt.show()

    # Plotting for horizontal (if available)
    if horizontal_array is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics_horizontal = [cc_data_horizontal, ev_data_horizontal, snr_improvement_data_horizontal]
        titles_horizontal = ['Cross-Correlation (Horizontal)', 'Expected Variance (Horizontal)', 'Denoised SNR (Horizontal)']

        for i, ax in enumerate(axes):
            data = metrics_horizontal[i]
            means = [d[0] for d in data]
            lower_bounds = [d[1] for d in data]
            upper_bounds = [d[2] for d in data]
            
            # Create a box plot-like representation of the mean and 95% CI
            ax.errorbar(range(len(data)), means, yerr=[np.array(means) - np.array(lower_bounds), np.array(upper_bounds) - np.array(means)], fmt='o')
            ax.set_title(titles_horizontal[i])
            ax.set_xticks(range(len(labels_horizontal)))
            ax.set_xticklabels(labels_horizontal, rotation=45)
            ax.set_xlabel('Starting SNR (dB)')
            ax.grid()

        axes[0].set_ylabel('Metric Value')
        plt.tight_layout()
        plt.savefig(f'results/{tag}_horizontal_metrics_box_plot.png')
        plt.show()
"""