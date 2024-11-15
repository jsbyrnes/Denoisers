import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
DEBUG = False

class LangsethDataset(Dataset):
    def __init__(self, folder_path_signal, folder_path_noise, signal_keys=None, noise_keys=None, mode='1d', nlen=512, nperseg=31, 
                hop_length=1, test_mode=False, channels='all', normalization_type="1C"):
        self.folder_path_signal = folder_path_signal
        self.folder_path_noise = folder_path_noise
        self.nlen = nlen
        self.nperseg = nperseg
        self.hop_length = hop_length
        self.window = torch.hann_window(self.nperseg)
        self.mode = mode
        self.channels = channels
        self.normalization_type = normalization_type

        # Collect HDF5 files from both signal and noise folders
        self.signal_hdf5_files_list = sorted([os.path.join(folder_path_signal, f) for f in os.listdir(folder_path_signal) if f.endswith('.hdf5')])
        self.noise_hdf5_files_list = sorted([os.path.join(folder_path_noise, f) for f in os.listdir(folder_path_noise) if f.endswith('.hdf5')])

        if test_mode:
            self.signal_hdf5_files_list = self.signal_hdf5_files_list[:2]
            self.noise_hdf5_files_list = self.noise_hdf5_files_list[:2]

        # Initialize keys if not provided
        if signal_keys is not None:
            if isinstance(signal_keys, list):
                self.signal_keys = pd.DataFrame(signal_keys, columns=["file_idx", "key"])
            elif isinstance(signal_keys, pd.DataFrame):
                self.signal_keys = signal_keys
            else:
                raise ValueError("signal_keys must be a list or pandas DataFrame")
        else:
            self.signal_keys = pd.DataFrame()

        if noise_keys is not None:
            if isinstance(noise_keys, list):
                self.noise_keys = pd.DataFrame(noise_keys, columns=["file_idx", "key"])
            elif isinstance(noise_keys, pd.DataFrame):
                self.noise_keys = noise_keys
            else:
                raise ValueError("noise_keys must be a list or pandas DataFrame")
        else:
            self.noise_keys = pd.DataFrame()

        if self.signal_keys.empty or self.noise_keys.empty:
            self._initialize_keys()

        # Placeholder for file handles (to be initialized per worker)
        self.signal_hdf5_files = None
        self.noise_hdf5_files = None

        # HDF5 optimization flags
        #these probably don't matter but I use chatgpt a lot and it recommended them. 
        self.hdf5_flags = {
            'rdcc_nbytes': 1024**2 * 128,  # 128MB chunk cache
            'rdcc_w0': 0.75,               # Cache write strategy
            'rdcc_nslots': 1013,           # Number of cache slots (prime number)
            'driver': 'sec2'               # I/O driver
        }

        # Set the total length as the sum of both lengths
        self.total_len = len(self.signal_keys) + len(self.noise_keys)

    def _initialize_hdf5_files(self):
        if self.signal_hdf5_files is None:
            self.signal_hdf5_files = {idx: h5py.File(file, 'r', **self.hdf5_flags) for idx, file in enumerate(self.signal_hdf5_files_list)}
        if self.noise_hdf5_files is None:
            self.noise_hdf5_files = {idx: h5py.File(file, 'r', **self.hdf5_flags) for idx, file in enumerate(self.noise_hdf5_files_list)}

    def _initialize_keys(self):
        """Initialize keys for signals and noises into pandas DataFrames."""
        signal_rows = []
        noise_rows = []

        # Collect block-level keys from signal files
        for file_idx, file in enumerate(self.signal_hdf5_files_list):
            with h5py.File(file, 'r') as hdf:
                data_group = hdf['data']
                block_names = list(data_group.keys())
                for block_name in block_names:
                    # Append block-level key for each block in signal file
                    signal_rows.append({'file_idx': file_idx, 'key': block_name})

        # Collect block-level keys from noise files
        for file_idx, file in enumerate(self.noise_hdf5_files_list):
            with h5py.File(file, 'r') as hdf:
                data_group = hdf['data']
                block_names = list(data_group.keys())
                for block_name in block_names:
                    # Append block-level key for each block in noise file
                    noise_rows.append({'file_idx': file_idx, 'key': block_name})

        # Convert lists of dictionaries to pandas DataFrames
        self.signal_keys = pd.DataFrame(signal_rows)
        self.noise_keys = pd.DataFrame(noise_rows)

    def _find_file_and_local_idx(self, idx, trace_counts):
        """Find the file and local index within the file for a given global idx."""
        cumulative_sum = 0

        for i, count in enumerate(trace_counts):
            cumulative_sum += count
            if idx < cumulative_sum:
                file_idx = i
                local_idx = idx - (cumulative_sum - count)  # Adjust idx for the file's starting index
                return file_idx, local_idx

        raise IndexError("Index out of range.")

    def _normalize_trace(self, trace, start_shift=None):
        """
        Normalize a single trace based on the specified normalization type.
        """
        if self.normalization_type == "3C":
            # First 3 channels together, 4th separately
            first_three_channels = trace[:3, :]
            fourth_channel = trace[3, :]

            if start_shift is not None:
                # Combined max for the first 3 channels in the window
                window = first_three_channels[:, start_shift:(start_shift + self.nlen)]
            else:
                window = first_three_channels

            combined_max_amp = torch.max(torch.abs(window))
            if combined_max_amp > 0:
                first_three_channels /= combined_max_amp

            fourth_channel_max_amp = torch.max(torch.abs(fourth_channel))
            if fourth_channel_max_amp > 0:
                fourth_channel /= fourth_channel_max_amp

            # Update the trace
            trace[:3, :] = first_three_channels
            trace[3, :] = fourth_channel

        elif self.normalization_type == "1C":
            # Normalize each channel separately
            for ch in range(trace.shape[0]):
                if start_shift is not None:
                    max_amp = torch.max(torch.abs(trace[ch, start_shift:(start_shift + self.nlen)]))
                else:
                    max_amp = torch.max(torch.abs(trace[ch, :]))
                if max_amp > 0:
                    trace[ch, :] /= max_amp
        else:
            raise ValueError("Normalization type must be either '3C' or '1C'")
        return trace

    def _scale_noise(self, noise_clip, presignal_rms):
        """
        Apply random amplitude scaling to the noise clip.
        """
        nsign = (-1) ** np.random.randint(0, 2)
        scaling = np.sqrt(np.random.uniform(0.001, 1))
        amp = 10 ** (-1.5 + scaling * (np.log10(3) + 1.5))

        for ch in range(noise_clip.shape[0]):
            presig = presignal_rms[ch]
            comp = noise_clip[ch, :]
            if torch.sum(comp ** 2) == 0:
                continue  # Skip zero-valued components
                        
            comp *= max(presig * 4, amp) * nsign
            noise_clip[ch, :] = comp
        return noise_clip

    def __len__(self):
        # Return the combined length
        return self.total_len

    def __getitem__(self, idx):

        if idx < len(self.signal_keys):
            # Index the signal deterministically
            signal_row = self.signal_keys.iloc[idx]
            signal_file_idx, signal_key = signal_row["file_idx"], signal_row["key"]
            signal_file_path = self.signal_hdf5_files_list[signal_file_idx]

            with h5py.File(signal_file_path, 'r') as signal_file:
                data_group = signal_file['data']
                dataset = torch.tensor(data_group[signal_key][:], dtype=torch.float32)

            # Randomly select a noise sample
            noise_idx = np.random.randint(len(self.noise_keys))
            noise_row = self.noise_keys.iloc[noise_idx]
            noise_file_idx, noise_key = noise_row["file_idx"], noise_row["key"]
            noise_file_path = self.noise_hdf5_files_list[noise_file_idx]

            with h5py.File(noise_file_path, 'r') as noise_file:
                data_group_noise = noise_file['data']
                dataset_noise = torch.tensor(data_group_noise[noise_key][:], dtype=torch.float32)

        else:
            # Index the noise deterministically
            noise_idx = idx - len(self.signal_keys)  # Adjust index for noise keys
            noise_row = self.noise_keys.iloc[noise_idx]
            noise_file_idx, noise_key = noise_row["file_idx"], noise_row["key"]
            noise_file_path = self.noise_hdf5_files_list[noise_file_idx]

            with h5py.File(noise_file_path, 'r') as noise_file:
                data_group_noise = noise_file['data']
                dataset_noise = torch.tensor(data_group_noise[noise_key][:], dtype=torch.float32)

            # Randomly select a signal sample
            signal_idx = np.random.randint(len(self.signal_keys))
            signal_row = self.signal_keys.iloc[signal_idx]
            signal_file_idx, signal_key = signal_row["file_idx"], signal_row["key"]
            signal_file_path = self.signal_hdf5_files_list[signal_file_idx]

            with h5py.File(signal_file_path, 'r') as signal_file:
                data_group = signal_file['data']
                dataset = torch.tensor(data_group[signal_key][:], dtype=torch.float32)

        # Determine the minimum number of traces to match the first dimension of both datasets
        min_num_traces = min(dataset.shape[0], dataset_noise.shape[0])

        # Clip both datasets to the same size along the first dimension
        dataset = dataset[:min_num_traces]
        dataset_noise = dataset_noise[:min_num_traces]

        # Shuffle the datasets along the first dimension with different permutations
        shuffle_indices_dataset = torch.randperm(min_num_traces)  # Random permutation for the signal dataset
        shuffle_indices_dataset_noise = torch.randperm(min_num_traces)  # Random permutation for the noise dataset

        # Apply the shuffles
        dataset = dataset[shuffle_indices_dataset]
        dataset_noise = dataset_noise[shuffle_indices_dataset_noise]

        batch_size, num_channels, data_length = dataset.shape  # Extract dimensions

        # Generate per-trace random start shifts for signals
        start_shifts = torch.randint(200, 1900, (batch_size,))
        picks = 2100 - start_shifts

        # Initialize tensors for processed signal and noise clips
        signal_clips = torch.zeros((batch_size, num_channels, self.nlen))
        noise_clips = torch.zeros_like(signal_clips)
        presignal_rms = torch.zeros((batch_size, num_channels))

        # Process each trace in the batch
        for i in range(batch_size):
            # Signal processing
            signal_trace = dataset[i]  # Shape: [channels, length]
            start_shift = start_shifts[i].item()
            pick = picks[i].item()

            # Normalize signal trace
            signal_trace = self._normalize_trace(signal_trace, start_shift)

            # Calculate presignal RMS
            presignal_rms[i] = torch.sqrt(torch.mean(signal_trace[:, :2000] ** 2, dim=1))

            # Clip the signal
            signal_clips[i] = signal_trace[:, start_shift:(start_shift + self.nlen)]

            signal_clips[i] = replace_zeros_with_noise(signal_clips[i], std=0.001)

            # Noise processing
            noise_trace = dataset_noise[i]
            max_noise_start = noise_trace.shape[-1] - self.nlen
            noise_start_shift = np.random.randint(0, max_noise_start) if max_noise_start > 0 else 0
            noise_clip = noise_trace[:, noise_start_shift:(noise_start_shift + self.nlen)]

            noise_clip = replace_zeros_with_noise(noise_clip)

            # Normalize noise trace
            noise_clip = self._normalize_trace(noise_clip)

            # Apply random amplitude scaling to the noise
            noise_clip = self._scale_noise(noise_clip, presignal_rms[i])

            # Zero noise channels if the corresponding signal channel is zero
            #zero_mask = (signal_clips[i].abs().sum(dim=1) == 0)
            #noise_clip[zero_mask, :] = 0.0

            # Store the processed noise clip
            noise_clips[i] = noise_clip

        ####calculate a signal to noise ratio
        snr = compute_snr_batch_multi_channel(noise_clips, signal_clips, picks, 200)
        
        # Combine signal and noise
        combined_clips = signal_clips + noise_clips

        # Normalize combined clips
        amp2 = combined_clips.abs().reshape(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
        signal_clips_normalized = signal_clips / amp2
        combined_clips_normalized = combined_clips / amp2

        # Handle channel selection
        if self.channels == 'vertical':
            signal_clips_normalized = signal_clips_normalized[:, :1, :]
            combined_clips_normalized = combined_clips_normalized[:, :1, :]
        elif self.channels == '3C':
            signal_clips_normalized = signal_clips_normalized[:, :3, :]
            combined_clips_normalized = combined_clips_normalized[:, :3, :]
        elif self.channels == 'pressure':
            signal_clips_normalized = signal_clips_normalized[:, [0, 3], :]
            combined_clips_normalized = combined_clips_normalized[:, [0, 3], :]
        elif self.channels == 'horizontals':
            signal_clips_normalized = signal_clips_normalized[:, [1, 2], :]
            combined_clips_normalized = combined_clips_normalized[:, [1, 2], :]

        # Handle 'stft' mode
        if self.mode == 'stft':
            # Initialize lists to collect STFT results for signal and noise
            stft_signal_clips = []
            stft_noise_clips = []

            num_channels = signal_clips_normalized.shape[1]

            # Loop over each trace in the batch
            for i in range(batch_size):
                # Initialize lists for channels of the current trace
                trace_stft_signal = []
                trace_stft_noise = []

                # Loop over each channel
                for channel in range(num_channels):
                    # Compute STFT for the signal and noise clips
                    stft_signal = torch.stft(
                        signal_clips_normalized[i, channel, :], 
                        n_fft=self.nperseg, 
                        hop_length=self.hop_length, 
                        window=self.window, 
                        return_complex=True
                    )

                    stft_noise = torch.stft(
                        noise_clips[i, channel, :], 
                        n_fft=self.nperseg, 
                        hop_length=self.hop_length, 
                        window=self.window, 
                        return_complex=True
                    )

                    # Append the STFT result for the current channel
                    trace_stft_signal.append(stft_signal.unsqueeze(0))  # Shape: [1, freq_bins, time_frames]
                    trace_stft_noise.append(stft_noise.unsqueeze(0))

                # Concatenate channels for the current trace
                stft_signal_clips.append(torch.cat(trace_stft_signal, dim=0).unsqueeze(0))  # Shape: [1, channels, freq_bins, time_frames]
                stft_noise_clips.append(torch.cat(trace_stft_noise, dim=0).unsqueeze(0))

            # Concatenate traces to form the final batch
            stft_signal_clips = torch.cat(stft_signal_clips, dim=0)  # Shape: [batch_size, channels, freq_bins, time_frames]
            stft_noise_clips = torch.cat(stft_noise_clips, dim=0)

            # Combine signal and noise clips
            combined_stft = stft_signal_clips + stft_noise_clips
            combined = data_2_input(combined_stft)

            # Initialize mask
            mask = torch.zeros(batch_size, num_channels, combined.shape[-2], combined.shape[-1])

            # Compute ratio for each channel in each batch element
            for i in range(num_channels):
                signal_abs = stft_signal_clips[:, i, :, :].abs()
                noise_abs = stft_noise_clips[:, i, :, :].abs()
                rat = torch.nan_to_num(noise_abs / signal_abs, nan=0.0, posinf=1e20)
                mask[:, i, :, :] = 1 / (1 + rat)

            # Handle channel selection
            if self.channels == 'vertical':
                mask = mask[:, :1, :, :]
                combined = combined[:, :2, :, :]
            elif self.channels == '3C':
                mask = mask[:, :3, :, :]
                combined = combined[:, :6, :, :]
            elif self.channels == 'pressure':
                mask = mask[:, [0, 3], :, :]
                combined = combined[:, [0, 1, 6, 7], :, :]
            elif self.channels == 'horizontals':
                mask = mask[:, [1, 2], :, :]
                combined = combined[:, [2, 3, 4, 5], :, :]

            # Return sample as a tuple with mask, combined input, and picks
            sample = (mask, combined, picks, snr, signal_clips_normalized, combined_clips_normalized)

        elif self.mode == '1d':
            # Return sample for '1d' mode
            sample = (signal_clips_normalized, combined_clips_normalized, picks, snr)

        return sample

def compute_snr_batch_multi_channel(noise_clips, signal_clips, pick_index, sample_rate):
    """
    Compute the signal-to-noise ratio (SNR) in dB for each channel in a batch of multi-channel datasets.

    Args:
        noise_clips: 3D numpy array containing noise clips (batch x channel x length).
        signal_clips: 3D numpy array containing signal clips (batch x channel x length).
        pick_index: 1D numpy array containing the index of the pick for each element in the batch.
        sample_rate: The sample rate of the signal.

    Returns:
        A torch tensor of SNR values in dB with shape (batch x channel).
    """
    # Initialize an array to hold SNR values
    snr_values_batch = torch.full((noise_clips.shape[0], noise_clips.shape[1]), -torch.inf)

    # Set full window size for integration
    full_window_size = int(2 * sample_rate)
    max_length = noise_clips.shape[2]  # Length of the signals

    for batch in range(noise_clips.shape[0]):
        pick = pick_index[batch]  # Get the pick index for this batch element

        # Calculate the post window size based on the boundaries
        post_window_size = min(full_window_size, pick - 50, max_length - pick)

        if pick < post_window_size or pick >= max_length:
            continue  # Skip if pick index is too close to the start or end

        for channel in range(noise_clips.shape[1]):
            noise_signal = noise_clips[batch, channel]
            signal_signal = signal_clips[batch, channel]

            # Compute power before the pick from noise_clips
            power_before = torch.sum(
                noise_signal[max(0, pick - 50 - post_window_size):(pick - 50)]**2
            ) / post_window_size

            # Compute integrated power after the pick from signal_clips
            power_after = torch.sum(
                signal_signal[pick:pick + post_window_size]**2
            ) / post_window_size

            # Convert to dB and store in the array
            if power_before != 0 and power_after != 0:
                snr_values_batch[batch, channel] = 10 * torch.log10(power_after / power_before)

    return snr_values_batch

def data_2_input(stft_input, norm_input=True):
    """
    Process STFT input to separate real and imaginary parts and handle normalization.

    Args:
        stft_input: Complex STFT input tensor of shape [batch, channels, freq_bins, time_frames].
        norm_input (bool): Whether to normalize each STFT input by its maximum absolute value.

    Returns:
        Tensor of shape [batch, channels * 2, freq_bins, time_frames] with real and imaginary parts.
    """
    if norm_input:
        # Normalize each channel by its maximum absolute value, add a small epsilon for numerical stability
        max_vals = torch.abs(stft_input).amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        stft_input = stft_input / max_vals

    # Stack real and imaginary parts along the second dimension
    real_part = stft_input.real
    imag_part = stft_input.imag
    combined_input = torch.cat([real_part, imag_part], dim=1)  # Concatenate along channels

    return combined_input

def create_dataloaders(folder_path_signal, folder_path_noise, nlen, batch_size, num_workers=4, mode='1d', nperseg=31,
    hop_length=1, test_mode=False, channels='all', normalization_type="1C", device=torch.device("cpu"),
    splits=[0.85, 0.1, 0.05]):
    
    # Instantiate the LangsethDataset to collect all keys
    dataset = LangsethDataset(folder_path_signal, folder_path_noise, nlen=nlen, mode=mode, nperseg=nperseg, 
                              test_mode=test_mode, hop_length=hop_length, channels=channels, normalization_type=normalization_type)

    # Get all signal and noise keys
    signal_keys = dataset.signal_keys
    noise_keys = dataset.noise_keys

    # Ensure the splits sum to 1.0 for valid percentages
    if sum(splits) != 1.0:
        raise ValueError("The split percentages must sum to 1.0")

    # Shuffle and split signal keys
    np.random.seed(42)  # For reproducibility
    signal_keys = signal_keys.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size_signal = int(splits[0] * len(signal_keys))
    val_size_signal = int(splits[1] * len(signal_keys))
    
    signal_train_keys = signal_keys[:train_size_signal]
    signal_val_keys = signal_keys[train_size_signal:train_size_signal + val_size_signal]
    signal_test_keys = signal_keys[train_size_signal + val_size_signal:]

    # Shuffle and split noise keys
    noise_keys = noise_keys.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size_noise = int(splits[0] * len(noise_keys))
    val_size_noise = int(splits[1] * len(noise_keys))
    
    noise_train_keys = noise_keys[:train_size_noise]
    noise_val_keys = noise_keys[train_size_noise:train_size_noise + val_size_noise]
    noise_test_keys = noise_keys[train_size_noise + val_size_noise:]

    if num_workers > 0:
        pw = True
    else:
        pw = False

    pm = device.type == "cuda"

    # Create training dataset
    train_dataset = LangsethDataset(
        folder_path_signal, folder_path_noise,
        signal_keys=signal_train_keys,
        noise_keys=noise_train_keys,
        nlen=nlen,
        mode=mode,
        nperseg=nperseg,
        hop_length=hop_length,
        test_mode=test_mode,
        channels=channels,
        normalization_type=normalization_type
    )

    # Create validation dataset
    val_dataset = LangsethDataset(
        folder_path_signal, folder_path_noise,
        signal_keys=signal_val_keys,
        noise_keys=noise_val_keys,
        nlen=nlen,
        mode=mode,
        nperseg=nperseg,
        hop_length=hop_length,
        test_mode=test_mode,
        channels=channels,
        normalization_type=normalization_type
    )

    # Create test dataset
    test_dataset = LangsethDataset(
        folder_path_signal, folder_path_noise,
        signal_keys=signal_test_keys,
        noise_keys=noise_test_keys,
        nlen=nlen,
        mode=mode,
        nperseg=nperseg,
        hop_length=hop_length,
        test_mode=test_mode,
        channels=channels,
        normalization_type=normalization_type
    )

    # Create DataLoaders
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pm,
        persistent_workers=pw, 
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    dataloader_valid = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pm,
        persistent_workers=pw,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    dataloader_test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pm,
        persistent_workers=pw,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    # Print the sizes of the datasets
    print(f"Training dataset size: {len(train_dataset) * 256}")
    print(f"Validation dataset size: {len(val_dataset) * 256}")
    print(f"Test dataset size: {len(test_dataset) * 256}")

    return dataloader_train, dataloader_valid, dataloader_test

def replace_zeros_with_noise(noise_clips, std=1):
    # Get the shape of the tensor
    channels, length = noise_clips.shape

    # Iterate over each batch and channel
    for c in range(channels):
        # Check if the trace is all zeros
        if torch.all(noise_clips[c, :] == 0):
            # Replace with white Gaussian noise
            noise_clips[c, :] = torch.randn(length)

    return noise_clips

def load_csv_metadata(folder_path):
    # Initialize an empty list to store DataFrames
    metadata_frames = []

    # Loop over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read each CSV file into a DataFrame
            df = pd.read_csv(file_path)
            metadata_frames.append(df)

    # Concatenate all DataFrames into a single DataFrame
    metadata_df = pd.concat(metadata_frames, ignore_index=True)
    return metadata_df

def custom_collate_fn(batch):
    """
    Custom collate function to stack each element in the batch's tuples along the first dimension.

    Args:
        batch: List of tuples, where each tuple contains three tensors:
            (signal_clips, combined_clips, picks), and each tensor has shape
            [256, channels, length] (for signal and combined_clips) or a similar shape for picks.

    Returns:
        Tuple of three stacked tensors, where each tensor has a new leading dimension
        representing the concatenated batch size.
    """
    # Extract each element of the tuples into separate lists
    #signal_clips = torch.cat([item[0] for item in batch], dim=0)
    #combined_clips = torch.cat([item[1] for item in batch], dim=0)
    #picks = torch.cat([item[2] for item in batch], dim=0)  # Assumes picks is also stackable along the first dim
    #snr = torch.cat([item[3] for item in batch], dim=0)  # Assumes snr is also stackable along the first dim

    # Initialize an empty tuple to hold the stacked results
    stacked_elements = ()
    
    # Get the number of elements in each item (assume all items have the same structure)
    num_elements = len(batch[0])
    
    # Iterate over each element position and stack the tensors
    for i in range(num_elements):
        stacked_element = torch.cat([item[i] for item in batch], dim=0)
        stacked_elements += (stacked_element,)
    
    return stacked_elements

    #return (signal_clips, combined_clips, picks, snr)
