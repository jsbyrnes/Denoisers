####reformat and hdf5 database into blocks for read performance
#Note that this does not connect back to the metadata. The order in the hd5f files compared to the metadata was lost. 
import os
import h5py
import numpy as np
import pandas as pd

def collect_keys(input_folder):
    """
    Collect keys from all HDF5 files in the input folder.
    """
    # Sort HDF5 filenames
    hdf5_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.hdf5')])
    keys_list = []
    for file_idx, file_path in enumerate(hdf5_files):
        with h5py.File(file_path, 'r') as hdf:
            data_group = hdf['data']
            # List keys within the HDF5 file
            keys = list(data_group.keys())
            for key in keys:
                if key.endswith('S'):
                    data_type = 'signal'
                elif key.endswith('N'):
                    data_type = 'noise'
                else:
                    continue
                keys_list.append({'file_idx': file_idx, 'file_path': file_path, 'key': key, 'type': data_type})
    keys_df = pd.DataFrame(keys_list)
    return keys_df

def ensure_folder_exists(folder_path):
    """
    Ensure that a folder exists; create it if it doesn't.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def process_data_type(data_df, block_size, max_file_size, output_folder):
    data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
    num_rows = data_df.shape[0]
    hdf5_files = {}
    file_indices = data_df['file_idx'].unique()
    for idx in file_indices:
        file_path = data_df.loc[data_df['file_idx'] == idx, 'file_path'].values[0]
        hdf5_files[idx] = h5py.File(file_path, 'r')
    output_file_idx = 1
    output_file_size = 0
    output_hdf5 = None
    block_idx = 0
    for start_idx in range(0, num_rows, block_size):
        end_idx = min(start_idx + block_size, num_rows)
        block_df = data_df.iloc[start_idx:end_idx]
        traces = []
        max_length = 0
        num_channels = None
        # Read the traces
        for idx_row, row in block_df.iterrows():
            file_idx = row['file_idx']
            key = row['key']
            data = hdf5_files[file_idx]['data'][key][()]
            if len(data.shape) == 1:
                data = data.reshape(1, -1)  # Ensure 2D with channels x length
            elif data.shape[0] >= data.shape[1]:
                # Data is (length, channels), transpose to (channels, length)
                data = data.T
            # Else, data is already (channels, length)
            traces.append(data)
            if data.shape[1] > max_length:
                max_length = data.shape[1]
            if num_channels is None:
                num_channels = data.shape[0]
            elif num_channels != data.shape[0]:
                raise ValueError(f"Inconsistent number of channels: expected {num_channels}, got {data.shape[0]}")
        # Pad the traces
        padded_traces = []
        for data in traces:
            pad_width = max_length - data.shape[1]
            if pad_width > 0:
                pad_array = np.zeros((data.shape[0], pad_width))
                padded_data = np.hstack([data, pad_array])
            else:
                padded_data = data
            padded_traces.append(padded_data)
        # Stack the padded traces into block_array
        block_array = np.stack(padded_traces, axis=0)  # Shape: [num_traces_in_block, num_channels, max_length]
        # If needed, transpose to (num_traces_in_block, max_length, num_channels)
        # block_array = block_array.transpose(0, 2, 1)
        # Estimate the size of the data to be written
        data_size = block_array.nbytes
        # Check if we need to open a new output file
        if output_hdf5 is None or (output_file_size + data_size) >= max_file_size:
            if output_hdf5 is not None:
                output_hdf5.close()
                output_file_idx += 1
                output_file_size = 0
                block_idx = 0  # Reset block index for new file
            output_file_path = os.path.join(output_folder, f'waveforms{output_file_idx:06d}.hdf5')
            output_hdf5 = h5py.File(output_file_path, 'w')
            output_hdf5.create_group('data')
        # Write the block into the output HDF5 file
        block_name = f'block{block_idx}'
        dset = output_hdf5['data'].create_dataset(block_name, data=block_array, compression='gzip')
        output_file_size += data_size
        block_idx += 1
    # Close all files
    for f in hdf5_files.values():
        f.close()
    if output_hdf5 is not None:
        output_hdf5.close()

def main(input_folder, output_folder_signal, output_folder_noise, block_size, max_file_size):
    ensure_folder_exists(output_folder_signal)
    ensure_folder_exists(output_folder_noise)
    # Collect keys
    keys_df = collect_keys(input_folder)
    # Separate signals and noises
    signal_df = keys_df[keys_df['type'] == 'signal'].reset_index(drop=True)
    noise_df = keys_df[keys_df['type'] == 'noise'].reset_index(drop=True)
    # Process signals
    #process_data_type(signal_df, block_size, max_file_size, output_folder_signal)
    # Process noises
    process_data_type(noise_df, block_size, max_file_size, output_folder_noise)

if __name__ == "__main__":
    input_folder = '../LangsethDataset_v3/'
    output_folder_signal = '../LangsethDatasetSignal_v4/'
    output_folder_noise = '../LangsethDatasetNoise_v4/'
    block_size = 256  # Set your desired block size
    max_file_size = 2 * 1024 ** 3  # 2GB in bytes
    main(input_folder, output_folder_signal, output_folder_noise, block_size, max_file_size)
