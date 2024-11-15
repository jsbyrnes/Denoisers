#####python sceipt to plot all of the model comparisons
import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

# Dictionary to map filenames to custom labels
custom_labels = {
    "model_T23_110224_fac1_vertical_results.npy": "T23, 8 filters, vertical only.",
    "model_T23_110224_fac2_vertical_results.npy": "T23, 16 filters, vertical only.",
    "model_T23_110324_fac1_horizontal_results.npy": "T23, 8 filters, 3C data.",
    "model_T23_110324_fac1_vertical_results.npy": "T23, 8 filters, 3C data.",
    "model_T23_110324_fac2_horizontal_results.npy": "T23, 16 filters, 3C data.",
    "model_T23_110324_fac2_vertical_results.npy": "T23, 16 filters, 3C data.",
    "model_T23_110424_att_horizontal_results.npy": "T23 with attention, 16 filters, 3C data.",
    "model_T23_110424_att_vertical_results.npy": "T23 with attention, 16 filters, 3C data.",
    "model_T23_110324_fac3_vertical_results.npy": "T23, 32 filters, 3C data.",
    "model_T23_110324_fac3_horizontal_results.npy": "T23, 32 filters, 3C data.",
    "model_WaveDecompY22_110224_fac1_vertical_results.npy": "Y22, 8 filters, vertical only.",
    "model_WaveDecompY22_110224_fac2_vertical_results.npy": "Y22, 16 filters, vertical only.",
    "model_WaveDecompY22_110324_fac1_horizontal_results.npy": "Y22, 8 filters, 3C data.",
    "model_WaveDecompY22_110324_fac1_vertical_results.npy": "Y22, 8 filters, 3C data.",
    "model_WaveDecompY22_110324_fac2_horizontal_results.npy": "Y22, 16 filters, 3C data.",
    "model_WaveDecompY22_110324_fac2_vertical_results.npy": "Y22, 16 filters, 3C data.",
    "model_WaveDecompY22_110424_Skips_horizontal_results.npy": "Y22, attention on SC, 16 filters, 3C data.",
    "model_WaveDecompY22_110424_Skips_vertical_results.npy": "Y22, attention on SC, 16 filters, 3C data.",
    "model_WaveDecompY22_110324_fac3_horizontal_results.npy": "Y22, 32 filters, 3C data.",
    "model_WaveDecompY22_110324_fac3_vertical_results.npy": "Y22, 32 filters, 3C data.",
}

# Define a color for each label to ensure consistency across plots
label_colors = {
    "T23, 8 filters, vertical only.": "blue",
    "T23, 16 filters, vertical only.": "green",
    "T23, 8 filters, 3C data.": "red",
    "T23, 16 filters, 3C data.": "purple",
    "T23 with attention, 16 filters, 3C data.": "orange",
    "Y22, 8 filters, vertical only.": "cyan",
    "Y22, 16 filters, vertical only.": "magenta",
    "Y22, 8 filters, 3C data.": "brown",
    "Y22, 16 filters, 3C data.": "gray",
    "Y22, attention on SC, 16 filters, 3C data.": "olive",
    # Add additional color mappings as needed
}

# Function to calculate the median and upper 95% confidence interval
def calculate_statistics(data):
    median = np.median(data)
    upper_95 = np.percentile(data, 95)
    return median, upper_95

# Function to process and bin data
def prepare_binned_data(old_snr, ev, new_snr, db_bins):
    binned_data = {bin_range: [] for bin_range in db_bins}

    for i in range(len(old_snr)):
        for bin_range in db_bins:
            if bin_range[0] <= old_snr[i] < bin_range[1]:
                binned_data[bin_range].append([ev[i], new_snr[i]])
                break

    # Prepare data for each bin
    ev_data, new_snr_data = [], []
    labels = []

    for bin_range, data in binned_data.items():
        if len(data) > 0:
            data = np.array(data)
            bin_mean = np.mean([bin_range[0], bin_range[1]])
            labels.append(f'{bin_mean:.1f} dB')

            # Calculate statistics
            ev_median, ev_upper_95 = calculate_statistics(data[:, 0])
            snr_median, snr_upper_95 = calculate_statistics(data[:, 1])

            ev_data.append([ev_median, ev_upper_95])
            new_snr_data.append([snr_median, snr_upper_95])

    return ev_data, new_snr_data, labels

# Main script to process all .npy files and save results
def process_all_models(directory, db_bins):
    results = {"vertical": {}, "horizontal": {}}

    # Loop through all files in the directory and process
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            data = np.load(os.path.join(directory, filename), allow_pickle=True)

            # Determine if the file is for vertical or horizontal data
            if 'vertical' in filename:
                old_snr = data[:, 0]
                new_snr = data[:, 1]
                ev = data[:, 4]
                #ev[ev < 0] = 0

                ev_data, new_snr_data, labels = prepare_binned_data(old_snr, ev, new_snr, db_bins)
                results["vertical"][filename] = {"ev_data": ev_data, "new_snr_data": new_snr_data, "labels": labels}

            elif 'horizontal' in filename:
                old_snr = data[:, 0]
                new_snr = data[:, 1]
                ev = data[:, 4]
                #ev[ev < 0] = 0

                ev_data, new_snr_data, labels = prepare_binned_data(old_snr, ev, new_snr, db_bins)
                results["horizontal"][filename] = {"ev_data": ev_data, "new_snr_data": new_snr_data, "labels": labels}

    return results

# Plotting function to compare models with line plots and different symbols
def plot_comparison(results, metric_type, ylabel, title_suffix, save_tag, y_lower=None, y_upper=None):
    # Separate plots for vertical and horizontal
    for orientation in ["vertical", "horizontal"]:
        if not results[orientation]:
            continue

        # Plot median values
        plt.figure(figsize=(10, 5))
        for model, data in results[orientation].items():
            metric_data = data[metric_type]
            labels = data["labels"]

            # Determine marker style and color based on filename
            if "T23" in model:
                marker = 's'  # Square marker for "T23"
            elif "Y22" in model:
                marker = 'o'  # Circle marker for "Y22"
            else:
                marker = '^'  # Triangle marker for other files

            # Use custom label if available, otherwise use filename
            label = custom_labels.get(model, model)
            color = label_colors.get(label, 'black')  # Default to black if no color specified

            # Extract medians
            medians = [d[0] for d in metric_data]
            plt.plot(range(len(medians)), medians, marker=marker, color=color, label=label)

        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.xlabel('Starting SNR (dB)')
        plt.ylabel(ylabel)
        plt.title(f'{title_suffix} - Median ({orientation.capitalize()})')
        if y_lower is not None or y_upper is not None:
            plt.ylim(bottom=y_lower, top=y_upper)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results/{save_tag}_{orientation}_median_comparison.pdf')
        plt.show()

        # Plot upper 95% confidence values
        plt.figure(figsize=(10, 5))
        for model, data in results[orientation].items():
            metric_data = data[metric_type]
            labels = data["labels"]

            # Determine marker style and color based on filename
            if "T23" in model:
                marker = 's'  # Square marker for "T23"
            elif "Y22" in model:
                marker = 'o'  # Circle marker for "Y22"
            else:
                marker = '^'  # Triangle marker for other files

            # Use custom label if available, otherwise use filename
            label = custom_labels.get(model, model)
            color = label_colors.get(label, 'black')  # Default to black if no color specified

            # Extract upper 95% confidence values
            upper_bounds = [d[1] for d in metric_data]
            plt.plot(range(len(upper_bounds)), upper_bounds, marker=marker, color=color, label=label)

        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.xlabel('Starting SNR (dB)')
        plt.ylabel(ylabel)
        plt.title(f'{title_suffix} - Upper 95% CI ({orientation.capitalize()})')
        if y_lower is not None or y_upper is not None:
            plt.ylim(bottom=y_lower, top=y_upper)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results/{save_tag}_{orientation}_upper_95_comparison.pdf')
        plt.show()

# Example usage
db_bins = [(i, i + 4) for i in range(-40, 30, 5)]
results = process_all_models('./results/', db_bins)

# Plot comparisons
plot_comparison(results, "ev_data", "Explained Variance", "Explained Variance", "explained_variance", y_lower=-0.5, y_upper=1)
plot_comparison(results, "new_snr_data", "Denoised SNR", "Denoised SNR", "denoised_snr", y_lower=-1)
