import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import seisbench.data as sbd
from scipy import signal
import warnings
import math
import torch.nn.init as init
import os
import multitaper.mtspec as mt

warnings.filterwarnings('ignore')

class ParallelConv1dTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final_layer=False):
        super(ParallelConv1dTransposeBlock, self).__init__()
        self.conv_transpose3 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose5 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv_transpose7 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.elu = nn.ELU()
        self.final_layer = final_layer

    def forward(self, x):
        out3 = self.conv_transpose3(x)
        out5 = self.conv_transpose5(x)
        out7 = self.conv_transpose7(x)
        out = torch.cat([out3, out5, out7], dim=1)  # Concatenate along the channel dimension
        if not self.final_layer:
            out = self.bn(out)
            out = self.elu(out)
        return out

class ParallelConvGenerator(nn.Module):
    def __init__(self, latent_dim, num_layers, initial_filters, output_channels, seq_length=512):
        super(ParallelConvGenerator, self).__init__()
        filter_multiplier = 2  # Hardwired

        # Linear layer to expand latent space to feature map size
        self.fc_initial = nn.Linear(latent_dim, initial_filters * (filter_multiplier ** (num_layers - 1)) * 3 * seq_length // (2 ** num_layers))

        layers = []
        for i in range(num_layers):
            in_channels = initial_filters * (filter_multiplier ** (num_layers - i - 1)) * 3
            out_channels = initial_filters * (filter_multiplier ** (num_layers - i - 2)) if i != num_layers - 1 else output_channels
            layers.append(ParallelConv1dTransposeBlock(in_channels, out_channels, final_layer=(i == num_layers - 1)))
        self.layers = nn.ModuleList(layers)
        self.seq_length = seq_length

    def forward(self, z):
        # Expand latent space to feature map size
        hh = self.fc_initial(z).view(z.size(0), -1, self.seq_length // (2 ** len(self.layers)))  # Reshape to match expected input shape of deconv layers

        for layer in self.layers:
            hh = layer(hh)

        return hh.mean(dim=1, keepdim=True)  # Output shape is (batch, 1, seq_length)

class ParallelConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final_layer=False):
        super(ParallelConv1dBlock, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.elu = nn.ELU()
        self.final_layer = final_layer

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out = torch.cat([out3, out5, out7], dim=1)  # Concatenate along the channel dimension
        if not self.final_layer:
            out = self.bn(out)
            out = self.elu(out)
        return out

class ParallelConvDiscriminator(nn.Module):
    def __init__(self, input_channels, num_layers, initial_filters, seq_length=512):
        super(ParallelConvDiscriminator, self).__init__()
        filter_multiplier = 2  # Hardwired

        layers = []
        for i in range(num_layers):

            in_channels = input_channels if i == 0 else initial_filters * (filter_multiplier ** (i - 1)) * 3
            out_channels = initial_filters * (filter_multiplier ** i)
            layers.append(ParallelConv1dBlock(in_channels, out_channels, final_layer=(i == num_layers - 1)))
        self.layers = nn.ModuleList(layers)

        # Final linear layer to map the features to a single output value (between 0 and 1)
        self.fc_final = nn.Linear(initial_filters * (filter_multiplier ** (num_layers - 1)) * 3 * (seq_length // (2 ** num_layers)), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward through convolutional layers
        for layer in self.layers:
            x = layer(x)

        # Flatten the output for the final classification
        x = x.view(x.size(0), -1)

        # Pass through the final linear layer and apply sigmoid to get a probability score
        x = self.fc_final(x)
        return self.sigmoid(x)

class GRUGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_channels, seq_length, dropout=0.2):
        super(GRUGenerator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional GRU
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_channels)  # Multiply by 2 for bidirectional

        self._initialize_weights()

    def forward(self, z):
        # Repeat the noise vector across the sequence length
        x = z.unsqueeze(1).repeat(1, self.seq_length, 1)

        # Initialize hidden state
        h_0 = torch.randn(self.num_layers * 2, z.size(0), self.hidden_dim).to(z.device)  # Multiply by 2 for bidirectional

        # Forward pass through GRU
        h, _ = self.gru(x, h_0)

        # Pass the output through the fully connected layer
        x = self.fc(h)
        return torch.tanh(x)

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

# Define the discriminator network using GRU
class GRUDiscriminator(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_layers, output_dim=1):
        super(GRUDiscriminator, self).__init__()
        self.gru = nn.GRU(input_channels, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Bidirectional GRU doubles the hidden_dim

    def forward(self, x):

        x = x.permute(0, 2, 1)  # (batch_size, input_channels, sequence_length) -> (batch_size, sequence_length, input_channels)                
        _, hn = self.gru(x)
        #hn = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate last hidden states of both GRU directions

        hn = hn.permute(1,0,2)[:, -1, :].squeeze()

        x = self.fc(hn)

        return torch.sigmoid(x)  # Use sigmoid for binary classification

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MyDataset(Dataset):
    def __init__(self, data_vector_1, nlen):
        self.data_vector_1 = torch.tensor(data_vector_1, dtype=torch.float32)
        self.nlen = nlen

    def __len__(self):
        return len(self.data_vector_1)

    def __getitem__(self, idx):

        x = self.data_vector_1[idx]

        x *= 1/(torch.max(torch.abs(x)) + 0.00001) #zero protect
        #x *= 1/(torch.std(x) + 0.00001) #zero protect

        sample = {'data': x}
        return sample

def getsnr(trace):

    #first, get the sta/lta around the arrival, which was centered before calling this
    ixs = np.arange(-25, 25) + int(trace.size/2)

    srt = 25 #samples, 0.5 s for 50 Hz
    lng = 200 #samples, 4 s for 50 Hz

    snr = 0

    for ix in ixs:

        s = np.sum(trace[ix:ix+srt]**2)
        l = np.sum(trace[ix-lng:ix]**2)

        if s/l > snr:
            snr = s/l

    if np.isinf(snr):
        snr = 0

    return snr

def format_OBS(nlen, min_snr):

    chunks = 1000

    ind = 0

    data = sbd.OBS(dimension_order="NCW", component_order="Z12H") #sampling rate doesn't work
    total, _ = data.metadata.shape

    sos = signal.butter(4, 2.5, 'hp', fs=sr, output='sos')

    data_vec = np.array(())

    #while ind < total:
    while ind < 25000:

        if ind + chunks < total:
            wf_vec_chunk  = data.get_waveforms(np.arange(ind, ind + chunks), sampling_rate=50)[:,(0,3),:]
        else:
            wf_vec_chunk  = data.get_waveforms(np.arange(ind, total), sampling_rate=50)[:,(0,3),:]
        shp = wf_vec_chunk.shape

        data_chunk = np.zeros((shp[0], 1, nlen*2))
        noise_chunk = np.zeros((shp[0], 1, nlen*2))

        #filter
        for k in range(shp[0]):
            wf_vec_chunk[k,0,:] = signal.sosfilt(sos, wf_vec_chunk[k, 0, :])
            #wf_vec_chunk[k,1,:] = signal.sosfilt(sos, wf_vec_chunk[k, 1, :])
            #wf_vec_chunk[k,2,:] = signal.sosfilt(sos, wf_vec_chunk[k, 2, :])

            #get the sampling rate
            sp = data.metadata['trace_sampling_rate_hz'][ind + k]
            #get the new index value
            itp = np.round(data.metadata['trace_p_arrival_sample'][ind + k]*(50/sp)).astype(int)

            #clip noise if possible, middle section of it
            if itp > 2*nlen:
                ix = np.round((itp - 2*nlen)/2).astype(int)

                noise_chunk[k, 0, :] = wf_vec_chunk[k, 0, ix:ix+2*nlen]
                #noise_chunk[k, 1, :] = wf_vec_chunk[k, 1, ix:ix+2*nlen]
                #noise_chunk[k, 2, :] = wf_vec_chunk[k, 2, ix:ix+2*nlen]

                #amp = np.max(np.abs(np.hstack( (noise_chunk[k, 0, :], noise_chunk[k, 1, :], noise_chunk[k, 2, :]) ))) + 0.001
                #noise_chunk[k, :, :] = noise_chunk[k, :, :]/amp

            if itp > 2*nlen: #currently just skips if too close

                #snr = getsnr(wf_vec_chunk[k, 0, (itp - nlen):(itp + nlen)])
                snr = 11
                
                if snr > min_snr:
                    data_chunk[k, 0, :] = wf_vec_chunk[k, 0, (itp - nlen):(itp + nlen)]
                    #data_chunk[k, 1, :] = wf_vec_chunk[k, 1, (itp - nlen):(itp + nlen)]
                    #data_chunk[k, 2, :] = wf_vec_chunk[k, 2, (itp - nlen):(itp + nlen)]

                    #amp = np.max(np.abs(np.hstack( (data_chunk[k, 0, :], data_chunk[k, 1, :], data_chunk[k, 2, :]) ))) + 0.001
                    #data_chunk[k, :, :] = data_chunk[k, :, :]/amp

        data_ind = np.min(np.sum(data_chunk**2,axis=2), axis=1) > 0.0
        noise_ind = np.min(np.sum(noise_chunk**2,axis=2), axis=1) > 0.0# + ~np.isnan(np.sum(np.sum(noise_chunk,axis=2), axis=1))

        if data_vec.size == 0:
            data_vec  = data_chunk[data_ind, :, :]
            noise_vec = noise_chunk[noise_ind, :, :]
        else:
            data_vec = np.vstack( (data_vec, data_chunk[data_ind, :, :]) )
            noise_vec = np.vstack( (noise_vec, noise_chunk[noise_ind, :, :]) )

        ind += chunks

    #remove microtraces
    #x = np.std(noise_vec, axis=2)
    #ind = np.logical_and(x[:, 0]/x[:,1] > 0.05, x[:, 1]/x[:,0] > 0.05)
    #noise_vec = noise_vec[ind, :, :]

    return data_vec, noise_vec

# Function to plot and save generated samples
def plot_generated_samples(generator, fixed_noise, epoch, num_samples=5):
    generator.eval()
    with torch.no_grad():
        generated_samples = generator(fixed_noise[:num_samples]).cpu().numpy()

    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(generated_samples[i][0])  # Plot the generated sequence
        plt.title(f"Sample {i + 1}")
        plt.axis('off')
    plt.suptitle(f"Generated Samples at Epoch {epoch + 1}")
    plt.tight_layout()
    plt.savefig(f'GAN_samples_epoch_{epoch + 1}.png')
    plt.close()

# Function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, loss, path="vae_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

np.random.seed(60) #same seed as used during training for an in-house check
torch.manual_seed(60)

sr = 50
seq_length = 512 # window length

print('Constructing the dataset')
data_vec, noise_vec = format_OBS(int(seq_length/2), 10)

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(5, 1, i + 1)
    plt.plot(noise_vec[i][0])  # Plot the generated sequence
    plt.title(f"Sample {i + 1}")
    plt.axis('off')
plt.suptitle("Real Samples")
plt.tight_layout()
plt.show()
#plt.savefig('Real_samples.png')
#plt.close()

# Hyperparameters
input_channels = 1  # 2-component seismograms
num_layers = 5  # Number of layers in U-Net
learning_rate = 1e-3
num_epochs = 5000
num_samples = 10000  # Example number of samples
initial_filters = 16  # Number of filters in the first layer
filter_multiplier = 2  # Multiplier for the number of filters in each subsequent layer

# Set up parameters for the GAN
noise_dim = 25
output_channels = 1
hidden_dim = 256

# Initialize GRU-based generator and GRU-based discriminator
generator = ParallelConvGenerator(noise_dim, num_layers, initial_filters, output_channels, seq_length)
#generator = GRUGenerator(noise_dim, hidden_dim, num_layers, output_channels, seq_length)

# Load the generator checkpoint
checkpoint_path = './checkpoints/checkpoint_epoch_10.pth'
checkpoint = torch.load(checkpoint_path)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()  # Set the generator to evaluation mode

# Storage for all spectra and frequency data for generated and real noise
all_spectra_db_gen = []
all_spectra_db_real = []
all_freqs_gen = None
all_freqs_real = None

#sos = signal.butter(4, 2.5, 'hp', fs=sr, output='sos')

# Generate samples and compute power spectra for generated noise
for i in range(num_samples):
    noise = torch.randn(1, noise_dim)  # Generate random noise
    generated_trace = generator(noise).detach().numpy().flatten()  # Generate trace and detach from graph
    #generated_trace = signal.sosfilt(sos, generated_trace)
    generated_trace = generated_trace / np.std(generated_trace)  # Normalize the trace

    # Compute the power spectrum using MTSpec for generated data
    A = mt.MTSpec(generated_trace, dt=1/50)
    freqs_gen = A.rspec()[0].flatten()
    spectra_gen = A.rspec()[1].flatten()

    # Store frequency values only once for generated noise
    if all_freqs_gen is None:
        all_freqs_gen = freqs_gen

    # Convert power spectrum to dB for generated noise
    spectra_db_gen = 10 * np.log10(spectra_gen)
    all_spectra_db_gen.append(spectra_db_gen)

    # Print progress for every 1% of samples
    if (i + 1) % (num_samples // 100) == 0:
        print(f'Progress: {(i + 1) / num_samples * 100:.0f}% of generated samples processed.')

# Now compute power spectra for real noise stored in `noise_vec`
for i in range(noise_vec.shape[0]):
    real_trace = noise_vec[i, 0, :].flatten()  # Assuming noise_vec shape is (samples x channel x length)
    real_trace = real_trace / np.std(real_trace)  # Normalize the trace

    # Compute the power spectrum using MTSpec for real data
    B = mt.MTSpec(real_trace, dt=1/50)
    freqs_real = B.rspec()[0].flatten()
    spectra_real = B.rspec()[1].flatten()

    # Store frequency values only once for real noise
    if all_freqs_real is None:
        all_freqs_real = freqs_real

    # Convert power spectrum to dB for real noise
    spectra_db_real = 10 * np.log10(spectra_real)
    all_spectra_db_real.append(spectra_db_real)

    # Print progress for every 1% of samples
    if (i + 1) % (noise_vec.shape[0] // 100) == 0:
        print(f'Progress: {(i + 1) / noise_vec.shape[0] * 100:.0f}% of real noise samples processed.')

# End of loop - final print statement
print("All samples (generated and real) processed.")

# Convert lists to numpy arrays for easier manipulation
all_spectra_db_gen = np.array(all_spectra_db_gen)
all_spectra_db_real = np.array(all_spectra_db_real)

# Adjust frequency and spectra data (remove first element as before)
all_freqs_gen = (all_freqs_gen[1:])
all_spectra_db_gen = all_spectra_db_gen[:, 1:]

all_freqs_real = (all_freqs_real[1:])
all_spectra_db_real = all_spectra_db_real[:, 1:]

# Generate a 2D histogram of frequency and dB values for generated noise
freq_bins_gen = np.linspace(min(all_freqs_gen), max(all_freqs_gen), 100)  # 100 bins for frequency
db_bins_gen = np.linspace(np.min(all_spectra_db_gen), np.max(all_spectra_db_gen), 100)  # 100 bins for dB

hist_gen, xedges_gen, yedges_gen = np.histogram2d(all_freqs_gen.repeat(num_samples), all_spectra_db_gen.T.flatten(), bins=[freq_bins_gen, db_bins_gen])

# Generate a 2D histogram of frequency and dB values for real noise
freq_bins_real = np.linspace(min(all_freqs_real), max(all_freqs_real), 100)  # 100 bins for frequency
db_bins_real = np.linspace(np.min(all_spectra_db_real), np.max(all_spectra_db_real), 100)  # 100 bins for dB

hist_real, xedges_real, yedges_real = np.histogram2d(all_freqs_real.repeat(noise_vec.shape[0]), all_spectra_db_real.T.flatten(), bins=[freq_bins_real, db_bins_real])

# Plot the contour maps side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Contour plot for generated noise
X_gen, Y_gen = np.meshgrid(xedges_gen[:-1], yedges_gen[:-1])
axs[0].contourf(X_gen, Y_gen, hist_gen.T, levels=20, cmap='PuBu')
axs[0].set_title('Generated Noise: Contour Map of Frequency vs Power (dB)')
axs[0].set_xlabel('Frequency, Hz')
axs[0].set_ylabel('Power (dB)')
axs[0].set_ylim([-50, 10])
#axs[0].set_xlim([np.log10(1), np.log10(25)])
#axs[0].set_xlim([1, 20])
axs[0].grid(True)
#plt.colorbar(axs[0].collections[0], ax=axs[0], label='Count')

# Contour plot for real noise
X_real, Y_real = np.meshgrid(xedges_real[:-1], yedges_real[:-1])
axs[1].contourf(X_real, Y_real, hist_real.T, levels=20, cmap='PuBu')
axs[1].set_title('Real Noise: Contour Map of Frequency vs Power (dB)')
axs[1].set_xlabel('Frequency, Hz')
axs[1].set_ylabel('Power (dB)')
axs[1].set_ylim([-50, 10])
#axs[0].set_xlim([1, 20])
axs[1].grid(True)
#plt.colorbar(axs[1].collections[0], ax=axs[1], label='Count')

# Display the plots
plt.tight_layout()
plt.show()
