import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from scipy import signal
import seisbench.data as sbd
import warnings

warnings.filterwarnings('ignore')
# Function to generate new samples from the trained VAE
def generate_samples(model, num_samples, var, mean, latent_dim=2, latent_length=16):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        # Sample from the standard normal distribution
        z = torch.randn(num_samples, latent_dim, latent_length)*torch.randn(1) + torch.randn(1)
        generated_samples = model.decoder(z)
    return generated_samples

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ELU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))

class Conv1dTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, final_layer=False):
        super(Conv1dTransposeBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.relu = nn.ELU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.final_layer = final_layer

    def forward(self, x):
        if self.final_layer:
            return self.conv_transpose(x)  # No ReLU and BatchNorm for the final layer
        else:
            return self.bn(self.relu(self.conv_transpose(x)))

class Encoder(nn.Module):
    def __init__(self, input_channels, num_layers, initial_filters, filter_multiplier):
        super(Encoder, self).__init__()
        layers = []
        self.num_layers = num_layers
        self.initial_filters = initial_filters
        self.filter_multiplier = filter_multiplier
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else initial_filters * (filter_multiplier ** (i - 1))
            out_channels = initial_filters * (filter_multiplier ** i)
            layers.append(Conv1dBlock(in_channels, out_channels))
        self.layers = nn.ModuleList(layers)
        self.final_channels = initial_filters * (filter_multiplier ** (num_layers - 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, initial_filters, filter_multiplier, output_channels):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.initial_filters = initial_filters
        self.filter_multiplier = filter_multiplier
        layers = []
        for i in range(num_layers):
            in_channels = initial_filters * (filter_multiplier ** (num_layers - i - 1))
            out_channels = initial_filters * (filter_multiplier ** (num_layers - i - 2)) if i != num_layers - 1 else output_channels
            layers.append(Conv1dTransposeBlock(in_channels, out_channels, final_layer=(i == num_layers - 1)))
        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        h = z
        for layer in self.layers:
            h = layer(h)
        return h

class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim, num_layers, initial_filters, filter_multiplier):
    #def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, num_layers, initial_filters, filter_multiplier)
        self.decoder = Decoder(num_layers, initial_filters, filter_multiplier, input_channels)
        
        ########
        #These are for if you want to use the code from WaveDecomNet
        #bottleneck = torch.nn.LSTM(64, 32, 2, bidirectional=True,
        #                       batch_first=True)

        #self.encoder = SeismogramEncoder()
        #self.decoder = SeismogramDecoder(bottleneck=bottleneck)
        ########

        self.conv_mu = nn.Conv1d(self.encoder.final_channels, latent_dim, kernel_size=3, stride=1, padding=1)
        self.conv_logvar = nn.Conv1d(self.encoder.final_channels, latent_dim, kernel_size=3, stride=1, padding=1)
        self.conv_initial = nn.Conv1d(latent_dim, self.encoder.final_channels, kernel_size=3, stride=1, padding=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_data = self.encoder(x)
        mu = self.conv_mu(enc_data)
        logvar = self.conv_logvar(enc_data)
        z = self.reparameterize(mu, logvar)
        z = self.conv_initial(z)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

class MyDataset(Dataset):
    def __init__(self, data_vector_1, nlen):
        self.data_vector_1 = torch.tensor(data_vector_1, dtype=torch.float32)
        self.nlen = nlen

    def __len__(self):
        return len(self.data_vector_1)

    def __getitem__(self, idx):

        x = self.data_vector_1[idx]
        #x *= 1/(torch.max(torch.abs(x)) + 0.00001) #zero protect
        x *= 1/(torch.std(x) + 0.00001) #zero protect

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

    chunks = 100

    ind = 0

    data = sbd.OBS(dimension_order="NCW", component_order="Z12H") #sampling rate doesn't work
    total, _ = data.metadata.shape

    sos = signal.butter(4, 2.5, 'hp', fs=sr, output='sos')

    data_vec = np.array(())

    #while ind < total:
    while ind < 1500:

        if ind + chunks < total:
            wf_vec_chunk  = data.get_waveforms(np.arange(ind, ind + chunks), sampling_rate=50)[:,(0,3),:]
        else:
            wf_vec_chunk  = data.get_waveforms(np.arange(ind, total), sampling_rate=50)[:,(0,3),:]
        shp = wf_vec_chunk.shape

        data_chunk = np.zeros((shp[0], 2, nlen*2))
        noise_chunk = np.zeros((shp[0], 2, nlen*2))

        #filter
        for k in range(shp[0]):
            wf_vec_chunk[k,0,:] = signal.sosfilt(sos, wf_vec_chunk[k, 0, :])
            wf_vec_chunk[k,1,:] = signal.sosfilt(sos, wf_vec_chunk[k, 1, :])
            #wf_vec_chunk[k,2,:] = signal.sosfilt(sos, wf_vec_chunk[k, 2, :])

            #get the sampling rate
            sp = data.metadata['trace_sampling_rate_hz'][ind + k]
            #get the new index value
            itp = np.round(data.metadata['trace_p_arrival_sample'][ind + k]*(50/sp)).astype(int)

            #clip noise if possible, middle section of it
            if itp > 2*nlen:
                ix = np.round((itp - 2*nlen)/2).astype(int)

                noise_chunk[k, 0, :] = wf_vec_chunk[k, 0, ix:ix+2*nlen]
                noise_chunk[k, 1, :] = wf_vec_chunk[k, 1, ix:ix+2*nlen]
                #noise_chunk[k, 2, :] = wf_vec_chunk[k, 2, ix:ix+2*nlen]

                #amp = np.max(np.abs(np.hstack( (noise_chunk[k, 0, :], noise_chunk[k, 1, :], noise_chunk[k, 2, :]) ))) + 0.001
                #noise_chunk[k, :, :] = noise_chunk[k, :, :]/amp

            if itp > 2*nlen: #currently just skips if too close

                #snr = getsnr(wf_vec_chunk[k, 0, (itp - nlen):(itp + nlen)])
                snr = 11
                
                if snr > min_snr:
                    data_chunk[k, 0, :] = wf_vec_chunk[k, 0, (itp - nlen):(itp + nlen)]
                    data_chunk[k, 1, :] = wf_vec_chunk[k, 1, (itp - nlen):(itp + nlen)]
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

    #remove the microverticals
    x = np.std(noise_vec, axis=2)
    ind = np.logical_and(x[:, 0]/x[:,1] > 0.05, x[:, 1]/x[:,0] > 0.05)
    noise_vec = noise_vec[ind, :, :]

    return data_vec, noise_vec

# Function to plot and save examples
def plot_examples(epoch, original, reconstructed, num_examples=5, filename='generated_examples_epoch_{epoch}.png'):
    fig, axs = plt.subplots(num_examples, 2, figsize=(15, 10))
    for i in range(num_examples):
        # Plot original channel 1
        axs[i, 0].plot(original[i, 0].cpu().numpy(), label='Original Channel 1')
        axs[i, 0].plot(reconstructed[i, 0].cpu().detach().numpy(), label='Reconstructed Channel 1')
        axs[i, 0].legend()
        # Plot original channel 2
        axs[i, 1].plot(original[i, 1].cpu().numpy(), label='Original Channel 2')
        axs[i, 1].plot(reconstructed[i, 1].cpu().detach().numpy(), label='Reconstructed Channel 2')
        axs[i, 1].legend()
    plt.tight_layout()
    plt.savefig(filename.format(epoch=epoch))
    plt.close(fig)

# Hyperparameters

#v1 2 channels, LD 20, depth 5, 16 filters
#v1 2 channels, LD 2, depth 5, 16 filters
#v1 2 channels, LD 2, depth 5, 2 filters

input_channels = 2  # 2-component seismograms
latent_dim = 20
num_layers = 5  # Number of layers in U-Net
initial_filters = 16  # Number of filters in the first layer
filter_multiplier = 2  # Multiplier for the number of filters in each subsequent layer
nlen = 512 # window length

num_examples = 5
sr = 50

model = VAE(input_channels, latent_dim, num_layers, initial_filters, filter_multiplier)
model.load_state_dict(torch.load('VAE_v2.pt'))

print('Constructing the dataset')
#data_vec, noise_vec, itp_vec = format_OBST()
data_vec, noise_vec = format_OBS(int(nlen/2), 10)

nsize = data_vec.shape[0]
print(str(nsize) + ' signal traces passed snr threshold')
print(str(noise_vec.shape[0]) + ' noise traces passed length threshold')

dataset = MyDataset(data_vector_1=noise_vec, nlen=nlen)
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

for batch in dataloader:
    recon_batch, _, _ = model(batch['data'])
    plot_examples(999, batch['data'][:5], recon_batch[:5], num_examples=5, filename='generated_examples_epoch_{epoch}.png')
    break

reconstructed = generate_samples(model, num_examples, latent_dim)

t = np.arange(512)/50

fig, axs = plt.subplots(num_examples, 2, figsize=(15, 10))
for i in range(num_examples):
    # Plot original channel 1
    axs[i, 0].plot(t, reconstructed[i, 0, :], label='Generated Vertical')
    # Plot original channel 2
    axs[i, 1].plot(t, reconstructed[i, 1, :], label='Generated Pressure')
plt.tight_layout()
plt.show()
