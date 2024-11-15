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

from WaveDecompNet import autoencoder_1D_models_torch as wdn

warnings.filterwarnings('ignore')

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
    def __init__(self, input_channels, latent_dim, num_layers, initial_filters, filter_multiplier, input_length):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, num_layers, initial_filters, filter_multiplier)
        self.decoder = Decoder(num_layers, initial_filters, filter_multiplier, input_channels)

        # Calculate the size of the final output of the encoder
        self.encoder_output_size = self.final_conv_output_size(input_length, num_layers)

        # Convolutional layers to compute mu and logvar
        self.conv_mu = nn.Conv1d(self.encoder.final_channels, latent_dim, kernel_size=1, stride=1, padding=0)
        self.conv_logvar = nn.Conv1d(self.encoder.final_channels, latent_dim, kernel_size=1, stride=1, padding=0)

        # Convolutional layer to project latent space back to feature map size
        self.conv_initial = nn.Conv1d(latent_dim, self.encoder.final_channels, kernel_size=1, stride=1, padding=0)

    def final_conv_output_size(self, input_length, num_layers):
        # Calculate the size after convolutional layers given stride=2
        return input_length // (2 ** num_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hh = self.encoder(x)

        # Compute mu and logvar
        mu = self.conv_mu(hh)
        logvar = self.conv_logvar(hh)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Project back to feature map size for the decoder
        z = self.conv_initial(z)

        # Decode
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def sample(self, num_samples):
        # Sample from the latent space
        z = torch.randn(num_samples, self.decoder.layers[0].in_channels, self.encoder_output_size).to(next(self.parameters()).device)
        samples = self.decoder(z)
        return samples

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

# Function to generate new samples from the trained VAE
def generate_samples(model, num_samples, latent_dim):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        # Sample from the standard normal distribution
        z = torch.randn(num_samples, latent_dim)
        # Generate samples using the decoder
        generated_samples = model.decoder(z)
    return generated_samples

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

    chunks = 10000

    ind = 0

    data = sbd.OBS(dimension_order="NCW", component_order="Z12H") #sampling rate doesn't work
    total, _ = data.metadata.shape

    sos = signal.butter(4, 2.5, 'hp', fs=sr, output='sos')

    data_vec = np.array(())

    while ind < total:
    #while ind < 15000:

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

    #remove microtraces
    x = np.std(noise_vec, axis=2)
    ind = np.logical_and(x[:, 0]/x[:,1] > 0.05, x[:, 1]/x[:,0] > 0.05)
    noise_vec = noise_vec[ind, :, :]

    return data_vec, noise_vec

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Function to generate new samples from the trained VAE
def generate_samples(model, num_samples, latent_dim):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        # Sample from the standard normal distribution
        z = torch.randn(num_samples, latent_dim)
        # Generate samples using the decoder
        generated_samples = model.decoder(z)
    return generated_samples

# Function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, loss, path="vae_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

np.random.seed(3) #same seed as used during training for an in-house check
torch.manual_seed(3)

batch_size = 32
sr = 50
nlen = 512 # window length
print_interval = 10  # Print every x batches

print('Constructing the dataset')
data_vec, noise_vec = format_OBS(int(nlen/2), 10)

dataset = MyDataset(data_vector_1=noise_vec, nlen=nlen)
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

nsize = noise_vec.shape[0]
print(str(noise_vec.shape[0]) + ' noise traces passed length threshold')
val_size = int(0.1 * nsize)  # 10% for validation
train_size = nsize - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader_train = DataLoader(train_dataset, batch_size = 32, shuffle=True)
dataloader_valid = DataLoader(val_dataset, batch_size = 32, shuffle=False) 

# Hyperparameters
input_channels = 2  # 2-component seismograms
latent_dim = 20
num_layers = 5  # Number of layers in U-Net
learning_rate = 1e-3
num_epochs = 5000
num_samples = 100  # Example number of samples
initial_filters = 16  # Number of filters in the first layer
filter_multiplier = 2  # Multiplier for the number of filters in each subsequent layer

# Model, optimizer, and training
model = VAE(input_channels, latent_dim, num_layers, initial_filters, filter_multiplier, nlen)
#model = VAE()
print("Model initialized with " + str(count_parameters(model)) + " trainable parameters")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=10, verbose=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_bce = 0
    total_kld = 0

    #training loop
    for batch in dataloader_train:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch['data'])
        #recon_batch = model(batch['data'])
        
        # Compute losses
        #loss = F.mse_loss(recon_batch, batch['data'], reduction='mean')
        BCE = F.mse_loss(recon_batch, batch['data'], reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = BCE + KLD

        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()

    #validation loop
    for batch in dataloader_valid:
        recon_batch, mu, logvar = model(batch['data'])
        
        # Compute losses
        #loss = F.mse_loss(recon_batch, batch['data'], reduction='mean')
        BCE = F.mse_loss(recon_batch, batch['data'], reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = BCE + KLD

        # Accumulate the loss for logging
        total_loss += loss.item()
        total_bce += BCE.item()
        total_kld += KLD.item()

    # Step the learning rate scheduler
    scheduler.step(total_loss)

    # Save the model checkpoint
    if epoch % print_interval == 0:
        avg_loss = total_loss / len(dataloader_valid)
        avg_bce = total_bce / len(dataloader_valid)
        avg_kld = total_kld / len(dataloader_valid)
        print(f'Epoch {epoch}, Total Loss: {avg_loss}, Reconstruction Loss: {avg_bce}, KL Divergence: {avg_kld}')

        # Plot and save examples
        for batch in dataloader_valid:
            recon_batch, _, _ = model(batch['data'])
            plot_examples(epoch, batch['data'][:5], recon_batch[:5], num_examples=5, filename='generated_examples_epoch_{epoch}.png')
            break

        if scheduler.get_last_lr()[0] < 1e-8:
            print('Learning rate reduce below minimum')
            break

#save_checkpoint(model, optimizer, epoch, total_loss, path=f"vae_checkpoint_epoch_{epoch}.pth")
torch.save(model.state_dict(), 'VAE_v3.pt')

print("Training complete")


