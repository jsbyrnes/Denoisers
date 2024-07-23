import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
import math
from sklearn.cluster import KMeans
from sklearn import mixture
from RISCluster_functions import models, networks

class PairedDataset(TensorDataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        data1 = self.dataset1[index]
        data2 = self.dataset2[index]
        return data1, data2

#data = sbd.OBST2024()
window = 256 #plus/minus
bs     = 32
lambda_separation = 0.2

nd = 5000
waveforms_clean = np.zeros((nd, window))
waveforms_dirty = np.zeros((nd, window))
for k in range(0, nd):

    if np.random.rand()<0.5:
        amp1 = 0.0
        amp2 = 1.0
    else:
        amp1 = 1.0
        amp2 = 0.0

    omega = 10**(-2 + np.random.rand()*1)
    x = np.random.rand()*np.sin(np.arange(0, window)*2*math.pi*omega)*amp1
    waveforms_clean[k, :] = x
    waveforms_dirty[k, :] = x

    for k2 in range(0, 20):

        ix = np.random.randint(window)
        #waveforms[k,:] += np.random.randn()*np.exp(-((ix-np.arange(0, window))**2)/(2*16))
        waveforms_dirty[k,ix] += np.random.uniform(low=-1.0, high=1.0, size=None)*amp2

        if np.random.rand() < 0.05:
            break

#clean_seismograms_tensor = torch.abs(torch.fft.fft(torch.tensor(waveforms_clean[:-100, :], dtype=torch.float32)))
#dirty_seismograms_tensor = torch.abs(torch.fft.fft(torch.tensor(waveforms_dirty[:-100, :], dtype=torch.float32)))
clean_seismograms_tensor = torch.tensor(waveforms_clean[:-100, :], dtype=torch.float32)
dirty_seismograms_tensor = torch.tensor(waveforms_dirty[:-100, :], dtype=torch.float32)

#valid_dirty = torch.abs(torch.fft.fft(dirty_seismograms_tensor[-100:, :].unsqueeze(1)))
#valid_clean = torch.abs(torch.fft.fft(clean_seismograms_tensor[-100:, :].unsqueeze(1)))
valid_dirty = dirty_seismograms_tensor[-100:, :].unsqueeze(1)
valid_clean = clean_seismograms_tensor[-100:, :].unsqueeze(1)

vn = 100
# Dataset and DataLoader setup
dataset_c = TensorDataset(clean_seismograms_tensor)
dataset_d = TensorDataset(dirty_seismograms_tensor)

#paireddata = PairedDataset(dataset_c, dataset_d)

dataloader = DataLoader(dataset_d, batch_size=bs, shuffle=True)
#dataloader_d = DataLoader(dataset_d, batch_size=32, shuffle=True)

class WaveDecomp(nn.Module):
    def __init__(self, input_channels=1, base_filters=4, sequence_length=256, kernel=3):
        super(WaveDecomp, self).__init__()
        # Convolutional Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=kernel, stride=2, padding=int((kernel-1)/2)),
            nn.ELU(),
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=kernel, stride=2, padding=int((kernel-1)/2)),
            nn.ELU(),
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=kernel, stride=2, padding=int((kernel-1)/2)),
        )
        
        # Calculate the output size of the last convolution layer
        self.final_conv_length = sequence_length // (2**3)  # Assuming three layers of stride 2 reductions
        
        # Linear layer to flatten the output from the convolutional layers
        self.flat_layer = nn.Linear(base_filters * 4 * self.final_conv_length, base_filters * 4 * self.final_conv_length)

        # Convolutional Decoder - mirroring the encoder but in reverse
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_filters*4, base_filters * 2, kernel_size=kernel, stride=2, padding=int((kernel-1)/2), output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(base_filters * 2, base_filters, kernel_size=kernel, stride=2, padding=int((kernel-1)/2), output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(base_filters, input_channels, kernel_size=kernel, stride=2, padding=int((kernel-1)/2), output_padding=1)
        )

    def encode(self, x):
        # Apply convolutional layers

        x = self.encoder(x)
        # Flatten output for the linear layer
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.flat_layer(x)  # Apply the linear layer
        
        return x

    def forward(self, x):
        x = self.encode(x)
        # Decode
        x = x.view(x.size(0), -1, self.final_conv_length)  # Reshape to match the expected input for the decoder
        x = self.decoder(x)
        return x
            
class SoftDifferentiableKMeans(nn.Module):
    def __init__(self, n_clusters, n_features, requires_grad=True, device='cpu'):
        super(SoftDifferentiableKMeans, self).__init__()
        self.centroids = nn.Parameter(torch.randn(n_clusters, n_features, device=device), requires_grad=requires_grad)

    def forward(self, x):
        distances = torch.cdist(x, self.centroids, p=2)
        soft_assignments = F.softmax(-distances, dim=1)
        return soft_assignments, distances

    def centroid_diversity_loss(self):
        centroid_distances = torch.cdist(self.centroids, self.centroids, p=2)
        # Avoid division by zero and self-comparison by adding a small value to the diagonal
        eye = torch.eye(centroid_distances.size(0), device=centroid_distances.device)
        centroid_distances += eye * 1e-8
        # Penalize the inverse of distances
        penalty = 1.0 / centroid_distances
        return penalty.sum()

class ConvToScalar(nn.Module):
    def __init__(self, base_filt=16, drop=0.2, lstm_features=50):
        super(ConvToScalar, self).__init__()
        self.drop = drop
        self.base_filt = base_filt

        # Convolutional layers
        self.enc_conv1 = nn.Conv1d(1, self.base_filt, kernel_size=3, stride=2, padding=1)
        self.drop1 = nn.Dropout(drop)
        self.enc_conv2 = nn.Conv1d(self.base_filt, self.base_filt * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(self.base_filt * 2)
        self.drop2 = nn.Dropout(drop)
        self.enc_conv3 = nn.Conv1d(self.base_filt * 2, self.base_filt * 2, kernel_size=3, stride=2, padding=1)

        # LSTM layer
        self.lstm = nn.LSTM(self.base_filt * 2, lstm_features, batch_first=True)

        # Global average pooling modified to handle the output of LSTM
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer to produce a scalar
        self.fc = nn.Linear(lstm_features, 1)

    def forward(self, x):
        # Convolutional layers
        x = F.elu(self.enc_conv1(x))
        x = self.drop1(x)
        x = F.elu(self.enc_conv2(x))
        x = self.drop2(x)
        x = F.elu(self.enc_conv3(x))

        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # Change to (batch, seq_len, features) for LSTM

        # LSTM layer
        x, _ = self.lstm(x)

        # Using the last hidden state
        x = x[:, -1, :]

        # Passing through a fully connected layer
        x = self.fc(x)

        # Sigmoid activation to output a probability
        x = torch.sigmoid(x)
        
        return x.view(-1)  # Flatten to batch of scalars

def weights_init(m):
    # Apply initializations based on the type of each module
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
        # Initialize weights for convolutional layers
        nn.init.normal_(m.weight.data, 0.0, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        # Initialize weights and biases for batch normalization layers
        nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        # Initialize weights for linear layers
        nn.init.normal_(m.weight.data, 0.0, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Initialize models
input_size = clean_seismograms_tensor.shape[1]  # Number of features in each seismogram
n_clusters = 2
model = WaveDecomp()
model.apply(weights_init)

clusterer = networks.DEC(2)

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=4, eps=1e-10, threshold=0.00001)

criterion = nn.MSELoss()

# Training Loop
n_epochs = 200
for epoch in range(n_epochs):
    model.train()

    for real_samples_d in dataloader:
        
        optimizer.zero_grad()

        #Train the generator
        real_samples_d = real_samples_d[0].unsqueeze(1)
        outputs = model(real_samples_d)

        loss = criterion(outputs, real_samples_d)

        # Backpropagate the loss
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(clustering.parameters()), 1.0)
        optimizer.step()

    # Example to visualize one generated seismogram
    model.eval()

    outputs = model(valid_dirty)
    loss = criterion(outputs, valid_dirty)

    #scheduler.step(loss)
    #scheduler2.step(loss2)
    print(f'Epoch {epoch}, Validation Loss {loss.item()}')

    for k in range(0, 10):
        plt.plot(k + 0.75*valid_dirty[k,0, :].detach().numpy(), 'r')
        plt.plot(k + 0.75*outputs[k,0,:].detach().numpy(), 'k', linewidth=0.66)

    plt.title('Autoencoded Examples')
    plt.savefig('AutoEncoded.png')
    plt.close()

    scheduler.step(loss)

    if scheduler.get_last_lr()[0] < 1e-8:
        print('Learning rate reduce below minimum')
        break

#now encode everything and cluster it
encoded_features = []

with torch.no_grad():  # No gradients needed
    encoded_features = model.encode(dirty_seismograms_tensor.unsqueeze(1))

# Concatenate all encoded features obtained from the batches
encoded_features = encoded_features.numpy()  # Convert to NumPy array for K-means

kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(np.abs(encoded_features))

#clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
#clf.fit(encoded_features**2)
#clf.fit_predict(encoded_features**2)

# Convert labels to torch tensor for operations
labels = torch.tensor(labels, dtype=torch.bool).numpy()

ix1 = np.argwhere(labels)
ix2 = np.argwhere(~labels)

for k in range(10):
    plt.subplot(2,1,1)
    plt.plot(0.5*dirty_seismograms_tensor[ix1[k].item(), :] + k)
    plt.subplot(2,1,2)
    plt.plot(0.5*dirty_seismograms_tensor[ix2[k].item(), :] + k)
plt.show()
"""
class Discriminator(nn.Module):
    def __init__(self, input_size, base_filters=64):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            # Hardcoded to single-channel input
            nn.Conv1d(1, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_filters * 4, base_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 8),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(base_filters * 8 * (self.input_size // 16), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def calculate_model_loss(model, real_samples, lambda_entropy=0.01):
    Calculate the total loss for the model, including reconstruction and separation components.

    Args:
    model (nn.Module): The model being trained or evaluated.
    real_samples (torch.Tensor): The ground truth samples for reconstruction comparison.
    lambda_separation (float): Weighting factor for the separation loss component.

    Returns:
    torch.Tensor: The total loss value.
    # Perform model prediction
    outputs = model(real_samples)

    # Assuming the model's forward method or another method returns the centroids along with the outputs
    #features = model.encode(real_samples)  # Extract features used for clustering
    #cluster_logits = model.cluster_net(features)
    #cluster_assignments = torch.argmax(cluster_logits, dim=1)
    
    # Update centroids based on current batch (if required and feasible)
    #centroids = update_centroids(features.detach(), cluster_assignments.detach(), model.n_clusters)

    # Calculate reconstruction loss
    # Assuming outputs are stacked along the cluster dimension and real_samples need to be reconstructed
    pred1 = outputs[:, 0, :]
    #pred2 = outputs[:, 1, 0, :]

    #reconstruction_loss = criterion(pred1 + pred2, real_samples.squeeze(1))
    reconstruction_loss = criterion(pred1, real_samples.squeeze(1))

    total_loss = reconstruction_loss# + lambda_entropy*cosine_similarity_penalty(pred1, pred2) + (0.25/128)*(pred1.norm(p=2) + pred2.norm(p=2))

    return total_loss

"""