import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv, GATConv, TopKPooling
from torch.utils.data import Dataset, DataLoader, random_split
#import pywt
import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Define custom transformer model with positional encoding
class CustomTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=6, num_heads=8, hidden_dim=512, dropout=0.1, max_len=128):
        super(CustomTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        input_tensor = self.pos_encoder(input_tensor)
        output = self.transformer_encoder(input_tensor)
        output = self.linear(output)
        return output

class UNetv1(nn.Module):
    def __init__(self, drop=0.1, ncomp=1, fac=1):
        super(UNetv1, self).__init__()
        
        if ncomp == 1:
            in_channels = 2
        elif ncomp == 3:
            in_channels = 6
        elif ncomp == 4:
            in_channels = 8

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

        self.transformer = CustomTransformer(128 * fac, num_layers=3, num_heads=2, hidden_dim=128 * fac, dropout=drop)

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

        self.final = nn.Conv2d(8 * fac, ncomp, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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

        #bottleneck
        v = enc5b.squeeze()

        #correction for nbatch = 1
        if len(v.shape)==2:
            v = v.unsqueeze(0)

        transformed = self.transformer(torch.permute(v, [0, 2, 1 ]))
        transformed = torch.permute(transformed, [0, 2, 1]).unsqueeze(2)

        dec4 = F.elu(self.bn4u(self.upconv4(transformed)))
        dec4 = F.elu(self.bn4u(self.upconv4(enc5b)))
        dec4 = torch.cat([dec4, enc4b], dim=1)
        dec4 = F.elu(self.bn4u2(self.conv4u(dec4)))
        
        dec3 = F.elu(self.bn3u(self.upconv3(dec4)))
        dec3 = torch.cat([dec3, enc3b], dim=1)
        dec3 = F.elu(self.bn3u2(self.conv3u(dec3)))
        
        dec2 = F.elu(self.bn2u(self.upconv2(dec3)))
        dec2 = torch.cat([dec2, enc2b], dim=1)
        dec2 = F.elu(self.bn2u2(self.conv2u(dec2)))
        
        dec1 = F.elu(self.bn1u(self.upconv1(dec2)))
        dec1 = torch.cat([dec1, enc1b], dim=1)
        dec1 = F.elu(self.bn1u2(self.conv1u(dec1)))

        out = self.final(dec1)
        out = self.sigmoid(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def data_2_input(stftinput1,stftinput2,stftinput3,stftinput4,norm_input):
    
    if norm_input:
        stftinput1=stftinput1/torch.max(torch.abs(stftinput1) + 0.000001)#zero protect
        stftinput2=stftinput2/torch.max(torch.abs(stftinput2) + 0.000001)#all zeros is not a bug
        stftinput3=stftinput3/torch.max(torch.abs(stftinput3) + 0.000001)
        stftinput4=stftinput4/torch.max(torch.abs(stftinput4) + 0.000001)

    tmp = torch.zeros((8, stftinput1.shape[0],stftinput1.shape[1]))

    tmp[0, :, :] = torch.real(stftinput1)
    tmp[1, :, :] = torch.imag(stftinput1)
    tmp[2, :, :] = torch.real(stftinput2)
    tmp[3, :, :] = torch.imag(stftinput2)
    tmp[4, :, :] = torch.real(stftinput3)
    tmp[5, :, :] = torch.imag(stftinput3)
    tmp[6, :, :] = torch.real(stftinput4)
    tmp[7, :, :] = torch.imag(stftinput4)

    return tmp

class MyDataset(Dataset):
    def __init__(self, data_vector_1, data_vector_2, nlen, nperseg):
        self.data_vector_1 = torch.tensor(data_vector_1, dtype=torch.float32)
        self.data_vector_2 = torch.tensor(data_vector_2, dtype=torch.float32)
        self.nlen = nlen
        self.nperseg = nperseg
        self.window = torch.hann_window(self.nperseg)

    def __len__(self):
        return len(self.data_vector_1)

    def __getitem__(self, idx):
        vector_1 = self.data_vector_1[idx]

        start_d = int(np.random.randint(low=0, high=self.nlen/2))

        vector_1 = vector_1[:, start_d:start_d+self.nlen]

        idx2 = np.random.randint(0, len(self.data_vector_2)-1)
        vector_2 = self.data_vector_2[idx2]

        if np.random.rand() < 0.2: #zerod signals are common
            amp=0.0
        else:
            amp = np.random.uniform(0,2)*(-1)**np.random.randint(low=1, high=10) # could still be tiny
        
        amp2 = np.random.uniform(0.1,2)*(-1)**np.random.randint(low=1, high=10) # could still be tiny
        
        start = int(np.random.randint(low=0, high=self.nlen/2))

        vector_2 = vector_2[:, start:start+self.nlen]

        vector_1 *= amp
        vector_2 *= amp2

        #This is probably the bottleneck
        stftsig1 = torch.stft(vector_1[0,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftnoise1 = torch.stft(vector_2[0,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftinput1 = stftsig1+stftnoise1

        stftsig2 = torch.stft(vector_1[1,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftnoise2 = torch.stft(vector_2[1,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftinput2 = stftsig2+stftnoise2

        stftsig3 = torch.stft(vector_1[2,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftnoise3 = torch.stft(vector_2[2,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftinput3 = stftsig3+stftnoise3

        stftsig4 = torch.stft(vector_1[3,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftnoise4 = torch.stft(vector_2[3,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftinput4 = stftsig4+stftnoise4

        combined = data_2_input(stftinput1,stftinput2,stftinput3,stftinput4,True)

        mask = torch.zeros(4, combined.shape[1], combined.shape[2])

        # batch outputs are real valued signal masks
        rat1=torch.nan_to_num(torch.abs(stftnoise1)/torch.abs(stftsig1),posinf=1e20)
        mask[0, : :]=1/(1+rat1) # signal mask
        rat2=torch.nan_to_num(torch.abs(stftnoise2)/torch.abs(stftsig2),posinf=1e20)
        mask[1, : :]=1/(1+rat2) # signal mask
        rat3=torch.nan_to_num(torch.abs(stftnoise3)/torch.abs(stftsig3),posinf=1e20)
        mask[2, : :]=1/(1+rat3) # signal mask
        rat4=torch.nan_to_num(torch.abs(stftnoise4)/torch.abs(stftsig4),posinf=1e20)
        mask[3, : :]=1/(1+rat4) # signal mask

        sample = {'mask': mask, 'combined': combined}

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
            wf_vec_chunk  = data.get_waveforms(np.arange(ind, ind + chunks), sampling_rate=50)
        else:
            wf_vec_chunk  = data.get_waveforms(np.arange(ind, total), sampling_rate=50)
        shp = wf_vec_chunk.shape

        data_chunk = np.zeros((shp[0], 4, nlen*2))
        noise_chunk = np.zeros((shp[0], 4, nlen*2))

        #filter
        for k in range(shp[0]):
            wf_vec_chunk[k,0,:] = signal.sosfilt(sos, wf_vec_chunk[k, 0, :])
            wf_vec_chunk[k,1,:] = signal.sosfilt(sos, wf_vec_chunk[k, 1, :])
            wf_vec_chunk[k,2,:] = signal.sosfilt(sos, wf_vec_chunk[k, 2, :])
            wf_vec_chunk[k,3,:] = signal.sosfilt(sos, wf_vec_chunk[k, 3, :])

            #get the sampling rate
            sp = data.metadata['trace_sampling_rate_hz'][ind + k]
            #get the new index value
            itp = np.round(data.metadata['trace_p_arrival_sample'][ind + k]*(50/sp)).astype(int)

            #clip noise if possible, middle section of it
            if itp > 2*nlen:
                ix = np.round((itp - 2*nlen)/2).astype(int)

                noise_chunk[k, 0, :] = wf_vec_chunk[k, 0, ix:ix+2*nlen]
                noise_chunk[k, 1, :] = wf_vec_chunk[k, 1, ix:ix+2*nlen]
                noise_chunk[k, 2, :] = wf_vec_chunk[k, 2, ix:ix+2*nlen]
                noise_chunk[k, 3, :] = wf_vec_chunk[k, 3, ix:ix+2*nlen]

                amp = np.max(np.abs(np.hstack( (noise_chunk[k, 0, :], noise_chunk[k, 1, :], noise_chunk[k, 2, :], noise_chunk[k, 3, :]) ))) + 0.001
                noise_chunk[k, :, :] = noise_chunk[k, :, :]/amp

            if itp > 2*nlen: #currently just skips if too close

                snr = getsnr(wf_vec_chunk[k, 0, (itp - nlen):(itp + nlen)])

                if snr > min_snr:
                    data_chunk[k, 0, :] = wf_vec_chunk[k, 0, (itp - nlen):(itp + nlen)]
                    data_chunk[k, 1, :] = wf_vec_chunk[k, 1, (itp - nlen):(itp + nlen)]
                    data_chunk[k, 2, :] = wf_vec_chunk[k, 2, (itp - nlen):(itp + nlen)]
                    data_chunk[k, 3, :] = wf_vec_chunk[k, 3, (itp - nlen):(itp + nlen)]

                    amp = np.max(np.abs(np.hstack( (data_chunk[k, 0, :], data_chunk[k, 1, :], data_chunk[k, 2, :], data_chunk[k, 3, :]) ))) + 0.001
                    data_chunk[k, :, :] = data_chunk[k, :, :]/amp

        data_ind = np.sum(np.sum(data_chunk**2,axis=2), axis=1) > 0.0# + ~np.isnan(np.sum(np.sum(data_chunk,axis=2), axis=1))
        noise_ind = np.sum(np.sum(noise_chunk**2,axis=2), axis=1) > 0.0# + ~np.isnan(np.sum(np.sum(noise_chunk,axis=2), axis=1))

        if data_vec.size == 0:
            data_vec  = data_chunk[data_ind, :, :]
            noise_vec = noise_chunk[noise_ind, :, :]
        else:
            data_vec = np.vstack( (data_vec, data_chunk[data_ind, :, :]) )
            noise_vec = np.vstack( (noise_vec, noise_chunk[noise_ind, :, :]) )

        ind += chunks

    return data_vec, noise_vec

# Hyperparameters
model_type = 'UNet'
model_v = 'v1'
train = True

batch_size = 32
learning_rate = 0.001
num_epochs = 100  # Default to 3 epochs
num_components = 4
hidden_dim = 1024*num_components #transformer only
sr = 50
fac = 1# model size
eps = 1e-9 # epsilon
drop=0.0 # model drop rate
nlen=256# window length
nperseg=31 # param for STFT
noverlap=30 # param for STFT
norm_input=True # leave this alone
print_interval = 100  # Print every x batches
cmap='PuRd'

np.random.seed(3)
torch.manual_seed(3)

#for later plotting
f, t, _ = signal.stft(np.zeros(nlen),fs=sr,nperseg=31, noverlap=30)

model = UNetv1(drop=drop, ncomp=num_components, fac=fac) #pytorch version of amanda's tensorflow UNet
print("CNN-UNetT" + model_v + " initialized with " + str(count_parameters(model)) + " trainable parameters")
if not train:
    model.load_state_dict(torch.load('./model_test-CNN_UNetT_fac' + str(fac) + model_v + '.pt'))
model = model.float()

print('Constructing the dataset')
#data_vec, noise_vec, itp_vec = format_OBST()
data_vec, noise_vec = format_OBS(nlen, 10)

nsize = data_vec.shape[0]
print(str(nsize) + ' signal traces passed snr threshold')
print(str(noise_vec.shape[0]) + ' noise traces passed length threshold')
val_size = int(0.1 * nsize)  # 5% for validation
train_size = nsize - val_size

dataset = MyDataset(data_vector_1=data_vec, data_vector_2=noise_vec, nlen=nlen, nperseg=nperseg)

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader_train = DataLoader(train_dataset, batch_size = 32, shuffle=True)
dataloader_valid = DataLoader(val_dataset, batch_size = 32, shuffle=False)

for batch in dataloader_valid: #for plotting
    valbatch = batch
    break

criterion = nn.L1Loss() #nn.L1Loss(), nn.MSELoss()

training_hist = np.array([])
val_hist      = np.array([])

if train:

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=5, eps=1e-8, threshold=0.00001)

    # Training loop with shuffling
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0  # Counter for batches

        #This step is slow. But hardly the bottle-neck compared with training. Do more batches if too slow, data is heavily augmented.
        print('-----> Epochs# ' + str(epoch+1))
        model.train()

        for batch in dataloader_train:
            n = batch['combined'].shape[0]

            optimizer.zero_grad()
            
            out = model(batch['combined'])

            loss = criterion(out, batch['mask'])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1  # Increment batch counter

            if batch_count % print_interval == 0:  # Print every `print_interval` batches
                print(f'Epoch {epoch + 1}, Batch {batch_count}, Loss: {(running_loss/batch_count):.5f}')

        #validation loss
        model.eval()

        val_loss = 0.0
        c = 0

        for batch in dataloader_valid:

            out = model(batch['combined'])
            loss = criterion(out, batch['mask'])

            val_loss += loss.item()
            c += 1

        val_loss /= c

        #trace_in = np.zeros(10, nlen)
        #trace_pred = np.zeros(10, nlen)
        #trace_true = np.zeros(10, nlen)

        #for k in range(10):
        mask = model(valbatch['combined'][0,:,:,:].unsqueeze(0)).detach().numpy()[0,0,:,:]
        mask_true = valbatch['mask'][0,0,:,:]
        trace = valbatch['combined'][0,0,:,:] + valbatch['combined'][0,1,:,:]*1j

        trace_in = torch.istft(trace,nperseg, hop_length=1, window=torch.hann_window(nperseg)).detach().numpy()
        trace_pred = torch.istft(trace*mask,nperseg, hop_length=1, window=torch.hann_window(nperseg)).detach().numpy()
        trace_true = torch.istft(trace*mask_true,nperseg, hop_length=1, window=torch.hann_window(nperseg)).detach().numpy()

        amp = np.max(np.abs(trace_in))

        plt.figure()
        plt.subplot(211)
        plt.plot(t, trace_in)
        plt.ylim([ -amp*1.2, amp*1.2 ])
        plt.grid()
        plt.subplot(212)
        plt.plot(t, trace_true, 'k')
        plt.plot(t, trace_pred, 'r')
        plt.ylim([ -amp*1.2, amp*1.2 ])
        plt.grid()
        plt.savefig('ValidationHydro_sample.png')
        plt.close()

        if model_v == 'v1':

            plt.figure()
            plt.subplot(211)
            plt.pcolormesh(t, f, mask_true, cmap=cmap, vmin=0, vmax=1, shading='auto')
            plt.subplot(212)
            plt.pcolormesh(t, f, mask, cmap=cmap, vmin=0, vmax=1, shading='auto')
            plt.savefig('ValidationHydro_SampleMask.png')
            plt.close()

        scheduler.step(val_loss)
        print(f'--> Validation loss: {val_loss:.5f}; learning rate {scheduler.get_last_lr()[0]}')

        training_hist = np.append(training_hist, running_loss)
        val_hist = np.append(val_hist, val_loss)

        #save current model
        torch.save(model.state_dict(), 'model_test-CNN_UNetTHydro_fac' + str(fac) + model_v + '.pt')

        if scheduler.get_last_lr()[0] < 1e-5:
            print('Learning rate reduce below minimum')
            break

    print('Finished Training')
    np.save('CNN_UnetT_Hydrofac' + str(fac), np.vstack( (training_hist, val_hist) ))
    

""" 
def stft_3comp_data_generator(model, batch_size, data_vec, noise_vec, itpvec, sr, nperseg, noverlap, norm_input, eps=1e-9, nlen=128, valid=False, post=False):
    #data is a data set from the seisbench plateform. Pretty handy.
    #don't need meta data for the noise

    nd, _, _ = data_vec.shape
    nn, _, _ = noise_vec.shape

    sos = signal.butter(2, 2.5, 'hp', fs=sr, output='sos')
    data_to_noise = 35393

    nf = int(np.ceil(nperseg/2))

    #if not valid:

    #    data_inds  = np.random.randint(data_to_noise - 500, size=(nd,)) #this samples the data in OBST2024 dataset
    #    noise_inds = np.random.randint(low=data_to_noise + 500, high=ntraces, size=(nd,)) #this samples the noise in OBST2024 dataset

    #else:

    #    data_inds  = np.random.randint(low=data_to_noise - 500, high=data_to_noise, size=(nd,)) #this samples the data in OBST2024 dataset
    #    noise_inds = np.random.randint(low=data_to_noise, high=data_to_noise + 500, size=(nd,)) #this samples the noise in OBST2024 dataset

    #datavec  = data.get_waveforms(data_inds)[:,0:3,:]
    #itpvec   = np.round(md.trace_p_arrival_sample[data_inds].to_numpy())*(sr/100)#they are valid for 100 Hz but OBST2024 does't scale, you have to do it
    #noisevec = data.get_waveforms(noise_inds)[:,0:3,:]

    while True:
        # grab batch

        #reformat this to get random data from the OBS dataset, and use the p waves picks to get noise as the stuff before it

        #what are the outputs I need to produce
        #batch_inputs - generated by data_2_input on the seperate stft
        #batch_outputs - processed ontop of the inputs
        #subsigs - concatenated data traces
        #subnoises - concatenated noise traces
        #raw_stft - same as batch_inputs just different normalization

        #I've replaced Amanda's fixed 16 with the size of the frequency vector

        batch_inputs = np.zeros((batch_size,nf,nlen,6))
        raw_stft     = np.zeros((batch_size,nf,nlen,6))

        if model=='v1':
            batch_outputs=np.zeros((batch_size,nf,nlen,3))
        elif model=='v2':
            batch_outputs=np.zeros((batch_size,nf,nlen,6))
        elif model=='v3':
            batch_outputs=np.zeros((batch_size,nf,nlen,9))

        subsigs=np.zeros((batch_size,3*nlen))
        subnoises=np.zeros((batch_size,3*nlen))

        for ii in range(batch_size):
            # window data, noise, and data+noise timeseries based on shift

            #The original data format was to have all three components in one vector, which is fucking bizare

            #make subsig1, subsig2, subsig3
            sig_valid = False
            while not sig_valid:

                sig_ind = np.random.randint(nd)

                if not valid:

                    sig_ind  = np.random.randint(0, nd - 500)
                    amp   = np.random.randn()*2.5

                else:

                    sig_ind  = nd - 500 + ii#np.random.randint(low=500, high=nd) #this samples the data in OBST2024 dataset
                    amp = 1

                sigwfs = data_vec[sig_ind, :, :]

                if not np.array([ v.any() for v in sigwfs ]).all():
                    continue

                itp = itpvec[sig_ind]

                if np.isnan(itp):
                    continue

                #trim the data, seisbench pads it to make everything equal length
                #that's fine for picking, terrible for denoising.

                sigwfs = sigwfs[:, np.where(np.sum(sigwfs, axis=0))[0]]
                if not valid:
                    start    = int(np.random.uniform(np.max((0, itp - nlen)), 3000 - nlen - 10))
                else:
                    start = np.min( (int(itp - 128), 3000 - nlen - 10))

                subsig1 = signal.resample(sigwfs[0,:], 3000)
                subsig2 = signal.resample(sigwfs[1,:], 3000)
                subsig3 = signal.resample(sigwfs[2,:], 3000)

                subsig1 = signal.sosfilt(sos, subsig1)
                subsig2 = signal.sosfilt(sos, subsig2)
                subsig3 = signal.sosfilt(sos, subsig3)

                subsig1 = subsig1[start:start+nlen]
                subsig2 = subsig2[start:start+nlen]
                subsig3 = subsig3[start:start+nlen]

                sig_amp = np.max(np.abs(np.concatenate((subsig1,subsig2,subsig3))))
                subsig1 = subsig1*(amp/sig_amp)
                subsig2 = subsig2*(amp/sig_amp)
                subsig3 = subsig3*(amp/sig_amp)

                sig_valid = True

            noise_valid = False
            while not noise_valid:

                if not valid:

                    noise_ind = np.random.randint(nn - 500) #this samples the noise in OBST2024 dataset

                else:

                    noise_ind  = np.random.randint(low=nn - 500+5, high=nn) #this samples the data in OBST2024 dataset
                    #amp = 2*ii/batch_size + 0.5#get's noiser and noiser

                amp = np.random.uniform(-2,2)

                noisewfs = noise_vec[noise_ind, :, :]

                if not np.array([ v.any() for v in noisewfs ]).all():
                    continue

                #trim the data, seisbench pads it to make everything equal length
                #that's fine for picking, terrible for denoising.
                noisewfs = noisewfs[:, np.where(np.sum(noisewfs, axis=0))[0]]

                subnoise1 = signal.resample(noisewfs[0,:], 3000)
                subnoise2 = signal.resample(noisewfs[1,:], 3000)
                subnoise3 = signal.resample(noisewfs[2,:], 3000)

                subnoise1 = signal.sosfilt(sos, subnoise1)
                subnoise2 = signal.sosfilt(sos, subnoise2)
                subnoise3 = signal.sosfilt(sos, subnoise3)

                if not valid:
                    start = int(np.random.uniform(0, 3000 - nlen - 10))#some buffer if the pick sucks
                else:
                    start = 1000

                subnoise1 = subnoise1[start:start+nlen]
                subnoise2 = subnoise2[start:start+nlen]
                subnoise3 = subnoise3[start:start+nlen]

                sig_amp = np.max(np.abs(np.concatenate((subnoise1,subnoise2,subnoise3))))
                subnoise1 = subnoise1*(amp/sig_amp)
                subnoise2 = subnoise2*(amp/sig_amp)
                subnoise3 = subnoise3*(amp/sig_amp)

                noise_valid = True

            subsigs[ii,:]=np.concatenate((subsig1,subsig2,subsig3))
            subnoises[ii,:]=np.concatenate((subnoise1,subnoise2,subnoise3))
            #subsigs[ii,:]=np.concatenate((np.random.rand(subsig1.size),np.random.rand(subsig1.size),np.random.rand(subsig1.size)))
            #subnoises[ii,:]=np.concatenate((np.random.rand(subsig1.size),np.random.rand(subsig1.size),np.random.rand(subsig1.size)))

            _, _, stftsig1 = signal.stft(subsig1, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise1 = signal.stft(subnoise1, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput1 = stftsig1+stftnoise1
            _, _, stftsig2 = signal.stft(subsig2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise2 = signal.stft(subnoise2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput2 = stftsig2+stftnoise2
            _, _, stftsig3 = signal.stft(subsig3, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise3 = signal.stft(subnoise3, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput3 = stftsig3+stftnoise3

            # batch inputs are the real and imaginary parts of the stft of signal+noise
            raw_stft[ii,:,:,:]     = data_2_input(stftinput1,stftinput2,stftinput3,norm_input=False)
            batch_inputs[ii,:,:,:] = data_2_input(stftinput1,stftinput2,stftinput3,norm_input)

            if model=='v1':
                # batch outputs are real valued signal masks
                with np.errstate(divide='ignore'):
                    rat1=np.nan_to_num(np.abs(stftnoise1)/np.abs(stftsig1),posinf=1e20)
                batch_outputs[ii,:,:,0]=1/(1+rat1) # signal mask
                # batch outputs are
                with np.errstate(divide='ignore'):
                    rat2=np.nan_to_num(np.abs(stftnoise2)/np.abs(stftsig2),posinf=1e20)
                batch_outputs[ii,:,:,1]=1/(1+rat2) # signal mask
                # batch outputs are
                with np.errstate(divide='ignore'):
                    rat3=np.nan_to_num(np.abs(stftnoise3)/np.abs(stftsig3),posinf=1e20)
                batch_outputs[ii,:,:,2]=1/(1+rat3) # signal mask
            elif model=='v2':
                # batch outputs are
                batch_outputs[ii,:,:,0]=np.real(stftsig1)
                batch_outputs[ii,:,:,1]=np.imag(stftsig1)
                batch_outputs[ii,:,:,2]=np.real(stftsig2)
                batch_outputs[ii,:,:,3]=np.imag(stftsig2)
                batch_outputs[ii,:,:,4]=np.real(stftsig3)
                batch_outputs[ii,:,:,5]=np.imag(stftsig3)
                if norm_input:
                    batch_outputs[ii,:,:,0]=batch_outputs[ii,:,:,0]/np.max(np.abs(stftinput1))
                    batch_outputs[ii,:,:,1]=batch_outputs[ii,:,:,1]/np.max(np.abs(stftinput1))
                    batch_outputs[ii,:,:,2]=batch_outputs[ii,:,:,2]/np.max(np.abs(stftinput2))
                    batch_outputs[ii,:,:,3]=batch_outputs[ii,:,:,3]/np.max(np.abs(stftinput2))
                    batch_outputs[ii,:,:,4]=batch_outputs[ii,:,:,4]/np.max(np.abs(stftinput3))
                    batch_outputs[ii,:,:,5]=batch_outputs[ii,:,:,5]/np.max(np.abs(stftinput3))
            elif model=='v3':
                # batch outputs are
                batch_outputs[ii,:,:,0]=np.log(np.abs(stftsig1/stftinput1)+eps)
                batch_outputs[ii,:,:,1]=np.cos(np.angle(stftsig1/stftinput1))
                batch_outputs[ii,:,:,2]=np.sin(np.angle(stftsig1/stftinput1))
                batch_outputs[ii,:,:,3]=np.log(np.abs(stftsig2/stftinput2)+eps)
                batch_outputs[ii,:,:,4]=np.cos(np.angle(stftsig2/stftinput2))
                batch_outputs[ii,:,:,5]=np.sin(np.angle(stftsig2/stftinput2))
                batch_outputs[ii,:,:,6]=np.log(np.abs(stftsig3/stftinput3)+eps)
                batch_outputs[ii,:,:,7]=np.cos(np.angle(stftsig3/stftinput3))
                batch_outputs[ii,:,:,8]=np.sin(np.angle(stftsig3/stftinput3))
            # make all angles positive

        if post:
            yield(batch_inputs,batch_outputs,subsigs,subnoises,raw_stft)
        else:
            yield(batch_inputs,batch_outputs)
 
 
            
class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, fac=1, dropout_rate=0.5, activation_fn=nn.ELU(inplace=True)):
        super(UNet, self).__init__()

        # Initialize features based on the scaling factor
        features = 4*fac
        self.encoder1 = UNet._block(in_channels, features, name="enc1", activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.encoder2 = UNet._block(features, features * 2, name="enc2", stride=2, activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", stride=2, activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4", stride=2, activation_fn=activation_fn, dropout_rate=dropout_rate)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck", stride=2, activation_fn=activation_fn, dropout_rate=dropout_rate)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4", activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1", activation_fn=activation_fn, dropout_rate=dropout_rate)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name, stride=1, activation_fn=nn.ReLU(inplace=True), dropout_rate=0.0):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            stride=stride
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", activation_fn),
                    (name + "dropout1", nn.Dropout(p=dropout_rate)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", activation_fn),
                    (name + "dropout2", nn.Dropout(p=dropout_rate)),
                ]
            )
        )

 
# GCN-UNet model
class GCNUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNUNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.unpool1 = TopKPooling(hidden_channels, ratio=2.0)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.unpool2 = TopKPooling(hidden_channels, ratio=2.0)
        self.conv6 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        # Encoder
        x1 = F.relu(self.conv1(x, edge_index))
        x1, edge_index1, _, batch1, _ = self.pool1(x1, edge_index, batch=batch)
        x2 = F.relu(self.conv2(x1, edge_index1))
        x2, edge_index2, _, batch2, _ = self.pool2(x2, edge_index1, batch=batch1)
        
        # Bottleneck
        x3 = F.relu(self.conv3(x2, edge_index2))
        
        # Decoder
        x4 = self.unpool1(x3, edge_index2, batch2)
        x4 = F.relu(self.conv4(x4, edge_index2))
        x5 = self.unpool2(x4, edge_index1, batch1)
        x5 = F.relu(self.conv5(x5, edge_index1))
        
        x_out = self.conv6(x5, edge_index)
        
        return x_out

# GAT-UNet model
class GATUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATUNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.pool1 = TopKPooling(hidden_channels * heads, ratio=0.5)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.pool2 = TopKPooling(hidden_channels * heads, ratio=0.5)
        
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.unpool1 = TopKPooling(hidden_channels * heads, ratio=2.0)
        self.conv5 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.unpool2 = TopKPooling(hidden_channels * heads, ratio=2.0)
        self.conv6 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
    
    def forward(self, x, edge_index, batch):
        # Encoder
        x1 = F.elu(self.conv1(x, edge_index))
        x1, edge_index1, _, batch1, _ = self.pool1(x1, edge_index, batch=batch)
        x2 = F.elu(self.conv2(x1, edge_index1))
        x2, edge_index2, _, batch2, _ = self.pool2(x2, edge_index1, batch=batch1)
        
        # Bottleneck
        x3 = F.elu(self.conv3(x2, edge_index2))
        
        # Decoder
        x4 = self.unpool1(x3, edge_index2, batch2)
        x4 = F.elu(self.conv4(x4, edge_index2))
        x5 = self.unpool2(x4, edge_index1, batch1)
        x5 = F.elu(self.conv5(x5, edge_index1))
        
        x_out = self.conv6(x5, edge_index)
        
        return x_out

def dwt_transform(seismogram, wavelet='db1'):
    coeffs = [pywt.wavedec(channel, wavelet) for channel in seismogram.T]
    return np.concatenate([np.concatenate(c) for c in coeffs])

def create_graph(seismogram):
    node_features = dwt_transform(seismogram)
    num_nodes = len(node_features)
    x = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)  # Unsqueeze to add feature dimension
    
    edge_index = []
    
    # Time-adjacent connections
    for j in range(num_nodes - 1):
        edge_index.append([j, j + 1])
        edge_index.append([j + 1, j])
        
        # Connections between wavelet levels (daughter wavelets)
        if (j + 1) % (num_nodes // 3) == 0 and (j + 1) < num_nodes - 1:
            edge_index.append([j, j + (num_nodes // 3)])
            edge_index.append([j + (num_nodes // 3), j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)



class MyDataset(Dataset):
    def __init__(self, data_vector_1, data_vector_2, itp, nlen, nperseg, style="cart"):
        self.data_vector_1 = torch.tensor(data_vector_1, dtype=torch.float32)
        self.data_vector_2 = torch.tensor(data_vector_2, dtype=torch.float32)
        self.nlen = nlen
        self.itp = itp
        self.nperseg = nperseg
        self.style = style #not currently used
        self.window = torch.hann_window(self.nperseg)

    def __len__(self):
        return len(self.data_vector_1)

    def __getitem__(self, idx):
        vector_1 = self.data_vector_1[idx]
        itp = self.itp[idx]

        start_d = int(np.min( (np.max( (int(itp - self.nlen/2), self.nlen/2)), 3000 - 3*self.nlen/2 - 64)))
        start_d += int(np.random.randint(low=-self.nlen/2, high=self.nlen/2))

        vector_1 = vector_1[:, start_d:start_d+self.nlen]

        idx2 = np.random.randint(0, len(self.data_vector_2)-1)
        vector_2 = self.data_vector_2[idx2]

        if np.random.rand() < 0.2: #zerod signals are common
            amp=0.0
        else:
            amp = np.random.uniform(0,2)*(-1)**np.random.randint(low=1, high=10) # could still be tiny
        
        amp2 = np.random.uniform(0,2)*(-1)**np.random.randint(low=1, high=10) # could still be tiny
        start = int(np.random.uniform(0, 3000 - self.nlen - 10))#some buffer if the pick sucks

        vector_2 = vector_2[:, start:start+self.nlen]

        vector_1 *= amp
        vector_2 *= amp2

        #This is probably the bottleneck
        stftsig1 = torch.stft(vector_1[0,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftnoise1 = torch.stft(vector_2[0,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftinput1 = stftsig1+stftnoise1

        stftsig2 = torch.stft(vector_1[1,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftnoise2 = torch.stft(vector_2[1,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftinput2 = stftsig2+stftnoise2
        stftsig3 = torch.stft(vector_1[2,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftnoise3 = torch.stft(vector_2[2,:], nperseg, hop_length=1, window=self.window, return_complex=True)
        stftinput3 = stftsig3+stftnoise3

        combined = data_2_input(stftinput1,stftinput2,stftinput3,True)

        mask = torch.zeros(3, combined.shape[1], combined.shape[2])

        # batch outputs are real valued signal masks
        rat1=torch.nan_to_num(torch.abs(stftnoise1)/torch.abs(stftsig1),posinf=1e20)
        mask[0, : :]=1/(1+rat1) # signal mask
        # batch outputs are
        rat2=torch.nan_to_num(torch.abs(stftnoise2)/torch.abs(stftsig2),posinf=1e20)
        mask[1, : :]=1/(1+rat2) # signal mask
        # batch outputs are
        rat3=torch.nan_to_num(torch.abs(stftnoise3)/torch.abs(stftsig3),posinf=1e20)
        mask[2, : :]=1/(1+rat3) # signal mask

        sample = {'mask': mask, 'combined': combined}

        return sample

def format_OBST():

    data = sbd.OBST2024() #sampling rate doesn't work
    total, _ = data.metadata.shape

    sos = signal.butter(4, 2.5, 'hp', fs=sr, output='sos')

    data_vec_full  = data.get_waveforms(np.arange(0, 35393))#[:,1:4,:]

    shp = data_vec_full.shape
    #half it and filter
    data_vec = np.zeros((shp[0], shp[1], 3000))
    for k in range(shp[0]):
        subsig1 = signal.resample(data_vec_full[k, 0, :], 3000)
        subsig2 = signal.resample(data_vec_full[k, 1, :], 3000)
        subsig3 = signal.resample(data_vec_full[k, 2, :], 3000)

        data_vec[k,0,:] = signal.sosfilt(sos, subsig1)
        data_vec[k,1,:] = signal.sosfilt(sos, subsig2)
        data_vec[k,2,:] = signal.sosfilt(sos, subsig3)

    data_vec_full = []

    itp_vec   = np.round(data.metadata.trace_p_arrival_sample[:35393].to_numpy()/2)#they are valid for 100 Hz but OBST2024 does't scale when you load it, you have to do it if you want
    noise_vec_full = data.get_waveforms(np.arange(35394, total))#[:,1:4,:]
    shp = noise_vec_full.shape
    #half it and filter
    noise_vec = np.zeros((shp[0], shp[1], 3000))
    good = np.ones((shp[0],), dtype=bool)
    for k in range(shp[0]):
        
        #some are near zero, some are just lines
        subsig1 = signal.detrend(noise_vec_full[k, 0, :])
        subsig2 = signal.detrend(noise_vec_full[k, 1, :])
        subsig3 = signal.detrend(noise_vec_full[k, 2, :])

        #quality control for dead channels
        std1 = np.std(subsig1)
        std2 = np.std(subsig2)
        std3 = np.std(subsig3)

        if np.max([ std1, std2, std3 ])/np.min([std1, std2, std3]) > 1000:
            good[k] = False
            continue

        subsig1 = signal.resample(subsig1, 3000)
        subsig2 = signal.resample(subsig2, 3000)
        subsig3 = signal.resample(subsig3, 3000)

        noise_vec[k,0,:] = signal.sosfilt(sos, subsig1)
        noise_vec[k,1,:] = signal.sosfilt(sos, subsig2)
        noise_vec[k,2,:] = signal.sosfilt(sos, subsig3)

    noise_vec = noise_vec[good, :, :]

    return data_vec, noise_vec, itp_vec


 """