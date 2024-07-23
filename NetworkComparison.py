#####script to compare the two different networks
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from torch.utils.data import Dataset, DataLoader, random_split
import seisbench.data as sbd
import copy
import obspy
import requests
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from scipy import signal
import os

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

        #self.transformer = CustomTransformer(128 * fac, num_layers=3, num_heads=2, hidden_dim=128 * fac, dropout=drop)

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

        if ncomp == 1:
            self.final = nn.Conv2d(8 * fac, 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(8 * fac, 3, kernel_size=1)

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
        #v = enc5b.squeeze()

        #correction for nbatch = 1
        #if len(v.shape)==2:
        #    v = v.unsqueeze(0)

        #transformed = self.transformer(torch.permute(v, [0, 2, 1 ]))
        #transformed = torch.permute(transformed, [0, 2, 1]).unsqueeze(2)

        #dec4 = F.elu(self.bn4u(self.upconv4(transformed)))
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

def data_2_input(stftinput1,stftinput2,stftinput3,norm_input):
    
    if norm_input:
        stftinput1=stftinput1/torch.max(torch.abs(stftinput1) + 0.000001)#zero protect
        stftinput2=stftinput2/torch.max(torch.abs(stftinput2) + 0.000001)#all zeros is not a bug
        stftinput3=stftinput3/torch.max(torch.abs(stftinput3) + 0.000001)

    tmp = torch.zeros((6, stftinput1.shape[0],stftinput1.shape[1]))

    tmp[0, :, :] = torch.real(stftinput1)
    tmp[1, :, :] = torch.imag(stftinput1)
    tmp[2, :, :] = torch.real(stftinput2)
    tmp[3, :, :] = torch.imag(stftinput2)
    tmp[4, :, :] = torch.real(stftinput3)
    tmp[5, :, :] = torch.imag(stftinput3)

    return tmp

def format_obspy_data(st, hp):

    w = torch.hann_window(31)

    st.resample(50.0)
    st.taper(0.2)
    st.filter("highpass", freq=hp)

    amp = np.max(np.abs(np.hstack( (st[0].data, st[1].data, st[2].data) )))

    c1 = st[0].stats['channel']
    c2 = st[1].stats['channel']
    c3 = st[2].stats['channel']

    if c1[-1] == 'Z':
        stftsig1 = torch.stft(torch.tensor(st[0].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)
        stftsig2 = torch.stft(torch.tensor(st[1].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)
        stftsig3 = torch.stft(torch.tensor(st[2].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)

        order = [0,1,2]

    if c2[-1] == 'Z':
        stftsig1 = torch.stft(torch.tensor(st[1].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)
        stftsig2 = torch.stft(torch.tensor(st[0].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)
        stftsig3 = torch.stft(torch.tensor(st[2].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)

        order = [1,0,2]

    if c3[-1] == 'Z':
        stftsig1 = torch.stft(torch.tensor(st[2].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)
        stftsig2 = torch.stft(torch.tensor(st[0].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)
        stftsig3 = torch.stft(torch.tensor(st[1].data/amp, dtype=torch.float32), 31, hop_length=1, window=w, return_complex=True)

        order = [2,0,1]

    return data_2_input(stftsig1,stftsig2,stftsig3,True).unsqueeze(0), order, amp

def denoise_3C(st1, model,nperseg):

    x, order, amp = format_obspy_data(st1, 2.5)

    mask = model(x)

    #mask = torch.ones_like(mask)

    c1 = torch.istft((x[0,0,:,:] + x[0,1,:,:]*1j)*mask[0, 0, :, :],nperseg, hop_length=1, window=torch.hann_window(nperseg)).detach().numpy()
    c2 = torch.istft((x[0,2,:,:] + x[0,3,:,:]*1j)*mask[0, 1, :, :],nperseg, hop_length=1, window=torch.hann_window(nperseg)).detach().numpy()
    c3 = torch.istft((x[0,4,:,:] + x[0,5,:,:]*1j)*mask[0, 2, :, :],nperseg, hop_length=1, window=torch.hann_window(nperseg)).detach().numpy()
    data = np.zeros((3, c1.size))

    data[0, :] = c1
    data[1, :] = c2
    data[2, :] = c3

    return data, mask, order

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

        combined = data_2_input(stftinput1,stftinput2,stftinput3,True)

        mask = torch.zeros(1, combined.shape[1], combined.shape[2])

        # batch outputs are real valued signal masks
        rat1=torch.nan_to_num(torch.abs(stftnoise1)/torch.abs(stftsig1),posinf=1e20)
        mask[0, : :]=1/(1+rat1) # signal mask
        # batch outputs are
        #rat2=torch.nan_to_num(torch.abs(stftnoise2)/torch.abs(stftsig2),posinf=1e20)
        #mask[1, : :]=1/(1+rat2) # signal mask
        # batch outputs are
        #rat3=torch.nan_to_num(torch.abs(stftnoise3)/torch.abs(stftsig3),posinf=1e20)
        #mask[2, : :]=1/(1+rat3) # signal mask

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

    chunks = 1000

    ind = 0

    data = sbd.OBS(dimension_order="NCW", component_order="Z12H") #sampling rate doesn't work
    total, _ = data.metadata.shape

    sos = signal.butter(4, 2.5, 'hp', fs=sr, output='sos')

    data_vec = np.array(())

    #while ind < total:
    while ind < 10000:

        if ind + chunks < total:
            wf_vec_chunk  = data.get_waveforms(np.arange(ind, ind + chunks), sampling_rate=50)[:,0:3,:]
        else:
            wf_vec_chunk  = data.get_waveforms(np.arange(ind, total), sampling_rate=50)[:,0:3,:]
        shp = wf_vec_chunk.shape

        data_chunk = np.zeros((shp[0], 3, nlen*2))
        noise_chunk = np.zeros((shp[0], 3, nlen*2))

        #filter
        for k in range(shp[0]):

            breakpoint()

            wf_vec_chunk[k,0,:] = signal.sosfilt(sos, wf_vec_chunk[k, 0, :])
            wf_vec_chunk[k,1,:] = signal.sosfilt(sos, wf_vec_chunk[k, 1, :])
            wf_vec_chunk[k,2,:] = signal.sosfilt(sos, wf_vec_chunk[k, 2, :])

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

                amp = np.max(np.abs(np.hstack( (noise_chunk[k, 0, :], noise_chunk[k, 1, :], noise_chunk[k, 2, :]) ))) + 0.001
                noise_chunk[k, :, :] = noise_chunk[k, :, :]/amp

            if itp > 2*nlen: #currently just skips if too close

                snr = getsnr(wf_vec_chunk[k, 0, (itp - nlen):(itp + nlen)])

                if snr > min_snr:
                    data_chunk[k, 0, :] = wf_vec_chunk[k, 0, (itp - nlen):(itp + nlen)]
                    data_chunk[k, 1, :] = wf_vec_chunk[k, 1, (itp - nlen):(itp + nlen)]
                    data_chunk[k, 2, :] = wf_vec_chunk[k, 2, (itp - nlen):(itp + nlen)]

                    amp = np.max(np.abs(np.hstack( (data_chunk[k, 0, :], data_chunk[k, 1, :], data_chunk[k, 2, :]) ))) + 0.001
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

np.random.seed(3)
torch.manual_seed(3)

sr = 50
cmap='PuRd'
nlen = 256
nperseg = 31
name = 'CNN_UNet_fac1v1'

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

dataloader_train = DataLoader(train_dataset, batch_size = 100, shuffle=True)
dataloader_valid = DataLoader(val_dataset, batch_size = 100, shuffle=False) 

for batch in dataloader_valid: #for plotting
    valbatch = batch
    break

f, t, _ = signal.stft(np.zeros(nlen),fs=sr,nperseg=31, noverlap=30)

########now load the model
model = UNetv1(drop=0, ncomp=3, fac=1) #pytorch version of amanda's tensorflow UNet
model.load_state_dict(torch.load('./model_test-' + name + '.pt'))
model = model.float()
model.eval()

os.makedirs('./TEST' + name)

for k in np.arange(0, 50):

    mask = model(valbatch['combined'][k,:,:,:].unsqueeze(0)).detach().numpy()[0,0,:,:]
    mask_true = valbatch['mask'][k,0,:,:]
    trace = valbatch['combined'][k,0,:,:] + valbatch['combined'][k,1,:,:]*1j

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
    plt.savefig('./TEST' + name + '/Validation_sample' + name + str(k) + '.png')
    plt.close()

    plt.figure()
    plt.subplot(211)
    plt.pcolormesh(t, f, mask_true, cmap=cmap, vmin=0, vmax=1, shading='auto')
    plt.subplot(212)
    plt.pcolormesh(t, f, mask, cmap=cmap, vmin=0, vmax=1, shading='auto')
    plt.savefig('./TEST' + name + '/Validation_SampleMask' + name + str(k) + '.png')
    plt.close()

#client = Client("IRIS")
#t = UTCDateTime("2012-03-20,08:43:30.00")
#st1 = client.get_waveforms("7D", "J28A", "*", "HH*,BH*", t, t + 120)
#out_traces, mask, order = denoise_3C(st1, model, 31)

######now make a nice plot
#plt.figure()
#plt.subplot(311)
#plt.plot(st1[0].times("relative"), st1[order[0]].data/np.max(st1[order[0]].data), 'k')
#plt.plot(st1[0].times("relative"), out_traces[0, :]/np.max(out_traces[0, :]), 'r')
#plt.title('Vertical')

#plt.subplot(312)
#plt.plot(st1[0].times("relative"), st1[order[1]].data/np.max(st1[order[1]].data), 'k')
#plt.plot(st1[0].times("relative"), out_traces[1, :]/np.max(out_traces[1, :]), 'r')
#plt.title('Horizontal 1')

#plt.subplot(313)
#plt.plot(st1[0].times("relative"), st1[order[2]].data/np.max(st1[order[2]].data), 'k')
#plt.plot(st1[0].times("relative"), out_traces[2, :]/np.max(out_traces[2, :]), 'r')
#plt.title('Horizontal 2')

#plt.show()