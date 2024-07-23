#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:51:40 2022

Make plots of denoising examples

@author: amt
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import gnss_tools

sr=50
nperseg=31
noverlap=30
nlen=512
model='model1'
fac=4
np.random.seed(0)

if model=='model1':
    # Getting back the objects:
    with open('model1_v'+str(fac)+'_results.pkl','rb') as f:  # Python 3: open(..., 'rb')
        x, y, sigs, noise, x1, test_predictions = pickle.load(f)
#plt.figure()
#plt.plot(sigs[0,:])

f, t, Zxx = signal.stft(sigs[0,:256], fs=sr, nperseg=nperseg, noverlap=noverlap)

maxrange=x.shape[0]
lowamp=False
if lowamp:
    minval=0
    maxval=1e9
else:
    minval=0
    maxval=1e9

for count, ind in enumerate([ 5, 6, 94]): # LSNR - 2163, 1187 HSNR - 46
    #print(str(np.max(np.abs(sigs[ind,:]))))
    print(ind)
    if (np.max(np.abs(sigs[ind,:])) < maxval) and (np.max(np.abs(sigs[ind,:])) > minval):
        if lowamp:
            lim=mlim=0.04
            SNRmax=0
        else:
            lim=mlim=0.4
        t=np.arange(x[0,:,:,0].shape[1])*sr

        comp = 0

        _,tru_sig_inv=signal.istft(y[ind,:,:,comp]*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
        _,tru_noise_inv=signal.istft((1-y[ind,:,:,comp])*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
        _,est_sig_inv=signal.istft(test_predictions[ind,:,:,comp]*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
        _,est_noise_inv=signal.istft((1-test_predictions[ind,:,:,comp])*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)

        true_noise=noise[ind,comp*nlen:(comp+1)*nlen]
        true_signal=sigs[ind,comp*nlen:(comp+1)*nlen]
        #inds=np.where(np.abs(true_signal)>0.00001)[0]
        #VR=gnss_tools.comp_VR(true_signal,est_sig_inv)
        # print(true_signal[:5])
        # print(true_noise[:5])
        # print(est_sig_inv[:5])

        amp1 = np.max(np.abs(sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen]))
        amp2 = np.max(np.abs(true_signal))
        amp3 = np.max(np.abs(est_sig_inv))

        plt.plot(t, true_signal/amp2 + 4, color=(67/256,0,152/256), label='signal')
        plt.plot(t,(sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen])/amp1 + 2, color=(0.33,0.33,0.33), label='signal+noise')
        # axs[4,comp].plot(t, tru_sig_inv, alpha=0.75, color=(0.6,0,0), label='reconstructed')
        plt.plot(t, est_sig_inv/amp3, color=(152/256,0,67/256), label='denoised signal')

        plt.savefig("nehrp"+str(ind)+".pdf")
        plt.clf()
