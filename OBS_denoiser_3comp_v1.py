#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:29:11 2021

MODEL 1: Train a CNN to denoise 3 component GNSS data by predicting a real-valued mask

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import obs_tools
import gnss_tools
import h5py
import datetime
import pickle
import seisbench.data as sbd

tf.config.threading.set_intra_op_parallelism_threads(8)

# SET OPTIONS
train=True # # do you want to train?
plots=True # # do you want to make some plots?
epos=200 # how many epocs?
fac=4 # model size
sr=50 # sample rate
eps=1e-9 # epsilon
drop=0.2 # model drop rate
nlen=1024# window length
nperseg=31 # param for STFT
noverlap=30 # param for STFT
norm_input=False # leave this alone
np.random.seed(3876)
tf.random.set_seed(1)

batch_size    = 256
test_set_size = 100 #not set correctly to seperate training and validation set

# LOAD THE DATA
#print("LOADING DATA")
data = sbd.OBST2024(sampling_rate=sr, dimension_order="NCW")
md = data.metadata
nd, _ = md.shape

#35393 is the index break from eqs to noise in OBST2024
#use smaller numbers for fast tests

data_vec  = data.get_waveforms(np.arange(0, 35393))[:,0:3,:]
itpvec   = np.round(md.trace_p_arrival_sample[:35393].to_numpy())*(sr/100)#they are valid for 100 Hz but OBST2024 does't scale when you load it, you have to do it
noise_vec = data.get_waveforms(np.arange(nd - 35393, nd))[:,0:3,:]

nd = 25000

# SET MODEL FILE NAME
if train:
    if not(norm_input):
        model_save_file="qOBS_3comp_v1_"+str(fac)+"_"+str(datetime.datetime.today().month)+"-"+str(datetime.datetime.today().day)+".weights.h5"
    else:
        model_save_file="qOBS_3comp_norm_input_v1_"+str(fac)+"_"+str(datetime.datetime.today().month)+"-"+str(datetime.datetime.today().day)+".weights.h5"
else:
    model_save_file="qOBS_3comp_v1_2_5-16.weights.h5"

# DATA GENERATOR
print("FIRST PASS WITH DATA GENERATOR")
my_data=obs_tools.stft_3comp_data_generator('v1',32,data_vec, noise_vec, itpvec,sr,nperseg,noverlap,norm_input,eps,nlen)
#my_data=gnss_tools.stft_3comp_data_generator('v1',32,x_data[x_train_inds,:],n_data[n_train_inds,:],sr,nperseg,noverlap,norm_input)
x,y=next(my_data)

my_test_data=obs_tools.stft_3comp_data_generator('v1',50,data_vec, noise_vec, itpvec,sr,nperseg,noverlap,norm_input,eps,nlen,valid=True, post=True)
#my_test_data=gnss_tools.stft_3comp_data_generator('v1',32,x_data[x_train_inds,:],n_data[n_train_inds,:],sr,nperseg,noverlap,norm_input, valid=True)
x,y,sigs,noise,x1=next(my_test_data)

# BUILD THE MODEL
print("BUILD THE MODEL")
model=obs_tools.make_small_unet_v1(int(np.ceil(nperseg/2)), nlen, drop=drop, ncomp=3,fac=fac)

# ADD SOME CHECKPOINTS
print("ADDING CHECKPOINTS")
checkpoint_filepath = './checks/'+model_save_file+'_{epoch:04d}.weights.h5'
#checkpoint_filepath = './checks/'+model_save_file+'_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, verbose=1,
    monitor='val_acc', mode='max', save_best_only=True)

# TRAIN THE MODEL
print("TRAINING!!!")
if train:
    # if resume:
    #     print('Resuming training results from '+model_save_file)
    #     model.load_weights(checkpoint_filepath)
    # else:
    print('Training model and saving results to '+model_save_file)

    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file+".csv", append=True)

    history=model.fit(x = obs_tools.stft_3comp_data_generator('v1',batch_size,data_vec, noise_vec, itpvec,sr,nperseg,noverlap,norm_input,eps,nlen),
                        steps_per_epoch=nd//batch_size,
                        validation_data=obs_tools.stft_3comp_data_generator('v1',batch_size,data_vec, noise_vec, itpvec,sr,nperseg,noverlap,norm_input,eps,nlen, valid=True),
                        validation_steps=100//batch_size,
                        epochs=epos, callbacks=[model_checkpoint_callback,csv_logger])

    model.save_weights("./"+model_save_file)
else:
    print('Loading training results from '+model_save_file)
    model.load_weights("./"+model_save_file)

# PLOT TRAINING STATS
print("PLOTTING TRAINING STATS")
obs_tools.plot_training_curves(model_save_file)

# MAKE SOME PREDICTIONS
print("PREDICTING")
#maxrange=x_test_inds.shape[0]

my_test_data=obs_tools.stft_3comp_data_generator('v1',test_set_size,data_vec, noise_vec, itpvec,sr,nperseg,noverlap,norm_input,eps,nlen, valid=True, post=True)
x,y,sigs,noise,x1=next(my_test_data)
test_predictions=model.predict(x)

# GET DENOISED SIGNALS FROM OUTPUT
print("GETTING DENOISED SIGNALS FROM MODEL OUTPUT")
tru_sig_inv, tru_noise_inv, est_sig_inv, est_noise_inv=obs_tools.output_2_data('v1',y,x1,data_vec, noise_vec, itpvec,sr,nperseg,noverlap,norm_input)

# SAVE SUBSET FOR PLOTTING PURPOSES
with open('model1_v'+str(fac)+'_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x,y,sigs,noise,x1,test_predictions], f)
