#figure out how to use deep denoiser
#This is mostly just coping shit from Weiqiang's website so I can figure this sort of thing out

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import obspy
import requests
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, plot_trigger

client = Client("IRIS")

f = open("./HawaiianRidge/MGL1806.csv", 'w')

f.write('fname\n')
t = UTCDateTime("2018-09-20,17:28:50")

st = client.get_waveforms("ZU", "135", "*", "ELZ", t, t + 60)

for tr in st:
    tr.detrend("demean")
    tr.resample(100.0)
    tr.taper(0.2)
    tr.filter("highpass", freq=2.5)

    if tr.data.size % 2 == 1:
        tr.trim(t, t+359.99, pad=True)

    tr.write('./test_data/Greenland/' + tr.stats.station + '.mseed', format='MSEED')

    f.write(tr.stats.station + '.mseed\n')

f.close()
