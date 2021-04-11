import numpy as np
import matplotlib.pyplot as plt
import LSW_model as lsw 
import Evolutionary_Wavelet_Spectrum as ews
import WaveletDecomposition as dec
import custom_wavelets as wav
import Smoother as smo
from tqdm import tqdm
import cProfile as profile
from timeit import timeit

l = 1024
mS = int(np.floor(np.log(l)/np.log(2)))
spect = np.zeros((mS, l), dtype=np.float64)
spect[0][256:589] = 1.0
spect[0][768:] = (np.sin(2*np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[768:]
spect[2][:256] = (np.sin(np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[:256]
spect[3][284:] = (np.sin(5*np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[284:]
# spect[3][:589] = (np.sin(5*np.pi*np.linspace(0,1,l) - np.pi/3)**2 + 0.5)[:589]
# spect[1][589:768] = (np.sin(5*np.pi*np.linspace(0,1,1024) - np.pi/4)**2 + 0.5)[589:768]

# spect[0][128:512] = 1.0
# # spect[0][768:] = (np.sin(2*np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[768:]
# spect[2][:400] = (np.sin(np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[:400]
# spect[1][284:] = (np.sin(5*np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[284:]

n = 500
order = 2

lsw1 = lsw.LSW(spect, 'db1', order=order)
incrCorrRows = lsw1.getConstantIncrementsCorrelationRows(np.array([1., 0.7]))
lsw1.addIncrementsCorrelationMatrix(incrCorrRows)
avg_simulation = np.zeros_like(lsw1.evol_spectrum.spectrum)
# smoother = smo.Kernel_smoother('Gaussian', 75)
smoother = smo.SWT_smoother(wav.Wavelet('db10', 6), 'soft')
for i in tqdm(range(n)):
    
    lsw1.simulate()
    # lsw1.simulation = lsw1.simulation + 0.025
    
    decomp_simulation = dec.WaveletDecomposition(lsw1.simulation, lsw1.wavelet)
    if i == 0:
        spect_simulation = ews.EWS(decomp_simulation.details, isSpectrum=False, order=order, wavelet=lsw1.wavelet)
    spect_simulation.updateDecomposition(decomp_simulation.details)
    
    # spect_simulation.smoothSpectrum(smoother)
    spect_simulation.correctSpectrum()
    
    avg_simulation += (spect_simulation.spectrum/n)

avg_ews = ews.EWS(decomp_simulation.details, isSpectrum=False, order=order, wavelet=lsw1.wavelet)
avg_ews.spectrum = avg_simulation
avg_ews.graph(sharey=True)

# import pandas as pd
# import matplotlib.dates as dates
# import datetime

# data = pd.read_csv('euribor_data.csv', delimiter=';', index_col=[0], parse_dates=True)
# data = data.loc[data.index.dropna()]
# # data.dropna(inplace=True, axis=0)
# # data.fillna(method='bfill', inplace=True)
# data = data.iloc[:, :-1]
# data = data + 1
# data = np.log1p(data.pct_change())
# data.plot(figsize=(12, 8), title='Euribor Rates in return')
# data.dropna(inplace=True)


# data_np = np.flip(data.T.to_numpy(), axis=1)

# names = [1, 3, 6, 9, 12, 2]
# # smoother = smo.SWT_smoother(wav.Wavelet('db10', 6), 'soft')
# smoother = smo.Kernel_smoother('Gaussian', 20)
# for i in tqdm(range(data_np.shape[0])):
#     fig, ax = plt.subplots(1,1,figsize=(12, 8))
#     fig.suptitle('Euribor ' + str(names[i]) + ' months rate in return', fontsize=20)
#     ax.plot_date(data.index, np.flip(data_np[i]), '-')
#     # ax.plot(data_np[i])
#     lsw1 = lsw.LSW(data_np[i], 'db1', order=0)
#     incrCorrRows = lsw1.getConstantIncrementsCorrelationRows(np.array([1.]))
#     lsw1.addIncrementsCorrelationMatrix(incrCorrRows)

#     lsw1.evol_spectrum.correctSpectrum()
#     lsw1.evol_spectrum.smoothSpectrum(smoother)
    
#     # lsw1.evol_spectrum.graph(sharey=True, title='Euribor ' + names[i] + ' month rate in level')
#     fig2, ax2 = plt.subplots(lsw1.evol_spectrum.spectrum.shape[0], figsize=(12, 15), sharex=True, sharey=True)
#     fig2.suptitle('EWS of Euribor ' + str(names[i]) + ' months rate in return \n smoothed using Gaussian kernel of window 20', fontsize=20)
#     ax2 = np.ravel(ax2)
#     for j in range(len(ax2)):
#         ax2[j].plot_date(data.index, np.flip(lsw1.evol_spectrum.spectrum[j, 0]), '-')
#         # ax2[j].plot(lsw1.evol_spectrum.spectrum[j, 0])
#         ax2[j].set_ylabel(f'Scale -{j+1}')
#         ax2[j].axhline(y=0.0, c='black')
    
# # import pywt
# # pyd = np.array(pywt.swt(data_np[0][:1024], 'db1', trim_approx=True))
# fig, ax = plt.subplots(pyd.shape[0], 1, figsize=(12, 15))
# ax = np.ravel(ax)
# for i in range(pyd.shape[0]):
#     ax[i].plot(pyd[i])
    
    
    
    
    
    
    
    
    

