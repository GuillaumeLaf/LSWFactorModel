import numpy as np
import matplotlib.pyplot as plt
import LSW_model as lsw 
import Evolutionary_Wavelet_Spectrum as ews
import WaveletDecomposition as dec
import custom_wavelets as wav
import Smoother as smo
import Factor_model_LSW as fmodel
from tqdm import tqdm
import cProfile as profile
from timeit import timeit

# l = 1024
# mS = int(np.floor(np.log(l)/np.log(2)))
# spect = np.zeros((mS, l), dtype=np.float64)
# spect[0][128:512] = 1.0
# spect[0][768:] = (np.sin(2*np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[768:]
# spect[2][:400] = (np.sin(np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[:400]
# spect[3][284:] = (np.sin(5*np.pi*np.linspace(0,1,l) - np.pi/4)**2 + 0.5)[284:]
# # spect[3][:589] = (np.sin(5*np.pi*np.linspace(0,1,l) - np.pi/3)**2 + 0.5)[:589]

# n = 1000
# n_signals = 2
# order = 0

# lsw1 = lsw.LSW(spect, 'db1', order=order)
# incrCorrRows = lsw1.getConstantIncrementsCorrelationRows(np.array([1.]))
# lsw1.addIncrementsCorrelationMatrix(incrCorrRows)

# # smoother = smo.Kernel_smoother('Gaussian', 75)
# smoother = smo.SWT_smoother(wav.Wavelet('db10', 6), 'soft')

# for i in tqdm(range(n)):
#     multi_decomp = []
#     for j in range(n_signals):
#         lsw1.simulate()
#         decomp = dec.WaveletDecomposition(lsw1.simulation, lsw1.wavelet)
#         multi_decomp.append(decomp.details)
#     multi_decomp = np.array(multi_decomp)
    
#     if i == 0:
#         spect = ews.CrossEWS(multi_decomp, isSpectrum=False, order=order, wavelet=lsw1.wavelet)
#         avg_simu = np.zeros_like(spect.spectrum, dtype=np.float64)
        
#     spect.updateDecomposition(multi_decomp)
#     spect.correctSpectrum()
#     avg_simu += (spect.spectrum/n)
    
# avg_ews = ews.CrossEWS(multi_decomp, isSpectrum=False, order=order, wavelet=lsw1.wavelet)
# avg_ews.spectrum = avg_simu
# avg_ews.graph(u=0,v=0,order=0,sharey=True)

# signals = []      
# for i in range(n_signals):
#     lsw1.simulate()
#     lsw1.simulation += 0.025
#     lsw1.simulation = np.cumsum(lsw1.simulation)
#     signals.append(lsw1.simulation)
# signals = np.array(signals)

# fm = fmodel.LSW_FactorModel(signals, lsw1.wavelet_name, order, 2)
# fm.smoothSpectrum(smoother) 
# fm.getLoadings()







import pandas as pd

data = pd.read_csv('euribor_daily.csv', delimiter=';', index_col=[0], parse_dates=True)
data.drop('date', inplace=True, axis=1)
data.index = pd.to_datetime(data.index, unit='D', origin='1899-12-30')

data = data - data.shift(1)
data.dropna(inplace=True)

# data = data.loc[data.index.dropna()]
data = data.iloc[::-1]
# data = data.reset_index()
# data.drop('index', inplace=True, axis=1)
# data.to_csv('euribor_daily.csv')
# # data = data + 1
# data = np.log1p(data.pct_change())
# data = data.pct_change()
# data = np.cumsum(data)


# smoother = smo.Kernel_smoother('Gaussian', 1000)
# # smoother = smo.SWT_smoother(wav.Wavelet('db10', 6), 'soft')

# # np_data = np.flip(data.T.to_numpy(), axis=1)

# fm = fmodel.LSW_FactorModel(data.T.to_numpy(), 'db1', order=0, n_factors=1)
# fm.smoothSpectrum(smoother)
# fm.getLoadings()
# fm.getCommonComp()


        
        
        







        





