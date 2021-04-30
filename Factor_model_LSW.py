import numpy as np
import matplotlib.pyplot as plt
import LSW_model as lsw 
import Evolutionary_Wavelet_Spectrum as ews
import WaveletDecomposition as dec
import custom_wavelets as wav
import Smoother as smo
import wavelet_utils as utils
import pywt
from scipy.sparse.linalg import eigs

class LSW_FactorModel:
    maxScale:int
    crossEWS:ews.CrossEWS
    loadings:np.ndarray
    eigValue:np.ndarray
    factors:np.ndarray
    commonComp:np.ndarray
    def __init__(self, signals:np.ndarray, wavelet_name:str, order:int, n_factors:int, maxScale:int=0):
        self.signals = signals      # (n_signals, length)
        self.n_signals = self.signals.shape[0]
        self.length_signal = self.signals.shape[1]
        self.wavelet_name = wavelet_name
        self.order = order
        self.n_factors = n_factors
        
        if maxScale == 0:
            self.__computeMaxScale()
        else:
            self.maxScale = maxScale
            
        self.wavelet = wav.Wavelet(self.wavelet_name, self.maxScale)
        self.wavelet.discretizeToMaxScale()
        
        self.__initializeModel()
        
    def __initializeModel(self):
        decomp = []
        for i in range(self.n_signals):
            decomp.append(dec.WaveletDecomposition(self.signals[i], self.wavelet).details)
        decomp = np.array(decomp)
        self.crossEWS = ews.CrossEWS(decomp, isSpectrum=False, order=self.order, wavelet=self.wavelet)
        self.crossEWS.correctSpectrum()
        
    def smoothSpectrum(self, smoother:smo.Smoother):
        self.crossEWS.smoothSpectrum(smoother)
        
    def getLoadings(self):
        self.__initializeLoadingsIfNot()
        self.__initializeFactorsIfNot()
        
        for z in range(self.length_signal):
            S = self.createBlockSatTimeZ(z)
            # _, eigVectors = eigs(S, k=self.n_factors, which='LM')
            # self.eigValue[:, z], _ = np.real(eigs(S, k=self.maxScale*self.n_signals-2, which='LM'))
            self.eigValue[:, z], eigVectors = self.getKLargestRealEigenvectors(S)
            loadings = np.sqrt(self.n_signals*self.maxScale) * eigVectors
            self.loadings[:, :, z] = loadings
            self.factors[:, z] = self.getFactors(z, loadings)
            
    def getKLargestRealEigenvectors(self, matrix:np.ndarray):
        eigvalues, eigvectors = np.linalg.eig(matrix)
        idx_sort = np.flip(np.argsort(np.abs(np.real(eigvalues))))
        p = self.getPermuMatrix(idx_sort)
        eigvalues = eigvalues @ p
        eigvectors = eigvectors @ p
        return np.real(eigvalues), np.real(eigvectors[:, :self.n_factors])
        
    def getPermuMatrix(self, ordered_idx:np.ndarray):
        p = np.zeros((len(ordered_idx), len(ordered_idx)), dtype=np.float64)
        for k in range(len(ordered_idx)):
            p[ordered_idx[k], k] = 1.0
        return p
        
    def createBlockSatTimeZ(self, z:int):       
        block = [[self.getCorrectMatrix(scale1, scale2, z) for scale2 in range(self.maxScale)] for scale1 in range(self.maxScale)]
        S = np.block(block)
        return S
                
    def getCorrectMatrix(self, scale1:int, scale2:int, z:int):
        order = scale2 - scale1  
        if np.abs(order) > self.order:
            return np.zeros((self.n_signals, self.n_signals), dtype=np.float64)
        elif order >= 0:
            return self.crossEWS.getSpectrumForAllSignalsAtTimeZ(scale1, order, z)
        else:
            return self.crossEWS.getSpectrumForAllSignalsAtTimeZ(scale2, -order, z).T
                
    def getFactors(self, z:int, loadings:np.ndarray):
        # Order 'F' since the inner arrangement of loadings is by time serie, 
        # the outer arrangement is by order/scale.
        decomp = np.ravel(self.crossEWS.decomposition[:, :, z], order='F') 
        return (loadings.T @ decomp)/(self.n_signals*self.maxScale)
    
    def getCommonCompInWaveletDomForSignalAndScale(self, u:int, i:int):
        idx = self.n_signals*i + u
        return (self.loadings[idx] * self.factors).sum(axis=0)
    
    def getCommonCompInTimeDomForSignal(self, u:int):
        out = np.zeros((self.length_signal,), dtype=np.float64)
        for j in range(self.maxScale):
            discr_wav_j = self.wavelet.discritization[j]
            # print(self.getCommonCompInWaveletDomForSignalAndScale(u, j).shape)
            comp_wav_j = self.getCommonCompInWaveletDomForSignalAndScale(u, j)
            out += utils.fft_convolve(comp_wav_j, discr_wav_j)
        return out
    
    def getCommonComponents(self):
        self.__initializeCommonCompIfNot()
        for u in range(self.n_signals):
            self.commonComp[u] = self.getCommonCompInTimeDomForSignal(u)
            
    def __computeMaxScale(self):
        filterLength = len(np.array(pywt.Wavelet(self.wavelet_name).dec_lo))
        self.maxScale = int(np.floor(np.log(self.length_signal)/np.log(2)))
        self.__getMaxScaleBelowTwoTimeSignalLength(filterLength)
        
    def __getMaxScaleBelowTwoTimeSignalLength(self, filterLength:int):
        while (2**self.maxScale - 1)*(filterLength-1)+1 > 2*self.length_signal:
            self.maxScale -= 1
            
    def __initializeLoadingsIfNot(self):
        if not utils.isArrayInitialized(self, 'loadings'):
            # loadings are sorted by scale then by element (outer : scale, inner : elements)
            self.loadings = np.empty((self.maxScale*self.n_signals, self.n_factors, self.length_signal), dtype=np.float64)
            self.eigValue = np.empty((self.maxScale*self.n_signals, self.length_signal), dtype=np.float64)
    
    def __initializeFactorsIfNot(self):
        if not utils.isArrayInitialized(self, 'factors'):
            self.factors = np.empty((self.n_factors, self.length_signal), dtype=np.float64)
            
    def __initializeCommonCompIfNot(self):
        if not utils.isArrayInitialized(self, 'commonComp'):
            self.commonComp = np.zeros((self.n_signals, self.length_signal), dtype=np.float64)






