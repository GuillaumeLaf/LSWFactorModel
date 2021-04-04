import numpy as np
import pywt
import matplotlib.pyplot as plt
import wavelet_utils as utils
import custom_wavelets as w
import WaveletDecomposition as dec
import Evolutionary_Wavelet_Spectrum as ews
import numba as nb

class LSW:
    simulation:np.ndarray
    isSignalASpectrum:bool
    maxScale:int
    decomposition:dec.WaveletDecomposition
    evol_spectrum:ews.EWS
    incrementsCorrelationMatrix:np.ndarray
    lengthSignal:int
    
    def __init__(self, signal:np.ndarray, wavelet_name:str, order:int=0):
        self.signal = signal
        self.order = order
        self.wavelet_name = wavelet_name
        
        self.__isSignalASpectrum()
        self.__computeMaxScale()
        
        self.wavelet = w.Wavelet(wavelet_name, self.maxScale)
        
        self.__initializeDecomposition()
    
    def getConstantIncrementsCorrelationRows(self, firstRow:np.ndarray):
        firstRow = firstRow[np.newaxis, :]
        return np.repeat(firstRow, self.lengthSignal, axis=0)
    
    def addIncrementsCorrelationMatrix(self, RowsForEachZ:np.ndarray):
        self.__initializeIncrementCorrelationMatrixIfNot()
        self.__checkSignalLengthMatch(RowsForEachZ.shape[0])
        for z in range(self.lengthSignal):
            buildingMatrix = incrCorrMatrixAtTimeZ(RowsForEachZ[z], self.maxScale)
            buildingMatrix = buildingMatrix @ buildingMatrix.T
            
            self.incrementsCorrelationMatrix[:, :, z] = getCorrelationMatrixFromCovMatrix(buildingMatrix)
        
        self.__initializeSpectrum()
        
    def getRandomizedCoeffs(self):
        # originalSpectrum = self.evol_spectrum.spectrum[:, 0, :]
        originalSpectrum = self.evol_spectrum.getSpectrumOfOrder(0)
        randomizedCoeffs = np.empty((self.maxScale, self.lengthSignal), dtype=np.float64)
        for z in range(self.lengthSignal):
            randomizedCoeffs[:, z] = self.randomizeCoeffsAtTimeZ(z, originalSpectrum[:, z])
        return randomizedCoeffs
    
    def randomizeCoeffsAtTimeZ(self, z:int, spectrumAtTimeZ:np.ndarray):
        mean = np.repeat(0, len(spectrumAtTimeZ))
        cov = self.incrementsCorrelationMatrix[:, :, z]
        randomizedCoeffsAtTimeZ = np.sqrt(spectrumAtTimeZ) * np.random.multivariate_normal(mean, cov, size=1).ravel()
        return randomizedCoeffsAtTimeZ
    
    def simulateScale(self, scaleCoeffs:np.ndarray, scale:int):
        discreteWavelet = self.evol_spectrum.crossWavelet.discritization[scale, :self.lengthSignal]
        return utils.fft_convolve(scaleCoeffs, discreteWavelet)
        
    def simulate(self):
        self.__initializeSpectrumIfNot()
        self.simulation = np.zeros((self.lengthSignal,), dtype=np.float64)
        randomizedCoeffs = self.getRandomizedCoeffs()
        for i in range(self.maxScale):
            self.simulation += self.simulateScale(randomizedCoeffs[i, :], i)
        
    def graph(self):
        fig, ax = plt.subplots(1,1,figsize=(17, 8))
        ax.plot(self.simulation)

    def __initializeSpectrum(self):
        if self.isSignalASpectrum:
            self.evol_spectrum = ews.EWS(self.signal, isSpectrum=self.isSignalASpectrum, order=self.order, wavelet=self.wavelet)
        else:
            self.evol_spectrum = ews.EWS(self.decomposition.details, isSpectrum=self.isSignalASpectrum, order=self.order, wavelet=self.wavelet)
        self.evol_spectrum.setIncrementsCorrelationMatrix(self.incrementsCorrelationMatrix)
        
    def __isSignalASpectrum(self):
        if self.signal.ndim == 1:
            self.isSignalASpectrum = False
        else:
            self.isSignalASpectrum = True
            
    def __computeMaxScale(self):
        filterLength = len(np.array(pywt.Wavelet(self.wavelet_name).dec_lo))
        if self.isSignalASpectrum:
            self.maxScale = self.signal.shape[0]
            self.lengthSignal = self.signal.shape[1]
            self.__getMaxScaleBelowTwoTimeSignalLength(filterLength)
            self.signal = self.signal[:self.maxScale]
        else:
            self.maxScale = int(np.floor(np.log(len(self.signal))/np.log(2)))
            self.lengthSignal = len(self.signal)
            self.__getMaxScaleBelowTwoTimeSignalLength(filterLength)
        
    def __getMaxScaleBelowTwoTimeSignalLength(self, filterLength:int):
        while (2**self.maxScale - 1)*(filterLength-1)+1 > 2*self.lengthSignal:
            self.maxScale -= 1
            
    def __initializeDecomposition(self):
        if not self.isSignalASpectrum:
            self.decomposition = dec.WaveletDecomposition(self.signal, self.wavelet)
            
    def __initializeIncrementCorrelationMatrixIfNot(self):
        if not  utils.isArrayInitialized(self, 'incrementsCorrelationMatrix'):
            self.incrementsCorrelationMatrix = np.zeros((self.maxScale, self.maxScale, self.lengthSignal), dtype=np.float64)
    
    def __initializeSpectrumIfNot(self):
        if not utils.isArrayInitialized(self, 'evol_spectrum'):
            self.__initializeSpectrum()
    
    def __checkSignalLengthMatch(self, length:int):
        if length != self.lengthSignal:
            raise ValueError("Length of the array doesn't match length of signal")
        
# class MLSW:
#     evol_spectrum:ews.CrossSpectrum
#     def __init__(self, signals:np.ndarray, wavelet_name:str, order:int=0):
#         self.signals = signals
#         self.wavelet_name = wavelet_name
#         self.order = order
    
    
    
@nb.njit(nogil=True)
def matchFirstRowLengthToMaxScale(FirstRow:np.ndarray, maxScale:int):
    if len(FirstRow) >= (maxScale):
        return FirstRow[:maxScale]
    else:
        return np.concatenate((FirstRow, np.repeat(0.0, maxScale-len(FirstRow))))
    
@nb.njit(nogil=True)
def incrCorrMatrixAtTimeZ(FirstRow:np.ndarray, maxScale:int):
    FirstRow = matchFirstRowLengthToMaxScale(FirstRow, maxScale)
    buildingMatrix = np.zeros((maxScale, maxScale), dtype=np.float64)
    
    for i in range(maxScale):
        buildingMatrix[i] = FirstRow
        FirstRow = np.roll(FirstRow, 1)
        
    buildingMatrix = np.triu(buildingMatrix)
    return buildingMatrix + buildingMatrix.T - np.diag(np.diag(buildingMatrix))

@nb.njit(nogil=True, fastmath=True)
def getCorrelationMatrixFromCovMatrix(cov:np.ndarray):
    std = np.sqrt(np.diag(cov))
    std = np.outer(std, std)
    return cov / std

        