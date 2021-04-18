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
        """
        This class contains everything related to the Locally Stationary Wavelet model of Nason and al. (2000).

        Parameters
        ----------
        signal : np.ndarray
            Signal that is assumed to follow a LSW process.
            The signal could be a spectrum since it is possible to simulate a process given a particular Evolutionary Wavelet Spectrum.
        wavelet_name : str
            Name of the wavelet used to model the process.
        order : int, optional
            Order of the model according to Koch (2015). The default is 0.

        Returns
        -------
        None.

        """
        
        self.signal = signal
        self.order = order
        self.wavelet_name = wavelet_name
        
        self.__isSignalASpectrum()
        self.__computeMaxScale()
        
        self.wavelet = w.Wavelet(wavelet_name, self.maxScale)
        
        self.__initializeDecomposition()
    
    def getConstantIncrementsCorrelationRows(self, firstRow:np.ndarray):
        """
        Get a constant correlation matrix for the increments of the process.

        Parameters
        ----------
        firstRow : np.ndarray
            First row of the correlation matrix.
            Note that the correlation matrix will be adapted in order to have a semi-positive matrix.

        Returns
        -------
        np.ndarray
            return the repeated row for the length of the signal.

        """
        
        firstRow = firstRow[np.newaxis, :]
        return np.repeat(firstRow, self.lengthSignal, axis=0)
    
    def addIncrementsCorrelationMatrix(self, RowsForEachZ:np.ndarray):
        """
        

        Parameters
        ----------
        RowsForEachZ : np.ndarray
            This array specifies the first row of the correlation matrix of the LSW increments.
            This array should be of shape : (FirstRowCorrelation, length_signal)

        Returns
        -------
        None.

        """
        
        self.__initializeIncrementCorrelationMatrixIfNot()
        # Check if the first row provided has the same shape of the number of scales. 
        # If not, add zeros at the end of the 'FirstRow' array.
        self.__checkSignalLengthMatch(RowsForEachZ.shape[0]) 
        for z in range(self.lengthSignal):
            buildingMatrix = incrCorrMatrixAtTimeZ(RowsForEachZ[z], self.maxScale)
            
            # This line allows to get a semi-positive matrix
            buildingMatrix = buildingMatrix @ buildingMatrix.T  
            
            self.incrementsCorrelationMatrix[:, :, z] = getCorrelationMatrixFromCovMatrix(buildingMatrix)
        
        self.__initializeSpectrum()
        
    def getRandomizedCoeffs(self):
        """
        This function gets the randomized coefficients for each time 'z'.

        Returns
        -------
        randomizedCoeffs : np.ndarray

        """
        
        originalSpectrum = self.evol_spectrum.getSpectrumOfOrder(0)
        randomizedCoeffs = np.empty((self.maxScale, self.lengthSignal), dtype=np.float64)
        for z in range(self.lengthSignal):
            randomizedCoeffs[:, z] = self.randomizeCoeffsAtTimeZ(z, originalSpectrum[:, z])
        return randomizedCoeffs
    
    def randomizeCoeffsAtTimeZ(self, z:int, spectrumAtTimeZ:np.ndarray):
        """
        Get the randomized coefficients from the EWS at one particular time 'z'.
        Note that we provide the EWS but the random coefficient is defined from the sqrt of the EWS.
        By having an identity matrix for the covariance of the increments, we recover the original model of Nason and al. (2000).
        If the covariance matrix is not identity, we get the model developped by Koch (2015).

        Parameters
        ----------
        z : int
            The particular time.
        spectrumAtTimeZ : np.ndarray
            Spectrum at time 'z'

        Returns
        -------
        randomizedCoeffsAtTimeZ : np.ndarray
            DESCRIPTION.

        """
        
        mean = np.repeat(0, len(spectrumAtTimeZ))
        cov = self.incrementsCorrelationMatrix[:, :, z]
        randomizedCoeffsAtTimeZ = np.sqrt(spectrumAtTimeZ) * np.random.multivariate_normal(mean, cov, size=1).ravel()
        return randomizedCoeffsAtTimeZ
    
    def simulateScale(self, scaleCoeffs:np.ndarray, scale:int):
        """
        Simulate one particular scale. The simulation is done via convolving the randomized coefficients with the given discrete wavelet.

        Parameters
        ----------
        scaleCoeffs : np.ndarray
            Randomized coefficients of the LSW model.
        scale : int
            Scale at which we simulate the LSW model.

        Returns
        -------
        np.ndarray
            Simulated scale.

        """
        
        discreteWavelet = self.evol_spectrum.crossWavelet.discritization[scale, :self.lengthSignal]
        return utils.fft_convolve(scaleCoeffs, discreteWavelet)
        
    def simulate(self):
        """
        This function allows to simulate the LSW process from the Evolutionary Wavelet Spectrum (provided or estimated)

        Returns
        -------
        None.

        """
        
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
            
"""
    The following functions could have been placed inside their corresponding classes.
    However, in order to speed up the computation using Numba, we have to extract them from the class environment.
    The 'nb.njit' decorator allows to get speed of the function comparable to C.
"""
    
@nb.njit(nogil=True)
def matchFirstRowLengthToMaxScale(FirstRow:np.ndarray, maxScale:int):
    if len(FirstRow) >= (maxScale):
        return FirstRow[:maxScale]
    else:
        return np.concatenate((FirstRow, np.repeat(0.0, maxScale-len(FirstRow))))
    
@nb.njit(nogil=True)
def incrCorrMatrixAtTimeZ(FirstRow:np.ndarray, maxScale:int):
    """
    This function constructs the correlation matrix from the first row provided.

    Parameters
    ----------
    FirstRow : np.ndarray
        First row that will be replicated in the covariance matrix.
    maxScale : int
        Maximum scale of the wavelet decomposition in the LSW process.

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """
    
    FirstRow = matchFirstRowLengthToMaxScale(FirstRow, maxScale)
    buildingMatrix = np.zeros((maxScale, maxScale), dtype=np.float64)
    
    for i in range(maxScale):
        buildingMatrix[i] = FirstRow
        FirstRow = np.roll(FirstRow, 1)
        
    buildingMatrix = np.triu(buildingMatrix)
    return buildingMatrix + buildingMatrix.T - np.diag(np.diag(buildingMatrix))

@nb.njit(nogil=True, fastmath=True)
def getCorrelationMatrixFromCovMatrix(cov:np.ndarray):
    """
    From a covariance matrix, get the associated correlation matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """
    
    std = np.sqrt(np.diag(cov))
    std = np.outer(std, std)
    return cov / std

        