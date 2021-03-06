import numpy as np
import pywt
import matplotlib.pyplot as plt
import custom_wavelets as w
import wavelet_utils as utils
import WaveletDecomposition as dec
import Smoother as smo
import numba as nb


plt.style.use('ggplot')

class EWS:
    spectrum:np.ndarray     #The spectrum does not include the approx. level (only the details coefficients)
    incrementsCorrelationMatrix:np.ndarray
    
    def __init__(self, decomposition:np.ndarray, isSpectrum:bool, order:int, wavelet:w.Wavelet):
        self.decomposition = decomposition
        self.isSpectrum = isSpectrum
        self.order = order
        self.crossWavelet = w.CrossCorrelationWavelet(wavelet.name, wavelet.maxScale, self.order)
        self.columnOrderIndexing = self.crossWavelet.columnOrderIndexing
        
        self.__getSpectrum()
        
    def updateDecomposition(self, decomposition:np.ndarray):
        """
        This function is usefull if we want to iterate over different decomposition without reinitializing the objects.
        I essentially use this function when checking for convergence of the EWS.

        Parameters
        ----------
        decomposition : np.ndarray
            New decomposition.

        Returns
        -------
        None.

        """
        
        self.decomposition = decomposition
        self.__getSpectrum()        
        
    def setIncrementsCorrelationMatrix(self, correlationMatrix:np.ndarray):
        self.incrementsCorrelationMatrix = correlationMatrix
        self.__scaleSpectrumIfSimulation()
    
    def graph(self, order:int=0, sharey:bool=True):
        n_scales = np.where(np.concatenate(self.columnOrderIndexing) == order, 1, 0).sum()
        fig, ax = plt.subplots(n_scales, 1, figsize=(12, 15), sharex=True, sharey=sharey)
        ax = np.ravel(ax)
        for j in range(n_scales):
            ax[j].plot(self.getSpectrumOfScaleAndOrder(j, order))
            ax[j].fill_between(np.arange(0,self.decomposition.shape[1]), self.getSpectrumOfScaleAndOrder(j, order))
            ax[j].spines["top"].set_visible(False)
            ax[j].spines["right"].set_visible(False)
            ax[j].spines["left"].set_visible(False)
            ax[j].set_facecolor("None")
            ax[j].get_yaxis().set_visible(False)
            # ax[j].set_ylabel(f'Scale -{j+1}')
            # ax[j].tick_params(axis='y', labelrotation=-90)
        # fig.savefig("plot_test", facecolor="None", edgecolor="b")
            
    def correctSpectrum(self):
        """
        This function corrects the spectrum with the 'A_operator' computed in the 'custom_wavelet.py' module.

        Returns
        -------
        None.

        """
        
        temp_spectrum = np.zeros_like(self.spectrum)
        idx_i = np.concatenate(self.columnOrderIndexing)
        
        # Iterate over all possible/unique 'orders'. 
        for i in set(idx_i):
            # Select the indices of the flattened array 'columnOrderIndexing' where the order is "i".
            idx = np.arange(len(idx_i))[idx_i == i]
            temp_spectrum[idx, :] = self.__correctSpectrumOfOrder(i, idx)
        self.spectrum = temp_spectrum
        
    def __correctSpectrumOfOrder(self, i:int, idx:np.ndarray):
        """
        This function corrects the spectrum of a particular order -i.e. "i".

        Parameters
        ----------
        i : int
            Order of the spectrum.
        idx : np.ndarray
            Array of indices of the flattened array 'columnOrderIndexing' where the order is "i".

        Returns
        -------
        correctedSpectrum : np.ndarray
            The corrected spectum of order "i".

        """
        
        correctedSpectrum = np.zeros((len(idx), self.spectrum.shape[1]), dtype=np.float64)
        for r in np.unique(np.concatenate(self.columnOrderIndexing)):
            correctedSpectrum += (self.crossWavelet.getA_operatorAtOrder(i, r, trimmed=False) @ self.getSpectrumOfOrder(r))[:len(idx), :]
        return correctedSpectrum
    
    def smoothSpectrum(self, smoother:smo.Smoother):
        """
        This function smooths the spectrum according to the specified smoother.

        Parameters
        ----------
        smoother : smo.Smoother
            smoother used to smooth the spectrum.

        Returns
        -------
        None.

        """
        
        for i in range(self.spectrum.shape[0]):
            self.spectrum[i, :] = smoother.smooth(self.spectrum[i, :])
    
    def getSpectrumOfScaleAndOrder(self, j:int, i:int):
        """
        This function extracts the spectrum at a given scale "j" and order "i".

        Parameters
        ----------
        j : int
            Scale.
        i : int
            Order.

        Returns
        -------
        np.ndarray
            Spectrum of scale "j" and order "i".

        """
        
        orders = np.concatenate(self.columnOrderIndexing)
        mask = orders == i
        idx_mask = np.arange(orders.shape[0])
        idx_mask = idx_mask[mask]
        idx_j = idx_mask[j]
        return self.spectrum[idx_j]
    
    def getSpectrumOfOrder(self, i:int):
        """
        Extract the spectrum of order "i" given the flattened array 'columnOrderIndexing'.

        Parameters
        ----------
        i : int
            Order to be extracted.

        Returns
        -------
        np.ndarray
            Return the spectrum of the given order "i".

        """
        empty_spectrum = np.zeros((self.crossWavelet.maxScale, self.spectrum.shape[1]), dtype=np.float64)
        
        orders = np.concatenate(self.columnOrderIndexing)
        mask = orders == i
        empty_spectrum[:empty_spectrum.shape[0] - np.abs(i), :] = self.spectrum[mask, :].reshape(-1, self.spectrum.shape[1])
        return empty_spectrum
        
    def __getSpectrum(self):
        """
        Function that initialize the spectrum given the decomposition.
        If 'self.decomposition' is already a spectrum (as indicated by 'self.isSpectrum')
        then we must that the square root of it.
        
        Note that the 'spectrum' array is of dimension (length of columnOrderIndexing, length of signal)
        The 'spectrum' array is thus not necessarily in a logical order. 
        We must once again use the 'columnOrderIndexing' to find the right scale and order.

        Returns
        -------
        None.

        """
        
        self.__initializeSpectrumArrayIfNot()
        if self.isSpectrum:
            self.__getCrossSpectrum(np.sqrt(self.decomposition))
        else:
            self.__getCrossSpectrum(self.decomposition)
    
    def __getCrossSpectrum(self, decomp:np.ndarray):
        """
        Function that sets the spectrum given the decomposition given as parameter.
        The spectrum is filled in accordance with the array 'columnOrderIndexing'.
        
        Note that even if negative orders are allowed they produce the same spectrum as positive orders, hence creating storing redundant information.
        However, let the redundancy be, as to be more consistent with the spectrum of the CrossEWS class.

        Parameters
        ----------
        decomp : np.ndarray
            Decomposition used to create the spectrum.

        Returns
        -------
        None.

        """
        
        counter = 0
        for idx_scale, scale in enumerate(self.columnOrderIndexing):
             for order in scale:
                 self.spectrum[counter, :] = decomp[idx_scale, :] * decomp[idx_scale + order, :]
                 counter += 1
            
    def __getIncrementsCorrelationScalingSpectrum(self):
        return np.array([[self.incrementsCorrelationMatrix[j, j+i, z] 
                          for j, orders in enumerate(self.columnOrderIndexing) 
                          for i in orders] 
                         for z in range(self.decomposition.shape[1])]).T
    
    def __scaleSpectrumIfSimulation(self):
        if self.__isSimulation():
            scalingSpectrum = self.__getIncrementsCorrelationScalingSpectrum()
            self.spectrum = np.multiply(self.spectrum, scalingSpectrum)       
            
    def __initializeSpectrumArrayIfNot(self):
            if not utils.isArrayInitialized(self, 'spectrum'):
                self.spectrum = np.zeros((np.concatenate(self.columnOrderIndexing).shape[0], self.decomposition.shape[1]), dtype=np.float64)
            
    def __isSimulation(self):
        return utils.isArrayInitialized(self, 'incrementsCorrelationMatrix')    # If it a simulation then we need to have a correlation matrix. Allow us to differentiate between simulation and real decomposition of a signal
        
class CrossEWS:
    spectrum:np.ndarray     #The spectrum does not include the approx. level (only the details coefficients)
    incrementsCorrelationMatrix:np.ndarray
    
    def __init__(self, decomposition:np.ndarray, isSpectrum:bool, order:int, wavelet:w.Wavelet):
        """
        This class is really a one to one correspondence with the EWS class. 
        It is only adapted to the multivariate case of the LSW model.

        Parameters
        ----------
        decomposition : np.ndarray
            Wavelet Decomposition. Size of the array is : (n_signals, n_details, length_signals)
        isSpectrum : bool
            Flag if the decomposition array is already a spectrum. This is used when doing some simulations
        order : int
            Order of the LSW model.
        wavelet : w.Wavelet
            Wavelet used to decompose the signal. This is an instance of the Wavelet class

        Returns
        -------
        None.

        """
        
        self.decomposition = decomposition      # (n_signals, n_details, lenght_signal)
        self.isSpectrum = isSpectrum
        self.order = order
        self.crossWavelet = w.CrossCorrelationWavelet(wavelet.name, wavelet.maxScale, self.order)
        self.columnOrderIndexing = self.crossWavelet.columnOrderIndexing
        
        self.__getSpectrum()
    
    def updateDecomposition(self, decomposition:np.ndarray):
        self.decomposition = decomposition
        self.__getSpectrum()
        
    def graph(self, u:int, v:int, order:int=0, sharey:bool=True):
        n_scales = np.where(np.concatenate(self.columnOrderIndexing) == order, 1, 0).sum()
        fig, ax = plt.subplots(n_scales, 1, figsize=(12, 15), sharex=True, sharey=sharey)
        ax = np.ravel(ax)
        for j in range(n_scales):
            ax[j].plot(self.getSpectrumOfScaleAndOrder(u, v, j, order))
            ax[j].set_ylabel(f'Scale -{j+1}')
        
    def smoothSpectrum(self, smoother:smo.Smoother):
        """
        Smooth the spectrum with the provided Smoother 

        Parameters
        ----------
        smoother : smo.Smoother
            Smoother object.

        Returns
        -------
        None.

        """
        
        for u in range(self.spectrum.shape[0]):
            for v in range(self.spectrum.shape[0]):
                for i in range(self.spectrum.shape[2]):
                        self.spectrum[u, v, i, :] = smoother.smooth(self.spectrum[u, v, i, :])
                
    def correctSpectrum(self):
        """
        Correct the spectrum at every scale and order.

        Returns
        -------
        None.

        """
        idx_i = np.concatenate(self.columnOrderIndexing)
        for u in range(self.spectrum.shape[0]):
            for v in range(self.spectrum.shape[0]):
                temp_spectrum = np.zeros_like(self.spectrum[u, v])
                for i in set(idx_i):
                    idx = np.arange(len(idx_i))[idx_i == i]
                    temp_spectrum[idx, :] = self.__correctSpectrumOfOrder(u, v, i, idx)
                self.spectrum[u, v] = temp_spectrum
        
    def __correctSpectrumOfOrder(self, u:int, v:int, i:int, idx:np.ndarray):
        """
        Correct the spectrum at a given order "i" and between two processes "u" and "v".

        Parameters
        ----------
        u : int
            First process.
        v : int
            Second process.
        i : int
            order that will be corrected.
        idx : np.ndarray
            Index array containing the location in the spectrum array of the EWS between "u" and "v" at order "i".

        Returns
        -------
        correctedSpectrum : TYPE
            Corrected EWS between "u" and "v" at order "i".

        """
        
        correctedSpectrum = np.zeros((len(idx), self.spectrum.shape[3]), dtype=np.float64)
        for r in set(np.concatenate(self.columnOrderIndexing)):
            correctedSpectrum += self.crossWavelet.getA_operatorAtOrder(i, r, trimmed=True) @ self.getSpectrumOfOrder(u, v, r)
        return correctedSpectrum
    
    def getSpectrumOfScaleAndOrder(self, u:int, v:int, j:int, i:int):
        """
        Function that returns the EWS between two processes "u" and "v" for a given scale "j" and order "i".

        Parameters
        ----------
        u : int
            First process.
        v : int
            Second process.
        j : int
            Scale.
        i : int
            Order.

        Returns
        -------
        np.ndarray
            Return the EWS.

        """
        orders = np.concatenate(self.columnOrderIndexing)
        mask = orders == i
        idx_mask = np.arange(orders.shape[0])
        idx_mask = idx_mask[mask]
        idx_j = idx_mask[j]
        return self.spectrum[u, v, idx_j]
    
    def getSpectrumOfOrder(self, u:int, v:int, i:int):
        """
        Function that returns the EWS between two processes "u" and "v" for a given order "i".

        Parameters
        ----------
        u : int
            First process.
        v : int
            Second process.
        i : int
            Order.

        Returns
        -------
        np.ndarray
            Return the EWS.

        """
        
        orders = np.concatenate(self.columnOrderIndexing)
        mask = orders == i
        return self.spectrum[u, v, mask].reshape(-1, self.spectrum.shape[3])

    def getSpectrumForAllSignalsAtTimeZ(self, j:int, i:int, z:int):
        """
        Function that allows us to recover the EWS for all processes at a given scale "j", order "i" and time "z".
        This function is used when constructing the huge "S" stacked matrix (analogue to the covariance matrix).

        Parameters
        ----------
        j : int
            Scale.
        i : int
            Order.
        z : int
            Time.

        Returns
        -------
        TYPE
            EWS for all processes at a given scale "j", order "i" and time "z".

        """
        
        orders = np.concatenate(self.columnOrderIndexing)
        idx = np.arange(len(orders))[orders == i]
        return self.spectrum[:, :, idx[j], z]

    def __getSpectrum(self):
        """
        The series of function attached to this one are used to compute the EWS from the processes' decompositions.

        Returns
        -------
        None.

        """
        
        self.__initializeSpectrumArrayIfNot()
        if self.isSpectrum:
            self.__getCrossSpectrum(np.sqrt(self.decomposition))
        else:
            self.__getCrossSpectrum(self.decomposition)
    
    def __getCrossSpectrum(self, decomp:np.ndarray):
        for u in range(self.spectrum.shape[0]):
            for v in range(self.spectrum.shape[0]):
                self.__getCrossSpectrumBetweenTS(u, v, decomp)
                
    def __getCrossSpectrumBetweenTS(self, u:int, v:int, decomp:np.ndarray):
        counter = 0
        for idx_scale, scale in enumerate(self.columnOrderIndexing):
            for order in scale:
                # if order >= 0:
                self.spectrum[u, v, counter] = decomp[u, idx_scale, :] * decomp[v, idx_scale + order, :]
                # else:
                #     self.spectrum[u, v, counter] = decomp[v, idx_scale, :] * decomp[u, idx_scale + order, :]
                counter += 1

    def __initializeSpectrumArrayIfNot(self):
            if not utils.isArrayInitialized(self, 'spectrum'):
                self.spectrum = np.zeros((self.decomposition.shape[0], self.decomposition.shape[0], np.concatenate(self.columnOrderIndexing).shape[0], self.decomposition.shape[2]), dtype=np.float64)

    
    
    
    
    
    
    
    
    
    
    
    
    
    