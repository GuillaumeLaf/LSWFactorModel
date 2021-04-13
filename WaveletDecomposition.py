import numpy as np
import pywt
import custom_wavelets as w
import wavelet_utils as utils
import matplotlib.pyplot as plt

class WaveletDecomposition:
    details:np.ndarray
    approx:np.ndarray
    
    def __init__(self, signal:np.ndarray, wavelet:w.Wavelet):
        self.signal = signal
        self.wavelet = wavelet
        
        self.SWTdecompose(norm=False)
    
    # First scale at the beginning of array
    def SWTdecompose(self, norm:bool):
        self.__initializeDetailsAndApproxArrayIfNot()
        low_pass = self.wavelet.dec_lo
        high_pass = self.wavelet.dec_hi
        
        approx = np.copy(self.signal)
        for i in range(self.wavelet.maxScale):
            details = utils.fft_convolve(approx, high_pass)/self.__getNormalizeConstant(norm)
            approx = utils.fft_convolve(approx, low_pass)/self.__getNormalizeConstant(norm)
            
            details = np.roll(details, -(2**i))
            approx = np.roll(approx, -(2**i))
            
            low_pass = utils.upsample(low_pass)
            high_pass = utils.upsample(high_pass)
            
            self.details[i] = details
        
        self.approx = approx
        
    def SWTReconstruct(self):
        for i in range(self.wavelet.maxScale):
            if i == 0:
                reconstruction = np.fft.fft(self.approx)
            else:
                reconstruction = np.fft.fft(reconstruction)
            rec_lo = self.wavelet.rec_lo
            rec_hi = self.wavelet.rec_hi
            
            for j in range(self.wavelet.maxScale-1-i):
                rec_lo = utils.upsample(rec_lo)
                rec_hi = utils.upsample(rec_hi)
            
            rec_lo = utils.adjustSecondArraySizeToFirst(self.signal, rec_lo)
            rec_hi = utils.adjustSecondArraySizeToFirst(self.signal, rec_hi)
            
            rec_lo = np.fft.fft(rec_lo)
            rec_hi = np.fft.fft(rec_hi)
            
            d = self.details[self.wavelet.maxScale - i - 1]
            d = np.fft.fft(d)
            
            convolutionApprox = reconstruction * rec_lo
            convolutionDetail = d * rec_hi
            convolutionApprox = np.real(np.fft.ifft(convolutionApprox))
            convolutionDetail = np.real(np.fft.ifft(convolutionDetail))
            
            reconstruction = (convolutionApprox + convolutionDetail)/2.
        return reconstruction
            
    def __getNormalizeConstant(self, norm:bool):
        if norm:
            return np.sqrt(2)
        else:
            return 1.0
        
    def __initializeDetailsAndApproxArrayIfNot(self):
        if not utils.isArrayInitialized(self, 'details'):
            self.details = np.zeros((self.wavelet.maxScale, len(self.signal)), dtype=np.float64)
            self.approx = np.zeros((1, len(self.signal)), dtype=np.float64)
            

    
  