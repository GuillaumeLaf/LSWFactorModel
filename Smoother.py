import numpy as np
import matplotlib.pyplot as plt
import custom_wavelets as w
import WaveletDecomposition as dec
import wavelet_utils as utils

class Smoother:
    def __init__(self):
        pass
    
    def smooth(self, signal:np.ndarray):
        pass
    
class Kernel_smoother(Smoother):
    def __init__(self, name:str, window:int):
        self.name = name
        self.window = window
        self.kernel = np.empty((window, ), dtype=np.float64)
        self.kernelFunction = np.vectorize(self.__getKernelFunctionFromName())
        self.discritize()
    
    def discritize(self):
        grid = np.linspace(-1, 1, self.window)
        self.kernel = self.kernelFunction(grid)
        self.kernel = self.kernel / np.sum(self.kernel)
        
    def smooth(self, signal:np.ndarray):
        return utils.fft_convolve(signal, self.kernel)
        
    def graph(self):
        fig, ax = plt.subplots(1,1, figsize=(12, 8))
        ax.plot(self.kernel, 'o')
        ax.axhline(y=0)
    
    def __getKernelFunctionFromName(self):
        return {
            'Simple': lambda x: 1., 
            'Triangular': lambda x: (1.- np.abs(x)), 
            'Epanechnikov': lambda x: 3./4. * (1 - x*x), 
            'Gaussian': lambda x: 3./np.sqrt(2*np.pi) * np.exp(-4.5*x*x), 
            'Silverman': lambda x: 2.5 * np.exp(-np.abs(5*x)/np.sqrt(2)) * np.sin(np.abs(5*x)/np.sqrt(2) + np.pi/4.)
            }.get(self.name, lambda x:1)

class SWT_smoother(Smoother):
    decomposition:dec.WaveletDecomposition
    def __init__(self, wavelet:w.Wavelet, smooth_type:str):
        self.wavelet = wavelet
        self.thresholder = Threshold(smooth_type)
        self.thresh_func = np.vectorize(self.__thresholdDetail, signature='(n)->(n)')
        
    def smooth(self, signal:np.ndarray):
        signal, n_extend = self.__matchSignalLengthToPower2(signal)
        self.decomposition = dec.WaveletDecomposition(signal, self.wavelet)
        self.decomposition.rotateDecomposition()
        self.decomposition.details[:3] = 0.0
        self.decomposition.details[3:] = self.thresh_func(self.decomposition.details[3:])
        
        if n_extend == 0:
            return self.decomposition.SWTReconstruct()
        else:
            return self.decomposition.SWTReconstruct()[:-n_extend]
        
    def __thresholdDetail(self, detail:np.ndarray):
        return self.thresholder.threshold(detail)
    
    def __matchSignalLengthToPower2(self, signal:np.ndarray):
        i = 0
        while len(signal) > 2**i:
            i += 1
        n_extend = 2**i - len(signal)
        zeros = np.zeros((n_extend,), dtype=np.float64)
        return np.concatenate([signal, zeros]), n_extend
        
class Threshold:
    thresh:np.float64
    func:None
    def __init__(self, name:str):
        self.name = name
        
    def threshold(self, signal:np.ndarray):
        self.__initializeThreshAndFuncIfNot(signal)
        return self.func(signal)
    
    def __getUniversalThresholdValue(self, signal:np.ndarray):
        return np.std(signal) * np.sqrt(2. * np.log(len(signal)))
    
    def __getThresholdFunctionFromName(self):
        return {
            'hard':self.__hardThresholdingFunction, 
            'soft':self.__softThresholdingFunction
            }.get(self.name, self.__softThresholdingFunction)
    
    def __hardThresholdingFunction(self, x):
        if np.abs(x)  > self.thresh:
            return x
        else:
            return 0.
    
    def __softThresholdingFunction(self, x):
        if np.abs(x) - self.thresh >= 0.:
            return np.sign(x) * (np.abs(x) - self.thresh)
        else:
            return 0.
        
    def __initializeThreshAndFuncIfNot(self, signal:np.ndarray):
        if not utils.isArrayInitialized(self, 'thresh'):
            self.thresh = self.__getUniversalThresholdValue(signal)
            self.func = np.vectorize(self.__getThresholdFunctionFromName())
        
        
        
x = np.array([1,5,2,4,6,3,9,8,2,0,1,2,5,4,2], dtype=np.float64)
        
        
        
        
        
        
        
        
        
        
        
        
        

        
