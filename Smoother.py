import numpy as np
import matplotlib.pyplot as plt
import custom_wavelets as w
import WaveletDecomposition as dec
import wavelet_utils as utils

class Smoother:
    def __init__(self):
        """
        Interface class used to aggregate multiple smoothing methods.
        This class will make adding new type of smoothing method easier.

        Returns
        -------
        None.

        """
        
        pass
    
    def smooth(self, signal:np.ndarray):
        pass
    
class Kernel_smoother(Smoother):
    def __init__(self, name:str, window:int):
        """
        Smoother Object based on a kernel.

        Parameters
        ----------
        name : str
            Name of the kernel.
        window : int
            Window of the kernel.

        Returns
        -------
        None.

        """
        
        self.name = name
        self.window = window
        self.kernel = np.empty((window, ), dtype=np.float64)
        self.kernelFunction = np.vectorize(self.__getKernelFunctionFromName())
        self.discritize()
    
    def discritize(self):
        """
        Discretize the continuous kernel.

        Returns
        -------
        None.

        """
        
        grid = np.linspace(-1, 1, self.window)
        self.kernel = self.kernelFunction(grid)
        self.kernel = self.kernel / np.sum(self.kernel)
        
    def smooth(self, signal:np.ndarray):
        """
        Smooth the signal provided

        Parameters
        ----------
        signal : np.ndarray
            Signal that needs to be smoothed.

        Returns
        -------
        np.ndarray
            Smoothed signal.

        """
        return utils.fft_convolve(signal, self.kernel)
        
    def graph(self):
        """
        Graph the kernel coefficients

        Returns
        -------
        None.

        """
        
        fig, ax = plt.subplots(1,1, figsize=(12, 8))
        ax.plot(self.kernel, 'o')
        ax.axhline(y=0)
    
    def __getKernelFunctionFromName(self):
        """
        Function that allows to get the kernel equation with its name.

        Returns
        -------
        Func
            Function of the kernel.

        """
        
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
        """
        This smoother is based on the de-noising wavelet method of Donoho (1995) "De-noising by soft-thresholding"

        Parameters
        ----------
        wavelet : w.Wavelet
            Wavelet object that will be used to smooth.
        smooth_type : str
            smoohting type : soft - hard

        Returns
        -------
        None.

        """
        
        self.wavelet = wavelet
        self.thresholder = Threshold(smooth_type)
        self.thresh_func = np.vectorize(self.__thresholdDetail, signature='(n)->(n)')
        
    def smooth(self, signal:np.ndarray):
        """
        Smooth the provided signal.
        Note that the stationary wavelet transform rotates each scale of the decomposition. 
        To recover the smoothed signal we therefore need to rotate the decomposition backwards beginning from the end.
        We also assume the first 3 scales to only be noise. Thus, we discard them completely -i.e. setting the wavelet coefficients to zero.
        The signal also need to be a multiple of 2, hence we add zeros at the end of the array if nedded.
        Finally we apply the thresholding method (soft, hard).
        
        Parameters
        ----------
        signal : np.ndarray
            Signal that needs to be smoothed.

        Returns
        -------
        np.ndarray
            Smoothed Signal.

        """
        
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
        # This function is self explanatory
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
        """
        Class containing the thresholding methods.
        This class is once again used to facilitate adding new types of thresholding methods.

        Parameters
        ----------
        name : str
            Name of the thresholding method.

        Returns
        -------
        None.

        """
        
        self.name = name
        
    def threshold(self, signal:np.ndarray):
        self.__initializeThreshAndFuncIfNot(signal)
        return self.func(signal)
    
    def __getUniversalThresholdValue(self, signal:np.ndarray):
        """
        Function that computes the universal threshold value for the provided array

        Parameters
        ----------
        signal : np.ndarray
            Array.

        Returns
        -------
        TYPE
            Universal Threshold.

        """
        
        return np.std(signal) * np.sqrt(2. * np.log(len(signal)))
    
    def __getThresholdFunctionFromName(self):
        """
        Function that allows to retrieve easily the tresholding method from its name.
        Note that the function returned here only apply to a single scalar and will have to be vectorized later.

        Returns
        -------
        Function
            Tresholding function.

        """
        
        return {
            'hard':self.__hardThresholdingFunction, 
            'soft':self.__softThresholdingFunction
            }.get(self.name, self.__softThresholdingFunction)
    
    def __hardThresholdingFunction(self, x):
        """
        Hard tresholding function.

        Parameters
        ----------
        x : TYPE
            Scalar on which to apply the treshold.

        Returns
        -------
        TYPE
            Tresholded input.

        """
        
        if np.abs(x)  > self.thresh:
            return x
        else:
            return 0.
    
    def __softThresholdingFunction(self, x):
        """
        Soft tresholding function.

        Parameters
        ----------
        x : TYPE
            Scalar on which to apply the treshold.

        Returns
        -------
        TYPE
            Tresholded input.

        """
        
        if np.abs(x) - self.thresh >= 0.:
            return np.sign(x) * (np.abs(x) - self.thresh)
        else:
            return 0.
        
    def __initializeThreshAndFuncIfNot(self, signal:np.ndarray):
        if not utils.isArrayInitialized(self, 'thresh'):
            self.thresh = self.__getUniversalThresholdValue(signal)
            self.func = np.vectorize(self.__getThresholdFunctionFromName())
        
        
        
x = np.array([1,5,2,4,6,3,9,8,2,0,1,2,5,4,2], dtype=np.float64)
        
        
        
        
        
        
        
        
        
        
        
        
        

        
