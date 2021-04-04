import numpy as np
import pywt
import wavelet_utils as utils
import matplotlib.pyplot as plt

class Wavelet:
    discritization:np.ndarray
    def __init__(self, name:str, maxScale:int):
        """
        
        The Wavelet Class is simply an interface for the PyWavelet Wavelet class. 
        This interface allows simplifications among our environment.
        
        Parameters
        ----------
        name : str
            Name of the wavelet. The name must be the same as in the PyWavelet Library.
        maxScale : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.name = name
        self.pywtWavelet = pywt.Wavelet(name)
        self.maxScale = maxScale
        self.maxLength = self.getWaveletLength(self.maxScale+1)
        self.dec_lo = np.array(self.pywtWavelet.dec_lo)
        self.dec_hi = np.array(self.pywtWavelet.dec_hi)
        self.rec_lo = np.array(self.pywtWavelet.rec_lo)
        self.rec_hi = np.array(self.pywtWavelet.rec_hi)
        
    def discretizeToMaxScale(self):
        self.__initializeDiscritizationArrayIfNot()
        for i in range(self.maxScale):
            waveletLength = self.getWaveletLength(i)
            self.discritization[i, :waveletLength] = self.discretizeOneScale(i)
        
    def __initializeDiscritizationArrayIfNot(self):
        if not utils.isArrayInitialized(self, 'discritization'):
            self.discritization = np.zeros((self.maxScale, self.maxLength), dtype=np.float64)
    
    def getWaveletLength(self, scale:int, cross_scale:int=0):
        filter_length = len(self.pywtWavelet.dec_lo)
        return int((2**(scale+cross_scale+1) - 1)*(filter_length - 1) + 1)
        
    def discretizeOneScale(self, scale:int):
        mother = self.pywtWavelet.wavefun(level=scale+1)[1]/np.sqrt(2**(scale+1))
        return np.trim_zeros(np.array(mother))
    
class CrossCorrelationWavelet(Wavelet):
    A_operator:np.ndarray
    columnOrderIndexing:np.ndarray
    phi_operator:np.ndarray
    def __init__(self, name:str, maxScale:int, order:int):
        super().__init__(name, maxScale)
        self.order = order
        self.discretizeToMaxScale()
        self.initializeA_operator()
        
    def getA_operatorAtOrder(self, i:int, r:int, trimmed:bool):
        idx_i_mask = (np.concatenate(self.columnOrderIndexing) == i)
        idx_r_mask = (np.concatenate(self.columnOrderIndexing) == r)
        
        operator_mask = np.outer(idx_i_mask, idx_r_mask)
        shape_i = self.maxScale - np.abs(i)
        shape_r = self.maxScale - np.abs(r)
        if not trimmed:
            out = np.zeros((self.maxScale, self.maxScale), dtype=np.float64)
            out[:shape_i, :shape_r] = self.A_operator[operator_mask].reshape(shape_i, shape_r)
        else:
            out = self.A_operator[operator_mask].reshape(shape_i, shape_r)
        return out
        
    def initializeA_operator(self):
        self.initializePhi_operator()
        self.A_operator = self.phi_operator.T @ self.phi_operator
        self.A_operator = self.__deleteExtraColumns(self.A_operator)
        self.A_operator = np.linalg.inv(self.A_operator)
        
    def __deleteExtraColumns(self, matrix:np.ndarray):
        col_order = np.concatenate(self.columnOrderIndexing)
        del_idx = np.arange(len(col_order))
        del_idx = del_idx[col_order < 0]
        matrix = np.delete(matrix, del_idx, axis=0)
        matrix = np.delete(matrix, del_idx, axis=1)
        
        for j in range(self.columnOrderIndexing.shape[0]):
            self.columnOrderIndexing[j] = self.columnOrderIndexing[j][self.columnOrderIndexing[j] >= 0]
        return matrix
    
    def initializePhi_operator(self):
        self.phi_operator = np.empty((self.maxLength, ), dtype=np.float64)
        col_order_j = []
        for j in range(self.maxScale):
            col_order_i = []
            self.__stackCCWFatScale(j, col_order_i)
            col_order_j.append(np.array(col_order_i))
            
        self.phi_operator = np.delete(self.phi_operator, 0, axis=1)
        self.columnOrderIndexing = np.array(col_order_j)
    
    def __stackCCWFatScale(self, j:int, col_order:list):
        mn = np.max([-self.maxScale+j+1, -self.order+1])
        mx = np.min([self.maxScale-j-1, self.order-1]) + 1
        for i in range(mn, mx):
            col_order.append(i)
            if i >= 0:
                self.phi_operator = np.column_stack((self.phi_operator, utils.fft_ConjugateConvolve(self.discritization[j+i], self.discritization[j])))
            else:
                self.phi_operator = np.column_stack((self.phi_operator, utils.fft_ConjugateConvolve(self.discritization[j], self.discritization[j-i])))
            
    
w = CrossCorrelationWavelet('db1', 3, 3)