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
        We could easily get rid of the dependence on the PyWavelet package by hard coding wavelet filters, ...
        
        Parameters
        ----------
        name : str
            Name of the wavelet. The name must be the same as in the PyWavelet Library.
        maxScale : int
            The maximum scale of the wavelet. Usually computed based on available amount of data.


        """
        
        self.name = name
        self.pywtWavelet = pywt.Wavelet(name)
        self.maxScale = maxScale
        self.maxLength = self.getWaveletLength(self.maxScale + 1)    
        # Maximum wavelet length. Note that the maximum length is twice the actual maximum length of the coarsest wavelet scale (to make sure the convolution with fft works)
        
        # High and Low Decomposition and Reconstruction filters (used to decompose a signal with a particular wavelet)
        self.dec_lo = np.array(self.pywtWavelet.dec_lo)
        self.dec_hi = np.array(self.pywtWavelet.dec_hi)
        self.rec_lo = np.array(self.pywtWavelet.rec_lo)
        self.rec_hi = np.array(self.pywtWavelet.rec_hi)
        
    def discretizeToMaxScale(self):
        """
        Function that allows to get a discretize version of the wavelet

        Returns
        -------
        This function does not return anything. 
        However it fills for each scale the array 'discritization' with the discritize wavelets.
        The first dimension of the array is the scale from 1st scale to 'MaxScale'.
        The second dimension is of size 'maxLength' and contains the discrete wavelet first padded in the 
        end by zeros (as the discritized wavelet have differents lengths for different scales).

        """
        
        self.__initializeDiscritizationArrayIfNot()
        for i in range(self.maxScale):
            waveletLength = self.getWaveletLength(i)
            self.discritization[i, :waveletLength] = self.discretizeOneScale(i)
            
    def discretizeOneScale(self, scale:int):
        """
        

        Parameters
        ----------
        scale : int
            The scale at which we want to have a discritized version of the wavelet.

        Returns
        -------
        An array filled with a discrete wavelet without any padding

        """
        
        mother = self.pywtWavelet.wavefun(level=scale+1)[1]/np.sqrt(2**(scale+1))
        return np.trim_zeros(np.array(mother))
    
    def getWaveletLength(self, scale:int, cross_scale:int=0):
        """
        

        Parameters
        ----------
        scale : int
            Scale of the wavelet.
        cross_scale : int, optional
            cross Scale of the wavelet. The default is 0.

        Returns
        -------
        The length of the wavelet at one particular scale and cross scale.

        """
        
        filter_length = len(self.pywtWavelet.dec_lo)
        return int((2**(scale+cross_scale+1) - 1)*(filter_length - 1) + 1)
    
    def __initializeDiscritizationArrayIfNot(self):
        if not utils.isArrayInitialized(self, 'discritization'):
            self.discritization = np.zeros((self.maxScale, self.maxLength), dtype=np.float64)
    
class CrossCorrelationWavelet(Wavelet):
    A_operator:np.ndarray
    columnOrderIndexing:np.ndarray  # This array allows us to easily know where the CCWF of scale 'j' and order 'i' is in the 'phi_operator' array
    phi_operator:np.ndarray     # Array which stacks the CCWF vertically based on scale and order
    def __init__(self, name:str, maxScale:int, order:int):
        super().__init__(name, maxScale)
        self.order = order
        self.discretizeToMaxScale()
        self.initializeA_operator()
        
    def getA_operatorAtOrder(self, i:int, r:int, trimmed:bool):
        """
        This function returns the inverse of the operator 'A' for a given two given 'orders' -i.e. "i" and "r".
        This function uses the 'columnOrderIndexing' list to extract the wanted orders from the array 'A_operator'.
        
        Parameters
        ----------
        i : int
            first order.
        r : int
            second order.
        trimmed : bool
            If False, this function returns a square matrix of dimension 'maxScale' 
            filled with the operator 'A' and zeros where it is needed.

        Returns
        -------
        out : TYPE
            Either a square matrix if the 'trimmed' option is False.
            Otherwise, it returns the operator 'A' for two given 'orders' -i.e. "i" and "r".

        """
        
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
        """
        Initialize the A operator used in the correction of the Evolutionary Wavelet Spectrum
        This function compute the Gramian Matrix of the phi_operator.
        Then it deletes the extra columns and rows of the operator 'A'.
        Finally it inverts the operator 'A' as required for the correction.

        Returns
        -------
        None.

        """
        
        self.initializePhi_operator()
        self.A_operator = self.phi_operator.T @ self.phi_operator
        self.A_operator = np.linalg.inv(self.A_operator)
        
    
    def initializePhi_operator(self):
        """
        Initialize the phi operator required to get the correction matrix 'A'.
        This operator is basically an array of all the Cross Correlation Wavelet Functions stacked vertically.
        Taking the inner product of that operator with itself gives a Gramian matrix.

        Returns
        -------
        This function does not return anything.

        """
        
        self.phi_operator = np.empty((self.maxLength, ), dtype=np.float64)
        
        # See the docstring of the function '__stackCCWFatScale' to understand the importance of this list.
        col_order_j = []
        for j in range(self.maxScale):
            col_order_i = []
            self.__stackCCWFatScale(j, col_order_i)
            col_order_j.append(np.array(col_order_i))
            
        # Delete the first columns since it was only usefull to get the first stack
        self.phi_operator = np.delete(self.phi_operator, 0, axis=1)     
        
        # Save the 'col_order_j' list permanently in the object.
        # However, it will later be modified when erasing some duplicate columns of the Gramian Matrix.
        self.columnOrderIndexing = np.array(col_order_j)
    
    def __stackCCWFatScale(self, j:int, col_order:list):
        """
        Stack the Cross Correlation Wavelet Function of a particular scale 'j' vertically in an array 'phi_operator' for the order of the CCWF.

        Parameters
        ----------
        j : int
            scale of the CCWF to be stacked.
        col_order : list
            List that saves the way the CCWF are stacked. This list contains the 'order' of the CCWF associated with the scale 'j'.
            ex : col_order = [-2, -1, 0, 1, 2]

        Returns
        -------
        This function does not return anything.
        However it modifies the list 'col_order'. The list is passed by reference, not by value.

        """
        
        # Maximum and minimum order following Daniel Koch's notations
        mx = np.min([self.maxScale-j, self.order+1])
        mn = np.max([-j, -self.order])
        for i in range(mn, mx):
            if i >= 0:
                col_order.append(i)
                
                # The sign of the order is importance since the CCWF are not symmetric. 
                # The CCWF with a negative order is the mirror around the y-axis of the positive order (for a given scale 'j')
                # Eventhough the 'negative is the mirror of the positive', the arrays are not mirror of each other.
                self.phi_operator = np.column_stack((self.phi_operator, utils.fft_ConjugateConvolve(self.discritization[j], self.discritization[j+i])))
            
    
w = CrossCorrelationWavelet('db1', 3, 3)

















