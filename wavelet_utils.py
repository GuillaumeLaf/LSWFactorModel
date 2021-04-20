import numpy as np
import matplotlib.pyplot as plt
import numba as nb

"""
    This file contains a handful of functions that are used throughout this repository.
"""

def np_convolve(in1:np.ndarray, in2:np.ndarray):        
    return np.convolve(in1, in2)
    
def fft_convolve(in1:np.ndarray, in2:np.ndarray):
    """
    Compute the convolution of two arrays with the Fast Fourier Transform.

    Parameters
    ----------
    in1 : np.ndarray
        First array.
    in2 : np.ndarray
        Second array. This array is supposed to be a filter

    Returns
    -------
    TYPE
        Convolution of 'in1' and 'in2'.

    """
    
    # This function allows to also convolve a filter 'in2' with an array 'in1'
    in2 = adjustSecondArraySizeToFirst(in1, in2)
    
    in1_fft = np.fft.fft(in1)
    in2_fft = np.fft.fft(in2)
    convolution = in1_fft * in2_fft
    convolution = np.fft.ifft(convolution)
    return np.real(convolution)

def fft_ConjugateConvolve(in1:np.ndarray, in2:np.ndarray):
    """
    Compute the cross-correlation of two array with the Fast Fourier Transform.
    Unlike the convolution, we take the conjugate of the filter 'in2'. 

    Parameters
    ----------
    in1 : np.ndarray
        First array.
    in2 : np.ndarray
        Second array. This array is supposed to be a filter.

    Returns
    -------
    TYPE
        Cross-correlation between 'in1' and 'in2'.

    """
    
    in2 = adjustSecondArraySizeToFirst(in1, in2)
    
    in1_fft = np.fft.fft(in1)
    in2_fft = np.fft.fft(in2)
    convolution = in1_fft * np.conjugate(in2_fft)
    convolution = np.fft.ifft(convolution)
    return np.real(convolution)

def adjustSecondArraySizeToFirst(in1:np.ndarray, in2:np.ndarray):
    if in1.size > in2.size:
        in2 = np.append(in2, np.repeat(0, in1.size-in2.size))
    return in2

def adjustFirstArraySizeToSecond(in1:np.ndarray, in2:np.ndarray):
    if in1.size < in2.size:
        in1 = np.append(in1, np.repeat(0, in2.size-in1.size))
    return in2

def isArrayInitialized(obj, name):
    return hasattr(obj, name)


"""
    The following functions could have been placed inside their corresponding classes.
    However, in order to speed up the computation using Numba, we have to extract them from the class environment.
    The 'nb.njit' decorator allows to get speed of the function comparable to C.
"""

@nb.njit()
def upsample(arr):
    """
    Upsample the given array by placing zeros between each elements.

    Parameters
    ----------
    arr : TYPE
        Array that needs to be upsampled.

    Returns
    -------
    upsampledArray : np.ndarray

    """
    
    upsampledArray = np.zeros((arr.size*2,), dtype=np.float64)
    upsampledArray[0::2] = np.copy(arr)
    return upsampledArray   

@nb.njit(nogil=True)
def rollMatrixByRow(matrix:np.ndarray):
    """
    This function allows to roll an entire matrix by row.

    Parameters
    ----------
    matrix : np.ndarray
        DESCRIPTION.

    Returns
    -------
    matrix : np.ndarray

    """
    
    for i in range(matrix.shape[0]):
        matrix[i, :] = np.roll(matrix[i, :], -i)
    return matrix    

@nb.njit(nogil=True, fastmath=True)
def initializeA_operatorOfOrders(crossCorrelationScaleI:np.ndarray, crossCorrelationScaleR:np.ndarray, maxScale:int):
    """
    Initialize the correction matrix 'A' between scales "i" and "r".

    Parameters
    ----------
    crossCorrelationScaleI : np.ndarray
        Cross correlation array at scale 'i'.
    crossCorrelationScaleR : np.ndarray
        Cross correlation array at scale 'r'.
    maxScale : int
        maximum scale.

    Returns
    -------
    out : TYPE
        correction matrix 'A.

    """
    
    out = np.zeros((maxScale, maxScale), dtype=np.float64)
    for j in range(maxScale):
        for l in range(maxScale):
            out[j, l] = crossCorrelationScaleI[j] @ crossCorrelationScaleR[l]
    return out



