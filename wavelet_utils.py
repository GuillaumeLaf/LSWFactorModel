import numpy as np
import matplotlib.pyplot as plt
import numba as nb


def np_convolve(in1:np.ndarray, in2:np.ndarray):    
    return np.convolve(in1, in2)
    
def fft_convolve(in1:np.ndarray, in2:np.ndarray):
    in2 = adjustSecondArraySizeToFirst(in1, in2)
    
    in1_fft = np.fft.fft(in1)
    in2_fft = np.fft.fft(in2)
    convolution = in1_fft * in2_fft
    convolution = np.fft.ifft(convolution)
    return np.real(convolution)

def fft_ConjugateConvolve(in1:np.ndarray, in2:np.ndarray):
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

@nb.njit()
def upsample(arr):
    upsampledArray = np.zeros((arr.size*2,), dtype=np.float64)
    upsampledArray[0::2] = np.copy(arr)
    return upsampledArray   

@nb.njit(nogil=True)
def rollMatrixByRow(matrix:np.ndarray):
    for i in range(matrix.shape[0]):
        matrix[i, :] = np.roll(matrix[i, :], -i)
    return matrix

@nb.njit(nogil=True)
def getScalingSpectrumAtTimeZ(scalingMatrix:np.ndarray):
    scalingMatrix = np.triu(scalingMatrix)
    scalingMatrix = rollMatrixByRow(scalingMatrix)
    return scalingMatrix

@nb.njit(nogil=True, fastmath=True)
def initializeA_operatorOfOrders(crossCorrelationScaleI:np.ndarray, crossCorrelationScaleR:np.ndarray, maxScale:int):
    out = np.zeros((maxScale, maxScale), dtype=np.float64)
    for j in range(maxScale):
        for l in range(maxScale):
            out[j, l] = crossCorrelationScaleI[j] @ crossCorrelationScaleR[l]
    return out



