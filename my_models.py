import numpy as np
import collections
from numba import cuda
from numba import guvectorize, vectorize, float32, float64, int32

#@vectorize([float64(float64, int32)])
@guvectorize(["(float64[:,:],int32[:], float64[:,:])"],"(n, m),(n)->(n, m)")
def GPU_get_freq(freq, apriori, result):
    result = (freq.transpose()/apriori).transpose()

@guvectorize(["(float64[:,:],int32[:], float64[:,:])"],"(n, m),(m)->(n, m)")
def mat_div(freq, apriori, result):
    result = freq / apriori

class NaiveBayes:
    def __init__(self, gpu = True):
        self.apriori = None
        self.freq = None
        if gpu:
            if len(cuda.gpus) > 0:
                pass
            else:
                raise Exception("You have no GPUs available.")
        self.gpu = gpu



    def fit(self, X: np.array, y: list):
        if self.apriori is None and self.freq is None:
            self.apriori = np.array(list(dict(sorted(collections.Counter(y).items())).values()))
            self.freq = np.zeros(shape=(len(self.apriori), len(X[0])))

        for sample, class_num in zip(X, y):
            self.freq[class_num] += sample

        if self.gpu:
            #dev_freq = cuda.to_device(np.transpose(self.freq))
            dev_freq = cuda.to_device(self.freq)
            dev_apriori = cuda.to_device(self.apriori)
            result = cuda.device_array(shape = self.freq.shape)
            #GPU_get_greq(dev_freq, dev_apriori, result)
            #result = mat_div(dev_freq, dev_apriori)
            #result GPU_get_freq
            #self.freq = np.transpose(result.copy_to_host())
            self.freq = GPU_get_freq(dev_freq, dev_apriori)
            self.apriori = self.apriori / self.apriori.sum()
        else:
            self.freq = np.transpose(np.transpose(self.freq) / self.apriori)
            self.apriori = self.apriori / self.apriori.sum()

    def predict(self, X: np.array):
        result = np.zeros(shape = (len(X)))
        for i in range(len(X)):
            result[i] = ((self.freq * X[i]).sum(axis = 1) * self.apriori).argmax()
        return result

