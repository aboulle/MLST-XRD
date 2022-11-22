import numpy as np

class Normalize():
    def __init__(self, y, norm="MinMax"):
        self.ymin = y.min(axis=0)
        self.ymax = y.max(axis=0)
        self.ymean = y.mean(axis=0)
        self.ystd = y.std(axis=0)
        self.norm = norm
    
    def forward(self, x):
        print("Normalization scheme:", self.norm)
        if self.norm == "MinMax":
            x -= self.ymin
            x /= (self.ymax-self.ymin)
        if self.norm == "Gauss":
            x -= self.ymean
            x /= self.ystd
        return x
    
    def backwards(self, x):
        print("Normalization scheme:", self.norm)
        if self.norm == "MinMax":
            x *= (self.ymax - self.ymin)
            x += self.ymin
        if self.norm == "Gauss":
            x *= self.ystd
            x += self.ymean
        return x

def cut_data(x, thresh):
    x[x < thresh] = thresh
    x = (x - thresh) / (x.max() - thresh)
    return x

def rescale_i(data):
    norm_i=data.min(axis=1)[:, np.newaxis, :,:]
    data -= norm_i
#     data /= np.abs(norm_i)
    return data