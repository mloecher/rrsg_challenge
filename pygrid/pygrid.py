""" This is a 2D version of a bigger gridding package I have previously made """

import numpy as np
from .grid_kernel import GridKernel
from . import c_grid
from scipy import signal, interpolate
import matplotlib.pyplot as plt

class Gridder:

    def __init__(self, imsize = [256, 256], kernel_type = 'kb', krad = 2.5, grid_mod = 64,
                 over_samp = 1.5):
        
        self.imsize = np.array(imsize, 'int32')
        self.imsize_os = self.imsize.astype('float64') * over_samp
        self.imsize_os = np.ceil(self.imsize_os).astype('int32')

        self.grid_params = {}
        self.grid_params['kernel_type'] = kernel_type
        self.grid_params['krad'] = krad
        self.grid_params['grid_mod'] = grid_mod
        self.grid_params['over_samp'] = over_samp
        self.grid_params['imsize_os'] = self.imsize_os

        self.kernel = GridKernel(self.grid_params)

    def grid2(self, data):
        kspace = c_grid.c_grid2d(data, self.grid_params, self.kernel, self.traj, self.dens)
        return kspace

    def igrid_proc2(self, im):
        im = zeropad_ratio(im, self.grid_params['over_samp'])

        im /= self.kernel.Dy[1][np.newaxis, :]
        im /= self.kernel.Dy[0][:, np.newaxis]

        kk = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(im)))

        data = c_grid.c_igrid2d(kk, self.grid_params, self.kernel, self.traj, self.dens)

        return data

    def grid_proc2(self, data, filter_corners = True, filter_sigma = 0):
        
        if filter_sigma > 0:
            kr = np.sqrt((self.traj**2.0).sum(-1))
            filt = np.exp(-(kr**2.0)/(filter_sigma**2.0))
            kk = c_grid.c_grid2d(data*filt[np.newaxis,...], self.grid_params, self.kernel, self.traj, self.dens)
        else:            
            kk = c_grid.c_grid2d(data, self.grid_params, self.kernel, self.traj, self.dens)

        if filter_corners:
            yy, xx = np.meshgrid(np.linspace(-1,1,kk.shape[0],False), np.linspace(-1,1,kk.shape[1],False), indexing ='ij')
            rr = np.sqrt(xx*xx + yy*yy)

            Nf = 2000
            beta = 100.0
            cutoff = 1.0

            filt = 0.5 + 1.0/np.pi * np.arctan(beta * (cutoff - rr.ravel()) / cutoff)
            filt = np.reshape(filt, rr.shape)
            kk *= filt

        im = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kk)))
        im /= self.kernel.Dy[1][np.newaxis, :]
        im /= self.kernel.Dy[0][:, np.newaxis]

        im = crop_ratio(im, self.grid_params['over_samp'])
        
        return im


def crop_ratio(im, ratio):
    shape_big = im.shape
    shape_small = tuple([int(x / ratio) for x in shape_big])

    out = np.zeros(shape_small)

    d0_start = shape_big[0]//2 - shape_small[0]//2
    d0_stop = shape_big[0]//2 + shape_small[0]//2
    d1_start = shape_big[1]//2 - shape_small[1]//2
    d1_stop = shape_big[1]//2 + shape_small[1]//2

    out = im[d0_start:d0_stop, d1_start:d1_stop]
    return out

def zeropad_ratio(im, ratio):
    shape0 = im.shape
    shape1 = tuple([int(x * ratio) for x in shape0])

    out = np.zeros(shape1, np.complex64)

    d0_start = shape1[0]//2 - shape0[0]//2
    d0_stop = shape1[0]//2 + shape0[0]//2
    d1_start = shape1[1]//2 - shape0[1]//2
    d1_stop = shape1[1]//2 + shape0[1]//2

    out[d0_start:d0_stop, d1_start:d1_stop] = im
    return out