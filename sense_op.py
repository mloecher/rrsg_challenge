import h5py
import numpy as np
from scipy import signal
from pygrid.pygrid import Gridder

class SenseOp:

    def __init__(self, filename, use_hamming = True, verbose = 0, fov_mod = (1,1), shift = (0,0), dsize = None):
        ff = h5py.File(filename, 'r') 

        self.data = np.array(ff['rawdata'])[0]
        self.traj = np.array(ff['trajectory'])[[1,0]]
        
        self.Nx = int(np.round(self.traj[0].max() - self.traj[0].min()))
        self.Ny = int(np.round(self.traj[1].max() - self.traj[1].min()))
        self.traj[0] /= self.Nx
        self.traj[1] /= self.Ny
        self.Nx *= fov_mod[0]
        self.Ny *= fov_mod[1]

        if dsize is not None:
            self.Nx = dsize[0]
            self.Ny = dsize[1]

        self.data = np.transpose(self.data, (2, 1, 0))
        self.traj = np.transpose(self.traj, (2, 1, 0))
        
        self.data *= np.exp(1j * (self.traj[np.newaxis, :, :, 0]) * shift[0])
        self.data *= np.exp(1j * (self.traj[np.newaxis, :, :, 1]) * shift[1])

        self.Ncoils = self.data.shape[0]

        self.kr = np.sqrt((self.traj**2.0).sum(-1))

        self.dens = self.kr
        if use_hamming:
            filt = signal.hamming(self.data.shape[-1], True)
            self.data *= filt[np.newaxis, np.newaxis, :]

        self.verbose = verbose
        if self.verbose:
            print('Nx = %d  Ny = %d  Ncoils = %d' % (self.Nx, self.Ny, self.Ncoils))

        self.data0 = self.data.copy()
        self.traj0 = self.traj.copy()
        self.dens0 = self.dens.copy()

    def retro_undersample(self, R, successive = False):
        self.undersample = R
        if successive:
            nproj = self.data0.shape[1]
            self.data = self.data0[:, :int(nproj//R)]
            self.traj = self.traj0[:int(nproj//R)]
            self.dens = self.dens0[:int(nproj//R)]
        else:
            ind = np.arange(0,self.traj0.shape[0],self.undersample)
            ind = np.round(ind).astype(np.int)
            self.data = self.data0[:, ind]
            self.traj = self.traj0[ind]
            self.dens = self.dens0[ind]

    def simple_sos(self):
        gg = Gridder(imsize=(self.Nx, self.Ny))
        gg.traj = self.traj
        gg.dens = self.dens

        im_sos = np.zeros((self.Nx, self.Ny))

        for ic in range(self.Ncoils):

            data_c = self.data[ic]
            im = gg.grid_proc2(data_c)
            im_sos += np.abs(im) ** 2.0

        im_sos = np.sqrt(im_sos)

        return im_sos

    def single_coil(self, coil_num = 0):
        gg = Gridder(imsize=(self.Nx, self.Ny))
        gg.traj = self.traj
        gg.dens = self.dens


        data_c = self.data[coil_num]
        return gg.grid_proc2(data_c)


    def gen_coilmaps(self, filt_sigma = 0.05):
        gg = Gridder(imsize=(self.Nx, self.Ny))
        gg.traj = self.traj
        gg.dens = self.dens

        all_im = []

        for ic in range(self.Ncoils):

            data_c = self.data[ic]
            all_im.append(gg.grid_proc2(data_c, filter_sigma = filt_sigma))

        all_im = np.array(all_im)

        im_sos = np.sqrt((np.abs(all_im)**2.0).sum(0))
        self.coilmaps = all_im/im_sos[np.newaxis,...]

    def base_coilrecon(self):
        gg = Gridder(imsize=(self.Nx, self.Ny))
        gg.traj = self.traj
        gg.dens = self.dens

        all_im = []

        for ic in range(self.Ncoils):

            data_c = self.data[ic]
            all_im.append(gg.grid_proc2(data_c))

        all_im = np.array(all_im)

        im = (all_im * np.conj(self.coilmaps)).sum(0)
        return im

    def Eh(self, data):
        gg = Gridder(imsize=(self.Nx, self.Ny))
        gg.traj = self.traj
        gg.dens = self.dens

        all_im = []
        for ic in range(self.Ncoils):
            all_im.append(gg.grid_proc2(data[ic]))
        all_im = np.array(all_im)

        im = (all_im * np.conj(self.coilmaps)).sum(0)
        return im

    def E(self, im):
        gg = Gridder(imsize=(self.Nx, self.Ny))
        gg.traj = self.traj
        gg.dens = self.dens

        all_im = im[np.newaxis, ...] * self.coilmaps
        all_data = []
        for ic in range(self.Ncoils):
            all_data.append(gg.igrid_proc2(all_im[ic]))
        all_data = np.array(all_data)

        return all_data

    def SENSE(self, n_iter = 10, truth = None, calc_error = False):
        
        n_print = min((20, n_iter))

        data = self.data
        data = np.reshape(data, (data.shape[0], -1))

        all_err = []
        all_d = []
        a = self.Eh(data)

        b0 = np.zeros_like(a)
        p = a.copy()
        r0 = p.copy()

        if calc_error:
            error = get_error(truth, b0)
            all_err.append(error)
            all_d.append(1.0)

        print('Iteration (n_iter %d) = ' % (n_iter,), end='', flush = True)
        for i in range(n_iter):
            
            if (i % (n_iter//n_print)) == 0:
                print('%d, ' % i, end='', flush = True)

            d = np.abs((r0.conj()*r0).sum()/(a.conj()*a).sum())
            # print('iter = %d  del = %.4f  error = %.4f' % (i, d, error))
            
            q = self.Eh(self.E(p))
            b1 = b0 + np.abs((r0.conj()*r0).sum()/(p.conj()*q).sum()) * p
            r1 = r0 - np.abs((r0.conj()*r0).sum()/(p.conj()*q).sum()) * q
            p = r1 + np.abs((r1.conj()*r1).sum()/(r0.conj()*r0).sum()) * p    
            
            r0 = r1
            b0 = b1
            
            if calc_error:
                error = get_error(truth, b0)
                all_err.append(error)
                all_d.append(d)

        print('Done!', flush = True)
        return b0, all_err, all_d

def get_error(truth, im):
    if np.linalg.norm(im) < 1.0e-12:
        return 1.0
    else:
        scale = np.linalg.norm(truth)/np.linalg.norm(im)
        return np.linalg.norm(scale*np.abs(im)-np.abs(truth))/np.linalg.norm(truth)
    