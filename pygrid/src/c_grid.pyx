# cython: language_level=2, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
cimport cython


cdef extern from "grid_op.cpp":

    void _grid2D "grid2D"(float complex *data, float *k_r, float *k_i,
                          float *traj, float *dens, int N, int Nx, int Ny, 
                          float krad, int grid_mod, float *kernel)
    void _igrid2D "igrid2D"(float complex *data, float complex *k_data, float *traj, float *dens,
                            int N, int Nx, int Ny, float krad,
                            int grid_mod, float *kernel)


def c_grid2d(data, grid_params, kernel, traj, dens):
    ksize = grid_params['imsize_os']
   
    k_r = np.zeros(ksize, np.float32)
    k_i = np.zeros(ksize, np.float32)

    cdef float[::1] k_r_view = array_prep(k_r, np.float32)
    cdef float[::1] k_i_view = array_prep(k_i, np.float32)

    cdef float complex[::1] data_view = array_prep(data, np.complex64)

    cdef float[::1] traj_view = array_prep(traj, np.float32)
    cdef float[::1] dens_view = array_prep(dens, np.float32)

    N = data.size
    Nx = grid_params['imsize_os'][1] 
    Ny = grid_params['imsize_os'][0]
    cdef float krad = grid_params['krad']
    cdef int grid_mod = grid_params['grid_mod']

    cdef float[::1] kernel_view = array_prep(kernel.ky, np.float32)

    _grid2D(&data_view[0], &k_r_view[0],&k_i_view[0], &traj_view[0], &dens_view[0], 
           N, Nx, Ny, krad, grid_mod, &kernel_view[0])

    return k_r + 1j*k_i

def c_igrid2d(kspace, grid_params, kernel, traj, dens):
    
    cdef float complex[::1] kspace_view = array_prep(kspace, np.complex64)

    N = dens.size
    data = np.zeros(N, np.complex64)
    cdef float complex[::1] data_view = array_prep(data, np.complex64)

    cdef float[::1] traj_view = array_prep(traj, np.float32)
    cdef float[::1] dens_view = array_prep(dens, np.float32)

    Nx = grid_params['imsize_os'][1] 
    Ny = grid_params['imsize_os'][0]
    cdef float krad = grid_params['krad']
    cdef int grid_mod = grid_params['grid_mod']

    cdef float[::1] kernel_view = array_prep(kernel.ky, np.float32)

    _igrid2D(&data_view[0], &kspace_view[0], &traj_view[0], &dens_view[0], 
           N, Nx, Ny, krad, grid_mod, &kernel_view[0])

    return data


def array_prep(A, dtype, linear=True):
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    
    A = A.astype(dtype, order='C', copy=False)
    
    if linear:
        A = A.ravel()

    return A
