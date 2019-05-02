import numpy as np

class GridKernel():

    def __init__(self, grid_params):
        
        self.krad = grid_params['krad']
        self.grid_mod = grid_params['grid_mod']
        self.calc_kernel_kb(grid_params)
        self.fourier_demod(grid_params)

    def get_kval(self, dr):
        if dr >= self.krad:
            return 0.0
        else:
            i = 0
            while self.kx[i] <= dr:
                i += 1
            dx = self.kx[i] - self.kx[i - 1]
            ddr = dr - self.kx[i - 1]
            y = self.ky[i - 1] * (1 - ddr / dx) + self.ky[i] * ddr / dx
            return y

    def get_kval_vec(self, dr):
        i = dr / self.krad * self.grid_mod
        ri = np.floor(i).astype('int32')
        di = i - ri
        y = self.ky[ri] * (1-di) + self.ky[ri+1] * di
        return y


    def calc_kernel_kb(self, grid_params):

        kw0 = 2.0 * grid_params['krad'] / grid_params['over_samp']
        kr = grid_params['krad']

        beta = np.pi * \
            np.sqrt((kw0 * (grid_params['over_samp'] - 0.5)) ** 2 - 0.8)

        x = np.linspace(0, kr, grid_params['grid_mod'])
        x_bess = np.sqrt(1 - (x / kr) ** 2)

        y = np.i0(beta * x_bess)
        y = y / y[0]

        x = np.concatenate((x, np.zeros(grid_params['grid_mod'])))
        y = np.concatenate((y, np.zeros(grid_params['grid_mod'])))

        self.kx = x
        self.ky = y

    def calc_kernel_tri(self, grid_params):

        kr = grid_params['krad']

        x = np.linspace(0, kr, grid_params['grid_mod'])

        y = 1.0 - x / kr

        self.kx = x
        self.ky = y

    def calc_kernel_ones(self, grid_params):

        kr = grid_params['krad']

        x = np.linspace(0, kr, grid_params['grid_mod'])

        y = np.ones(x.size)

        self.kx = x
        self.ky = y

    def fourier_demod(self, grid_params):
        self.Dy = []

        for i in range(len(grid_params['imsize_os'])):
            xres = grid_params['imsize_os'][i]
            Dx = np.arange(xres)
            Dx = Dx - xres / 2.0
            Dy = np.zeros(Dx.size, 'complex128')

            for i in range(1, self.kx.size):
                temp = self.ky[i] * 2 * \
                    np.exp(2 * 1j * np.pi * Dx / xres * self.kx[i] )
                Dy += temp

            Dy = Dy.real
            Dy = Dy + self.ky[0]
            Dy = Dy / self.kx.size

            self.Dy.append(Dy)