#include <cstdlib>
#include <iostream>
#include <complex>
#include <vector>
// #include <omp.h>
#include <math.h>

using namespace std;

float get_kval(float dr, float krad, int grid_mod, float *kernel);
void grid2D(complex<float> *data, float *k_r, float *k_i, float *traj, float *dens,
            int N, int Nx, int Ny, float krad, int grid_mod, float *kernel);
void igrid2D(complex<float> *data, complex<float> *k_data, float *traj, float *dens,
            int N, int Nx, int Ny, float krad, int grid_mod, float *kernel);


float get_kval(float dr, float krad, int grid_mod, float *kernel)
{
    float i = dr / krad * grid_mod;
    int ri = floor(i);
    float di = i - ri;

    return (kernel[ri] * (1-di) + kernel[ri+1] * di);
}


void igrid2D(complex<float> *data, complex<float> *k_data, float *traj, float *dens,
            int N, int Nx, int Ny, float krad, int grid_mod, float *kernel)
{
    // omp_set_num_threads(16);
    // #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float x = (traj[i*2] + 0.5) * Nx;
        float y = (traj[i*2 + 1] + 0.5) * Ny;
        int xmin = floor(x - krad);
        int ymin = floor(y - krad);
        int xmax = ceil(x + krad);
        int ymax = ceil(y + krad);
        for (int iy = ymin; iy <= ymax; iy++)
        {
            if ( (iy >= 0) && (iy < Ny) )
            {
                float dy = abs(y - iy);
                float kvy = get_kval(dy, krad, grid_mod, kernel);
                for (int ix = xmin; ix <= xmax; ix++)
                {
                    if ( (ix >= 0) && (ix < Nx) )
                    {
                        float dx = abs(x - ix);
                        float kvx = get_kval(dx, krad, grid_mod, kernel);

                        data[i] += k_data[iy*Nx + ix] * kvx * kvy;

                    }
                }
            }
        }
    }
    return;
}

void grid2D(complex<float> *data, float *k_r, float *k_i, float *traj, float *dens,
            int N, int Nx, int Ny, float krad, int grid_mod, float *kernel)
{
    // omp_set_num_threads(16);
    // #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float x = (traj[i*2] + 0.5) * Nx;
        float y = (traj[i*2 + 1] + 0.5) * Ny;
        int xmin = floor(x - krad);
        int ymin = floor(y - krad);
        int xmax = ceil(x + krad);
        int ymax = ceil(y + krad);
        for (int iy = ymin; iy <= ymax; iy++)
        {
            if ( (iy >= 0) && (iy < Ny) )
            {
                float dy = abs(y - iy);
                float kvy = get_kval(dy, krad, grid_mod, kernel);
                for (int ix = xmin; ix <= xmax; ix++)
                {
                    if ( (ix >= 0) && (ix < Nx) )
                    {
                        float dx = abs(x - ix);
                        float kvx = get_kval(dx, krad, grid_mod, kernel);

                        float dat_r  = data[i].real() * dens[i] * kvx * kvy;
                        float dat_i  = data[i].imag() * dens[i] * kvx * kvy;

                        // #pragma omp atomic
                        k_r[iy*Nx + ix] += dat_r;
                        // #pragma omp atomic
                        k_i[iy*Nx + ix] += dat_i;


                    }
                }
            }
        }
    }
    return;
}
