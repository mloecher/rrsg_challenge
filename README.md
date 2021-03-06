# rrsg_challenge

This is an implementation for the ISMRM 2019 reproducibility challenge. [LINK](https://blog.ismrm.org/2019/04/02/ismrm-reproducible-research-study-group-2019-reproduce-a-seminal-paper-initiative/) 

## Installation

This code is mostly written in Python.  If you have Anaconda installed, you should have all necessary packages, but if you don't, then you will need Numpy, Scipy, Matplotlib, Seaborn, and Cython.  

The gridding function is written in C, and wrapped with Cython.  It needs to be built by running:

```
python setup.py build_ext --inplace
```

in the root directory.  This will build the gridding function in the current directory (it will not install anything else outside of this folder).  A Windows+Python 3.7 binary is already compiled, other compilations on Windows may need the Visual Studio build tools from the middle of [THIS](https://visualstudio.microsoft.com/downloads/) page.  Mac and Linux should work just fine.

## Usage

The three figures we are reprocuding can be run with:
```
python gen_fig4.py
```
and the equivalent for fig5 and fig6.  The figures will be output into the figs/ folder.

Additionally all of this code is in a jupyter notebook call "all_figs.ipynb"

## Implementation Notes

This is a relatively straightforward implementation, trying to match the paper as well as I could.

- Coil maps are generated by dividing coil images by the sum of squares image, both of which are heavily blurred with a Gaussian window to get low res coil maps.  The coil maps are generated from the fully sampled data.

- All of the data is filtered with a Hamming window

- The density compensation being used is simply the k-space radius

- Gridding is performed with a pretty standrad Kaiser-Bessel kernel

- No intensity correction is being used

- A filter is used to get rid of k-space corners (equation (30) in the manuscript)

- The CG algorithm is hopefully exactly as written in the manuscript


