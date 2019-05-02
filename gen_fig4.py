from sense_op import SenseOp

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

senseop = SenseOp('./data/rawdata_brain_radial_96proj_12ch.h5', use_hamming=True, shift = (0,150))

senseop.gen_coilmaps()
truth = senseop.base_coilrecon()

res = []

for R in [4, 3, 2]:
    print('-- Figure 4: running SENSE for R = %d' % R)
    senseop.retro_undersample(R, False)
    im, all_err, all_d = senseop.SENSE(80, truth = truth, calc_error = True)
    res.append((R, im, all_err, all_d))


sns.set_context("poster")

f, axarr = plt.subplots(2, 1, squeeze=False, figsize = (6,12))

axarr[0,0].plot(res[0][2], label='R=%d' % res[0][0])
axarr[0,0].plot(res[1][2], label='R=%d' % res[1][0], linestyle=':')
axarr[0,0].plot(res[2][2], label='R=%d' % res[2][0], linestyle='-.')
axarr[0,0].set_yscale('log')
axarr[0,0].set_ylabel(r"$\Delta$")
axarr[0,0].set_xlabel(r"Iteration")
axarr[0,0].legend()

axarr[1,0].plot(res[0][3], label='R=%d' % res[0][0])
axarr[1,0].plot(res[1][3], label='R=%d' % res[1][0], linestyle=':')
axarr[1,0].plot(res[2][3], label='R=%d' % res[2][0], linestyle='-.')
axarr[1,0].set_yscale('log')
axarr[1,0].set_ylabel(r"$\delta$")
axarr[1,0].set_xlabel(r"Iteration")
axarr[1,0].legend()

sns.despine()
plt.tight_layout()

plt.savefig('figs/fig4.png',bbox_inches='tight',dpi=300)