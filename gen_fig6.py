from sense_op import SenseOp

import matplotlib.pyplot as plt
import numpy as np

senseop = SenseOp('./data/rawdata_heart_radial_55proj_34ch.h5', use_hamming=True)
senseop.gen_coilmaps()

res = []

for R in [1,1.6666,2.5,5]:
    print('-- Figure 6: running SENSE for R = %d' % R)
    senseop.retro_undersample(R, True)
    im = senseop.SENSE(8)[0]
    imc = senseop.single_coil()
    im0 = senseop.base_coilrecon()
    res.append((R, imc, im0, im))


f, axarr = plt.subplots(4, 3, squeeze=False, figsize = (10,10))

axarr[0,0].set_title('Single Coil', fontsize=20)
axarr[0,1].set_title('Initial', fontsize=20)
axarr[0,2].set_title('Final', fontsize=20)

for ii in range(4):
    axarr[ii,0].imshow(np.abs(res[ii][1]), interpolation='nearest', origin='lower', cmap='gray')
    axarr[ii,0].axis('off')
    axarr[ii,0].text(-180,110,'R = %.2f' % res[ii][0], rotation=0, fontsize=20)

    axarr[ii,1].imshow(np.abs(res[ii][2]), interpolation='nearest', origin='lower', cmap='gray')
    axarr[ii,1].axis('off')

    axarr[ii,2].imshow(np.abs(res[ii][3]), interpolation='nearest', origin='lower', cmap='gray')
    axarr[ii,2].axis('off')

plt.tight_layout()
plt.savefig('figs/fig6.png',bbox_inches='tight',dpi=300)