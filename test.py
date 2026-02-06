from support_functions.gen_spc_uncert_map_2d import gen_uncertainty_map, gen_uncertainty_map2

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm


if __name__ == '__main__':
    x, y, A, B = gen_uncertainty_map2(4, 30)

    x1, y1, A1, B1 = gen_uncertainty_map(4, 30)

    X, Y   = np.meshgrid(x, y, indexing='ij')
    X1, Y1 = np.meshgrid(x1, y1, indexing='ij')

    fig, ax = plt.subplots(1,2, figsize=(12,6), layout='constrained')
    ax[0].pcolormesh(X, Y, 100*A/Y, norm=LogNorm(vmin=1, vmax=100))
    ax[1].pcolormesh(X1, Y1, 100*A1/Y1, norm=LogNorm(vmin=1, vmax=100))
    ax[1].set_xlim([0,90])

    [ax[i].set_yscale('log') for i in range(2)]

    plt.show()

    fig, ax = plt.subplots(1,2, figsize=(12,6), layout='constrained')
    ax[0].pcolormesh(X, Y, 100*B/Y, norm=LogNorm(vmin=1, vmax=100))
    ax[1].pcolormesh(X1, Y1, 100*B1/Y1, norm=LogNorm(vmin=1, vmax=100))
    ax[1].set_xlim([0,90])

    [ax[i].set_yscale('log') for i in range(2)]

    plt.show()