#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_init(title, xlabel, ylabel, pp):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

markers = ['o','s','D','*','v','^','p','<','>','d']

def plot(cpus, speedups, label, mark_index):
    plt.semilogx(cpus, speedups, basex = 2, label = label, linestyle = 'dashed', markersize = 7, marker = markers[mark_index], linewidth=0.5)

machine = 'Keeneland'
balance_plot_cpus = [1*1, 1*2, 1*4, 1*8, 2*8, 4*8, 8*8, 16*8]
balance_plot = [('XZ-slices', [326.847, 168.896, 86.627, 47.429, 25.086, 14.82, 11.327, 12.47]),
                ('XYZ-slices static', [326.847, 168.896, 86.627, 77.81, 40.961, 22.52, 14.807, 11.98]),
                ('XYZ-slices balanced', [326.847, 168.896, 86.627, 50.751, 26.818, 15.785, 11.196, 10.68]),
]
baseline = np.array(balance_plot[0][1])

if __name__ == '__main__':
    fig = plt.figure(figsize = (10,4))
    plt.xlabel("Grids")
    plt.ylabel("Relative Speedup (in %)")
    plt.grid(True)
    index = 1
    for divs, data in balance_plot:
        plot(balance_plot_cpus, ((baseline/np.array(data)) - 1.0)*100.0, divs, index)
        index += 1
    plt.legend(loc=4)
    ax = plt.axis([1, 32*8, -50, 20])
    #plt.axes().xaxis.set_ticks([1,2,4,8,16])
    plt.axes().xaxis.set_ticklabels([1,2,4,8,16,32,64,128])
    plt.savefig("figs/fluid_balance.pdf", format="pdf", bbox_inches="tight")
