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
    plt.plot(cpus, speedups, label = label, linestyle = 'dashed', markersize = 7, marker = markers[mark_index], linewidth=0.5)

baseline = {
    'Viz': 4.466,
    'Sapling': 3.442,
    }

small_plot = [('Viz', 'PARSEC pthreads',
               [(1, 4.999), (2, 2.634), (4, 1.425), (8, 0.851), (16, 0.829)]),
              ('Viz', 'Legion',
               [(1, 4.14), (2, 2.244), (4, 1.239), (8, 0.823), (10, 0.734), (12, 0.816), (2*10, 0.684)]),
              ('Sapling', 'PARSEC pthreads',
               [(1, 3.928), (2, 2.098), (4, 1.094), (8, 0.751), (16, 0.585)]),
              ('Sapling', 'Legion',
               [(1, 3.731), (2, 2.046), (4, 1.133), (8, 0.897), (12, 0.788), (2*10, 0.653)]),
]

if __name__ == '__main__':
    fig = plt.figure(figsize = (10,7))
    plt.xlabel("Threads")
    plt.ylabel("Relative Speedup (vs PARSEC serial)")
    plt.grid(True)
    index = 1
    for machine, framework, data in small_plot:
        cpus, times = zip(*data)
        plot(cpus, baseline[machine] / np.array(times),
             '%s %s' % (machine, framework), index)
        index += 1
    plt.legend(loc=4)
    plt.axis([0, 18, 0, 7])
    plt.savefig("figs/fluid_singlenode.pdf", format="pdf", bbox_inches="tight")
