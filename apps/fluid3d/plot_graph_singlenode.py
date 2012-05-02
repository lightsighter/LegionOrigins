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

def plot(cpus, speedups, label, mark_index, pp):
    plt.plot(cpus, speedups, label = label, linestyle = 'dashed', markersize = 7, marker = markers[mark_index], linewidth=0.5)

baseline = {
    'Viz': 4.525,
    'Sapling': 3.442,
    }

small_plot = [('Viz', 'PARSEC pthreads',
               [(1, 5.03), (2, 2.684), (4, 1.447), (8, 0.875), (16, 0.901)]),
              ('Viz', 'Legion',
               [(1, 4.14), (2, 2.244), (4, 1.239), (8, 0.823), (10, 0.734), (12, 0.816), (2*10, 0.684)]),
              ('Sapling', 'PARSEC pthreads',
               [(1, 3.928), (2, 2.098), (4, 1.094), (8, 0.751), (16, 0.585)]),
              ('Sapling', 'Legion',
               [(1, 3.731), (2, 2.046), (4, 1.133), (8, 0.897), (12, 0.788), (2*10, 0.653)]),
]

if __name__ == '__main__':
    pp = PdfPages('singlenode_parsec_vs_legion.pdf')
    plot_init('Legion vs PARSEC at 300K Particles', 'CPUs', 'Relative Speedup (vs PARSEC serial)', pp)
    index = 1
    for machine, framework, data in small_plot:
        cpus, times = zip(*data)
        plot(cpus, baseline[machine] / np.array(times),
             '%s %s' % (machine, framework), index, pp)
        index += 1
    plt.legend(loc=4)
    plt.axis([0, 18, 0, 8])
    pp.savefig()
    pp.close()
