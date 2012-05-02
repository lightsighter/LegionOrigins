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

total_particles = {'300K': 305809, '2400K': 2446472, '19M': 19554392}
total_cells = {'300K': 135424, '2400K': 1115721, '19M': 9056971}
total_steps = 10

large_plot_nodes = [1, 2, 4, 8, 16]
large_plot = [('Keeneland',
               [('19M', np.array([46.377, 21.444, 13.483, 10.891, 10.68])),
                ('2400K', np.array([5.169, 3.206, 2.673, 2.764, 3.801])),
                ('300K', np.array([0.912, 0.847, 0.836, 1.158, 2.415]))]),
              ('Viz',
               [('19M', np.array([43.058, 24.962, 17.059, 12.021])),
                ('2400K', np.array([5.449, 3.331, 2.691, 2.519])),
                ('300K', np.array([0.734, 0.684, 0.873, 1.122]))]),
              ('Sapling',
               [('19M', np.array([39.89, 21.859, 13.323])),
                ('2400K', np.array([4.957, 2.896, 2.413])),
                ('300K', np.array([0.788, 0.653, 0.761]))]),
]

if __name__ == '__main__':
    pp = PdfPages('multinode_legion.pdf')
    plot_init('Legion Scaling to Multiple Nodes', 'Nodes', 'Particles per Second (in Tens of Millions)', pp)
    index = 1
    for machine, mdata in large_plot:
        for psize, data in mdata:
            plot(large_plot_nodes[:len(data)], total_particles[psize] * total_steps / data,
                 '%s %s' % (machine, psize), index, pp)
            index += 1
    plt.legend(loc=4)
    pp.savefig()
    pp.close()
