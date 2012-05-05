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

def plot(cpus, speedups, label, marker, color, size, fill):
    plt.plot(cpus, speedups, label = label, linestyle = 'dashed', 
             markersize = size, 
             marker = marker, 
             color = color, 
             markerfacecolor = (color if fill else "None"),
             linewidth=0.5)

total_particles = {'300K': 305809, '2400K': 2446472, '19M': 19554392}
total_cells = {'300K': 135424, '2400K': 1115721, '19M': 9056971}
total_steps = 10

large_plot_nodes = [1, 2, 4, 8, 16]
large_plot = [('Keeneland', '19M', np.array([46.377, 21.444, 13.483, 10.891, 10.68])),
              ('Viz', '19M', np.array([43.058, 20.849, 13.434, 10.459])),
              ('Sapling', '19M', np.array([39.89, 21.859, 13.323])),
              ('Keeneland', '2400K', np.array([5.169, 3.206, 2.673, 2.764, 3.801])),
              ('Viz', '2400K', np.array([5.449, 3.331, 2.691, 2.519])),
              ('Sapling', '2400K', np.array([4.957, 2.896, 2.413])),
              ('Keeneland', '300K', np.array([0.912, 0.847, 0.836, 1.158, 2.415])),
              ('Viz', '300K', np.array([0.734, 0.684, 0.873, 1.122])),
              ('Sapling', '300K', np.array([0.788, 0.653, 0.761])),
              ]

machine_marker = { "Sapling": 'o',
                   "Viz": 's',
                   "Keeneland": 'D' }

machine_color = { "Sapling": 'b',
                  "Viz": 'g',
                  "Keeneland": 'r' }

psize_size = { "19M": 7,
               "2400K": 7,
               "300K": 5 }

psize_fill = { "19M": True,
               "2400K": False,
               "300K": True }

if __name__ == '__main__':
    fig = plt.figure(figsize = (10,7))
    plt.xlabel("Nodes")
    plt.ylabel("Particles per second (in millions)")
    plt.grid(True)
    index = 1
    for machine, psize, data in large_plot:
        plot(large_plot_nodes[:len(data)], total_particles[psize] * total_steps / data / 1e6,
             '%s %s' % (machine, psize), 
             machine_marker[machine],
             machine_color[machine],
             psize_size[psize],
             psize_fill[psize],
             )
    plt.legend(loc=4)
    fig.savefig("figs/fluid_multinode.pdf", format="pdf", bbox_inches="tight");
    #pp.close()
