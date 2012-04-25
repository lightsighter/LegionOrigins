#!/usr/bin/env python

# Notes:
#   PLEASE use Python >= 2.6.
#   Torque will copy your script, so customize _root_dir appropriately.

_root_dir = os.path.abspath(os.path.dirname(__file__))

import math, numpy, os, re, shutil, subprocess as sp, sys
sys.path.append(_root_dir)
from compare import read_file, compare

want_plot = False
if want_plot:
    import matplotlib.pyplot as plt

############################################################
## Utils

def check_output(command):
    proc = sp.Popen(command, stdout = sp.PIPE)
    (out, err) = proc.communicate()
    return out

def newer (filename1, filename2):
    if not os.path.exists(filename1):
        return False
    if not os.path.exists(filename2):
        return True
    return os.path.getmtime(filename1) > os.path.getmtime(filename2)

############################################################
## Machine Specs

_cpu_count_per_node = 12
def get_cpu_count_per_node():
    return _cpu_count_per_node

_node_count = 4
def get_node_count():
    return _node_count

############################################################
## PARSEC

_parsec_dir = _parsec_mgmt = _parsec_fluid_serial = _parsec_fluid_pthreads = None
def prep_parsec():
    global _parse_dir, _parsec_mgmt, _parsec_fluid_serial, _parsec_fluid_pthreads
    try:
        _parsec_dir = os.path.abspath(os.environ['PARSEC_DIR'])
    except KeyError:
        print 'The environment variable PARSEC_DIR is not set.'
        print 'Unable to make baseline comparisons.'
        return
    _parsec_mgmt = os.path.join(_parsec_dir, 'bin', 'parsecmgmt')
    _parsec_fluid_serial_src = os.path.join(
        _parsec_dir, 'pkgs', 'apps', 'fluidanimate',
        'src', 'serial.cpp')
    _parsec_fluid_pthreads_src = os.path.join(
        _parsec_dir, 'pkgs', 'apps', 'fluidanimate',
        'src', 'pthreads.cpp')
    _parsec_fluid_serial = os.path.join(
        _parsec_dir, 'pkgs', 'apps', 'fluidanimate',
        'inst', 'amd64-linux.gcc-serial', 'bin', 'fluidanimate')
    _parsec_fluid_pthreads = os.path.join(
        _parsec_dir, 'pkgs', 'apps', 'fluidanimate',
        'inst', 'amd64-linux.gcc-pthreads', 'bin', 'fluidanimate')
    if (newer(_parsec_fluid_pthreads_src, _parsec_fluid_pthreads) or
        newer(_parsec_fluid_serial_src, _parsec_fluid_serial)):
        sp.check_call([_parsec_mgmt, '-a', 'fullclean'])
        sp.check_call([_parsec_mgmt, '-a', 'fulluninstall'])
        sp.check_call([_parsec_mgmt, '-a', 'build', '-p', 'fluidanimate', '-c', 'gcc-serial'])
        sp.check_call([_parsec_mgmt, '-a', 'build', '-p', 'fluidanimate', '-c', 'gcc-pthreads'])
        print

def parsec_serial(steps = 1, input = None, **_ignored):
    return check_output(
        [_parsec_fluid_serial, str(1), str(steps),
         str(input)])

def parsec_pthreads(nbx = 1, nby = 1, nbz = 1, steps = 1, input = None,
           **_ignored):
    return check_output(
        [_parsec_fluid_pthreads, str(nbx*nby*nbz), str(steps),
         str(input)])

############################################################
## Legion

_legion_fluid = None
_legion_use_gasnet = True
def prep_legion():
    global _legion_fluid
    _legion_fluid = os.path.join(_root_dir, 'fluid3d')
    if sp.call(['make', '--question', '--silent'], cwd=_root_dir) != 0:
        sp.check_call(['make'], cwd=_root_dir)
        print

def legion(nbx = 1, nby = 1, nbz = 1, steps = 1, nodes = 1, cpus = 0,
           input = None,
           legion_logging = 4,
           **_ignored):
    if cpus <= 0: cpus = nbx*nby*nbz/nodes + 1
    return check_output(
        (['gasnetrun_ibv', '-n', str(nodes)] if _legion_use_gasnet else []) +
        [_legion_fluid,
         '-ll:csize', '16384', '-ll:gsize', '2000',
         '-ll:cpu', str(cpus),
        ] +
         # Low-level message threads
        (['-ll:dma', str(2), '-ll:amsg', str(2), '-ll:senders',]
         if _legion_use_gasnet and nodes > 1 else []) +
         # High-level scheduler look-ahead
        ['-hl:sched', str(2*nbx*nby*nbz),
         '-level', str(legion_logging),
         '-nbx', str(nbx), '-nby', str(nby), '-nbz', str(nbz), '-s', str(steps),
         '-input', str(input),
        ])

############################################################
## Input

_input_filename = None
def prep_input(size = 2400):
    global _input_filename
    _input_filename = os.path.join(_root_dir, 'in_%dK.fluid' % size)

def get_input():
    return _input_filename

############################################################
## Performance Check

prep = [prep_parsec, prep_legion, prep_input]

_re_timing = re.compile(r'^ELAPSED TIME\s=\s+(\d+\.\d+)\ss$', re.MULTILINE)
def parse_timing(output):
    match = re.search(_re_timing, output)
    if match is None:
        return None
    return float(match.group(1))

_baseline = []
def get_baseline():
    return _baseline

def summarize_timing(timing):
    if timing is None:
        return {'error': 'timing not available'}
    if get_baseline() is None:
        return {'total': '%0.3f s' % timing, 'speedup': None}
    return {'total': '%0.3f s' % timing,
            'speedup': ' / '.join('%0.3f' % (base / timing)
                                  for base in (get_baseline()))}

def summarize_params(nbx = 1, nby = 1, nbz = 1, steps = 1, **others):
    return '%sx%sx%s (%s step%s)%s%s' % (
        nbx, nby, nbz, steps, '' if steps == 1 else 's',
        ' ' if len(others) > 0 else '',
        ', '.join(['%s %s' % kv for kv in others.iteritems()]))

def summarize(params, results):
    return '%s ==> %s' % (
        summarize_params(**params),
        ', '.join(['%s %s' % kv for kv in results.iteritems()])
        )

def perf_check(program, reps, **params):
    timing = min(parse_timing(program(input = get_input(), **params)) for i in xrange(reps))
    summary = summarize(params, summarize_timing(timing))
    print summary
    return timing

def init_baseline(program, reps, **params):
    global _baseline
    timing = perf_check(program, reps, **params)
    _baseline.append(timing)
    return timing

def plot(nums, title):
    plt.figure()
    plt.hist(nums, facecolor='green', alpha=0.75)
    plt.xlabel('Run Time (s)')
    plt.ylabel('Count')
    plt.title(title)
    plt.axis([0, 100, 0, 10])
    plt.grid(True)

_num_steps = 100
_num_reps = 1
if __name__ == '__main__':
    for thunk in prep: thunk()

    sizes = set([
        'p1', 'p8', 'p16',
        'l1-8', 'l1-12',
        'l2-8', 'l2-10', 'l2-12',
        'l4-8', 'l4-10', 'l4-12',
        'l8-8', 'l8-10', 'l8-12',
        ])

    if 'p1' in sizes:
        print 'Baseline PARSEC serial:'
        init_baseline(parsec_serial, _num_reps, nbx = 1, nby = 1, nbz = 1, steps = _num_steps)
        print

    if 'p8' in sizes:
        print 'Parsec 8-cpu:'
        parsec8_timings = []
        for i in xrange(20):
            parsec8_timings.append(perf_check(parsec_pthreads, _num_reps, nbx = 4, nby = 1, nbz = 2, steps = _num_steps, nodes = 1))
        print 'Mean:', numpy.average(parsec8_timings)
        print 'Median:', numpy.median(parsec8_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(parsec8_timings)[2:-2])
        print parsec8_timings
        print

    if 'p16' in sizes:
        print 'Parsec 16-cpu:'
        parsec16_timings = []
        for i in xrange(20):
            parsec16_timings.append(perf_check(parsec_pthreads, _num_reps, nbx = 4, nby = 1, nbz = 4, steps = _num_steps, nodes = 1))
        print 'Mean:', numpy.average(parsec16_timings)
        print 'Median:', numpy.median(parsec16_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(parsec16_timings)[2:-2])
        print parsec16_timings
        print

    if 'l1-8' in sizes:
        print 'Legion 1-node 8-cpu:'
        legion1_8_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 4, nby = 1, nbz = 2, steps = _num_steps, nodes = 1, cpus = 9)
            if timing is not None: legion1_8_timings.append(timing)
        print 'Mean:', numpy.average(legion1_8_timings)
        print 'Median:', numpy.median(legion1_8_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion1_8_timings)[2:-2])
        print legion1_8_timings
        print

    if 'l1-12' in sizes:
        print 'Legion 1-node 12-cpu:'
        legion1_12_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 4, nby = 1, nbz = 4, steps = _num_steps, nodes = 1, cpus = 13)
            if timing is not None: legion1_12_timings.append(timing)
        print 'Mean:', numpy.average(legion1_12_timings)
        print 'Median:', numpy.median(legion1_12_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion1_12_timings)[2:-2])
        print legion1_12_timings
        print

    if 'l2-8' in sizes:
        print 'Legion 2-node 8-cpu:'
        legion2_8_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 4, nby = 1, nbz = 4, steps = _num_steps, nodes = 2, cpus = 9)
            if timing is not None: legion2_8_timings.append(timing)
        print 'Mean:', numpy.average(legion2_8_timings)
        print 'Median:', numpy.median(legion2_8_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion2_8_timings)[2:-2])
        print legion2_8_timings
        print

    if 'l2-10' in sizes:
        print 'Legion 2-node 10-cpu:'
        legion2_10_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 8, nby = 1, nbz = 4, steps = _num_steps, nodes = 2, cpus = 11)
            if timing is not None: legion2_10_timings.append(timing)
        print 'Mean:', numpy.average(legion2_10_timings)
        print 'Median:', numpy.median(legion2_10_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion2_10_timings)[2:-2])
        print legion2_10_timings
        print

    if 'l2-12' in sizes:
        print 'Legion 2-node 12-cpu:'
        legion2_12_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 8, nby = 1, nbz = 4, steps = _num_steps, nodes = 2, cpus = 13)
            if timing is not None: legion2_12_timings.append(timing)
        print 'Mean:', numpy.average(legion2_12_timings)
        print 'Median:', numpy.median(legion2_12_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion2_12_timings)[2:-2])
        print legion2_12_timings
        print

    if 'l4-8' in sizes:
        print 'Legion 4-node 8-cpu:'
        legion4_8_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 8, nby = 1, nbz = 4, steps = _num_steps, nodes = 4, cpus = 9)
            if timing is not None: legion4_8_timings.append(timing)
        print 'Mean:', numpy.average(legion4_8_timings)
        print 'Median:', numpy.median(legion4_8_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion4_8_timings)[2:-2])
        print legion4_8_timings
        print

    if 'l4-10' in sizes:
        print 'Legion 4-node 10-cpu:'
        legion4_10_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 8, nby = 1, nbz = 8, steps = _num_steps, nodes = 4, cpus = 11)
            if timing is not None: legion4_10_timings.append(timing)
        print 'Mean:', numpy.average(legion4_10_timings)
        print 'Median:', numpy.median(legion4_10_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion4_10_timings)[2:-2])
        print legion4_10_timings
        print

    if 'l4-12' in sizes:
        print 'Legion 4-node 12-cpu:'
        legion4_12_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 8, nby = 1, nbz = 8, steps = _num_steps, nodes = 4, cpus = 13)
            if timing is not None: legion4_12_timings.append(timing)
        print 'Mean:', numpy.average(legion4_12_timings)
        print 'Median:', numpy.median(legion4_12_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion4_12_timings)[2:-2])
        print legion4_12_timings
        print

    if 'l8-8' in sizes:
        print 'Legion 8-node 8-cpu:'
        legion8_8_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 8, nby = 1, nbz = 8, steps = _num_steps, nodes = 8, cpus = 9)
            if timing is not None: legion8_8_timings.append(timing)
        print 'Mean:', numpy.average(legion8_8_timings)
        print 'Median:', numpy.median(legion8_8_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion8_8_timings)[2:-2])
        print legion8_8_timings
        print

    if 'l8-10' in sizes:
        print 'Legion 8-node 10-cpu:'
        legion8_10_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 16, nby = 1, nbz = 8, steps = _num_steps, nodes = 8, cpus = 11)
            if timing is not None: legion8_10_timings.append(timing)
        print 'Mean:', numpy.average(legion8_10_timings)
        print 'Median:', numpy.median(legion8_10_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion8_10_timings)[2:-2])
        print legion8_10_timings
        print

    if 'l8-12' in sizes:
        print 'Legion 8-node 12-cpu:'
        legion8_12_timings = []
        for i in xrange(20):
            timing = perf_check(legion, _num_reps, nbx = 16, nby = 1, nbz = 8, steps = _num_steps, nodes = 8, cpus = 13)
            if timing is not None: legion8_12_timings.append(timing)
        print 'Mean:', numpy.average(legion8_12_timings)
        print 'Median:', numpy.median(legion8_12_timings)
        print 'Mean (minus top and bottom 2):', numpy.average(sorted(legion8_12_timings)[2:-2])
        print legion8_12_timings
        print

    if want_plot:
        if 'p8' in sizes:    plot(parsec8_timings, 'Histogram of PARSEC on 8 CPUs')
        if 'p16' in sizes:   plot(parsec16_timings, 'Histogram of PARSEC on 16 CPUs')
        if 'l1-8' in sizes:  plot(legion1_8_timings, 'Histogram of Legion on 1 Nodes 8 CPUs')
        if 'l1-12' in sizes: plot(legion1_12_timings, 'Histogram of Legion on 1 Nodes 12 CPUs')
        if 'l2-8' in sizes:  plot(legion2_8_timings, 'Histogram of Legion on 2 Nodes 8 CPUs')
        if 'l2-10' in sizes: plot(legion2_10_timings, 'Histogram of Legion on 2 Nodes 10 CPUs')
        if 'l2-12' in sizes: plot(legion2_12_timings, 'Histogram of Legion on 2 Nodes 12 CPUs')
        if 'l4-8' in sizes:  plot(legion4_8_timings, 'Histogram of Legion on 4 Nodes 8 CPUs')
        if 'l4-10' in sizes: plot(legion4_10_timings, 'Histogram of Legion on 4 Nodes 10 CPUs')
        if 'l4-12' in sizes: plot(legion4_12_timings, 'Histogram of Legion on 4 Nodes 12 CPUs')
        if 'l8-8' in sizes:  plot(legion8_8_timings, 'Histogram of Legion on 8 Nodes 8 CPUs')
        if 'l8-10' in sizes: plot(legion8_10_timings, 'Histogram of Legion on 8 Nodes 10 CPUs')
        if 'l8-12' in sizes: plot(legion8_12_timings, 'Histogram of Legion on 8 Nodes 12 CPUs')
        plt.show()

    print 'KILL ME NOW PLEASE'
    sys.exit()

    print 'PARSEC pthreads:'
    sizes = range(0, 5)
    for size in sizes:
        nbx = 1 << (size/2);
        nby = 1
        nbz = 1 << (size/2);
        if nbx*nbz != 1 << size:
            nbx *= 2
        perf_check(parsec_pthreads, _num_reps, nbx = nbx, nby = nby, nbz = nbz, steps = _num_steps)
    print

    print 'Legion 1-node:'
    sizes = range(0, 5)
    for size in sizes:
        nbx = 1 << (size/2);
        nby = 1
        nbz = 1 << (size/2);
        if nbx*nbz != 1 << size:
            nbx *= 2
        perf_check(legion, _num_reps, nbx = nbx, nby = nby, nbz = nbz, steps = _num_steps, nodes = 1)
    print

    print 'Legion 2-nodes:'
    sizes = range(4, 7)
    for size in sizes:
        for sx in xrange(size + 1):
            for sy in xrange(size - sx + 1):
                sz = size - sx - sy
                nbx, nby, nbz = 1 << sx, 1 << sy, 1 << sz
                # estimate max ghost cell region size and avoid running any over LMB_SIZE
                dims = (93 / nbx, 129 / nby, 93 / nbz)
                LMB_SIZE = 4.0
                ghosts = max([dims[x]*dims[y] for x in xrange(len(dims)) for y in xrange(len(dims)) if x != y]) * 836.0 / 1024 / 1024
                if ghosts > LMB_SIZE: continue
                perf_check(legion, _num_reps, nbx = nbx, nby = nby, nbz = nbz, steps = _num_steps, nodes = 2, cpus = 10)
    print


    print 'Legion 4-nodes:'
    sizes = range(4, 7)
    for size in sizes:
        for sx in xrange(size + 1):
            for sy in xrange(size - sx + 1):
                sz = size - sx - sy
                nbx, nby, nbz = 1 << sx, 1 << sy, 1 << sz
                # estimate max ghost cell region size and avoid running any over LMB_SIZE
                dims = (93 / nbx, 129 / nby, 93 / nbz)
                LMB_SIZE = 4.0
                ghosts = max([dims[x]*dims[y] for x in xrange(len(dims)) for y in xrange(len(dims)) if x != y]) * 836.0 / 1024 / 1024
                if ghosts > LMB_SIZE: continue
                perf_check(legion, _num_reps, nbx = nbx, nby = nby, nbz = nbz, steps = _num_steps, nodes = 4, cpus = 10)
