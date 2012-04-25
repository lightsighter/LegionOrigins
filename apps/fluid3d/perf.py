#!/usr/bin/env python

# Notes:
#   PLEASE use Python >= 2.6.
#   Torque will move this script, so customize _root_dir appropriately.

import math, numpy, os, re, shutil, subprocess as sp, sys
_root_dir = os.path.abspath(os.path.dirname(__file__))
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
         # The upper limit for gsize is 2048 - 2*num_nodes*LMB_SIZE(in MB) .
         '-ll:csize', str(16384), '-ll:gsize', str(1000),
         '-ll:cpu', str(cpus),
        ] +
         # Low-level message threads
        (['-ll:dma', str(2), '-ll:amsg', str(2),]
         if _legion_use_gasnet and nodes > 1 else []) +
         # HACK: Turn off -ll:senders with 4 or more nodes
        (['-ll:senders']
         if _legion_use_gasnet and nodes > 1 and nodes < 4 else []) +
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

def plural(n):
    return ('' if n == 1 else 's')

def summarize_params(nbx = 1, nby = 1, nbz = 1, steps = 1, **others):
    return '%sx%sx%s (%s step%s)%s%s' % (
        nbx, nby, nbz, steps, plural(steps),
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

def run_parsec(cpus, reps, steps):
    print 'PARSEC pthreads (%d cpu%s)' % (cpus, plural(cpus))

    size = int(math.ceil(math.log(cpus, 2)))
    nbx = 1 << (size/2)
    nby = 1
    nbz = 1 << (size/2)
    if nbx*nbz != 1 << size:
        nbx *= 2

    timings = []
    for i in xrange(reps):
        timing = perf_check(parsec_pthreads, 1, nbx = nbx, nby = nby, nbz = nbz, steps = steps)
        if timing is not None: timings.append(timing)
    print 'Mean: %.3f' % numpy.average(timings)
    print 'Median: %.3f' % numpy.median(timings)
    print 'Mean (minus top and bottom 2): %.3f' % numpy.average(sorted(timings)[2:-2])
    print 'Raw: %s' % timings
    print
    if want_plot:
        plot(timings, 'Histogram of PARSEC on %d CPU%s' % (cpus, plural(cpus)))

def run_legion(nodes, cpus, reps, steps):
    print 'Legion (%d node%s %d cpu%s)' % (nodes, plural(nodes),
                                           cpus, plural(cpus))

    size = int(math.ceil(math.log(nodes*cpus, 2)))
    nbx = 1 << (size/2)
    nby = 1
    nbz = 1 << (size/2)
    if nbx*nbz != 1 << size:
        nbx *= 2

    if cpus == 1:
        actual_cpus = cpus
    else:
        # add 1 for utility process
        actual_cpus = cpus + 1

    timings = []
    for i in xrange(reps):
        timing = perf_check(legion, 1, nbx = nbx, nby = nby, nbz = nbz,
                            steps = steps, nodes = nodes, cpus = actual_cpus)
        if timing is not None: timings.append(timing)
    print 'Mean: %.3f' % numpy.average(timings)
    print 'Median: %.3f' % numpy.median(timings)
    print 'Mean (minus top and bottom 2): %.3f' % numpy.average(sorted(timings)[2:-2])
    print 'Raw: %s' % timings
    print
    if want_plot:
        plot(timings, 'Histogram of Legion on %d Node%s %d CPU%s' %
             (nodes, plural(nodes), cpus, plural(cpus)))

_num_steps = 100
_num_reps = 10
if __name__ == '__main__':
    for thunk in prep: thunk()

    parsec_sizes = (
        1, 2, 4, 8, 16,
        )
    legion_sizes = (
        (1, 1), (1, 2), (1, 4), (1, 8), (1, 12),
        (2, 8), (2, 10), (2, 12),
        (4, 8), (4, 10), (4, 12),
        (8, 8), (8, 10), (8, 12),
        )

    if 1 in parsec_sizes:
        print 'Baseline PARSEC serial:'
        init_baseline(parsec_serial, _num_reps, nbx = 1, nby = 1, nbz = 1, steps = _num_steps)
        print

    for size in parsec_sizes:
        run_parsec(size, _num_reps, _num_steps)

    for size in legion_sizes:
        run_legion(size[0], size[1], _num_reps, _num_steps)

    if want_plot:
        plt.show()
