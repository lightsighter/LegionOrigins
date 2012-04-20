#!/usr/bin/env python

import math, os, re, shutil, subprocess as sp, sys
_root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_root_dir)
from compare import read_file, compare

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

def legion(nbx = 1, nby = 1, nbz = 1, steps = 1, input = None,
           legion_logging = 4,
           **_ignored):
    divisions = nbx*nby*nbz
    cpu_count = min(divisions, get_cpu_count_per_node())
    node_count = min(int(math.ceil(float(divisions) / get_cpu_count_per_node())),
                     get_node_count())
    return check_output(
        (['gasnetrun_ibv', '-n', str(node_count)] if _legion_use_gasnet else []) +
        [_legion_fluid,
         '-ll:csize', '16384', '-ll:gsize', '2000',
         '-ll:cpu', str(cpu_count),
         '-level', str(legion_logging),
         '-nbx', str(nbx), '-nby', str(nby), '-nbz', str(nbz), '-s', str(steps),
        ])

############################################################
## Input

_input_filename = None
def prep_input():
    global _input_filename
    _input_filename = os.path.join(_root_dir, 'init.fluid')
    shutil.copyfile(os.path.join(_root_dir, 'in_300K.fluid'),
                    _input_filename)

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
    return '%sx%sx%s (%s step%s)%s' % (
        nbx, nby, nbz, steps, '' if steps == 1 else 's',
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

_num_steps = 4
_num_reps = 3
if __name__ == '__main__':
    for thunk in prep: thunk()

    print 'Baseline PARSEC serial:'
    init_baseline(parsec_serial, _num_reps, nbx = 1, nby = 1, nbz = 1, steps = _num_steps)
    print

    print 'PARSEC pthreads:'
    init_baseline(parsec_pthreads, _num_reps, nbx = 1, nby = 1, nbz = 1, steps = _num_steps)
    sizes = range(1, 5)
    for size in sizes:
        nbx = 1 << (size/2);
        nby = 1
        nbz = 1 << (size/2);
        if nbx*nbz != 1 << size:
            nbx *= 2
        perf_check(parsec_pthreads, _num_reps, nbx = nbx, nby = nby, nbz = nbz, steps = _num_steps)
    print

    print 'Legion:'
    init_baseline(legion, _num_reps, nbx = 1, nby = 1, nbz = 1, steps = _num_steps)
    sizes = range(1, 8)
    for size in sizes:
        nbx = 1 << (size/2);
        nby = 1
        nbz = 1 << (size/2);
        if nbx*nbz != 1 << size:
            nbx *= 2
        perf_check(legion, _num_reps, nbx = nbx, nby = nby, nbz = nbz, steps = _num_steps)
    print

    #init_baseline(legion, _num_reps, nbx = 1, nby = 1, nbz = 1, steps = _num_steps)
    #sizes = range(1, 5)
    #for size in sizes:
    #    for sx in xrange(size + 1):
    #        for sy in xrange(size - sx + 1):
    #            sz = size - sx - sy
    #            nbx, nby, nbz = 1 << sx, 1 << sy, 1 << sz
    #            perf_check(legion, _num_reps, nbx = nbx, nby = nby, nbz = nbz, steps = _num_steps)
