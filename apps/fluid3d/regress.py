#!/usr/bin/env python

import os, shutil, subprocess as sp, sys
_root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_root_dir)
from compare import read_file, compare

def newer (filename1, filename2):
    if not os.path.exists(filename1):
        return False
    if not os.path.exists(filename2):
        return True
    return os.path.getmtime(filename1) > os.path.getmtime(filename2)

_parsec_dir = _parsec_mgmt = _parsec_fluid = None
def prep_parsec():
    global _parse_dir, _parsec_mgmt, _parsec_fluid
    try:
        _parsec_dir = os.path.abspath(os.environ['PARSEC_DIR'])
    except KeyError:
        print 'Please set the PARSEC_DIR environment variable.'
        sys.exit(1)
    _parsec_mgmt = os.path.join(_parsec_dir, 'bin', 'parsecmgmt')
    _parsec_fluid_src = os.path.join(
        _parsec_dir, 'pkgs', 'apps', 'fluidanimate',
        'src', 'pthreads.cpp')
    _parsec_fluid = os.path.join(
        _parsec_dir, 'pkgs', 'apps', 'fluidanimate',
        'inst', 'amd64-linux.gcc', 'bin', 'fluidanimate')
    if (newer(_parsec_fluid_src, _parsec_fluid)):
        sp.check_call([_parsec_mgmt, '-a', 'fullclean'])
        sp.check_call([_parsec_mgmt, '-a', 'fulluninstall'])
        sp.check_call([_parsec_mgmt, '-a', 'build', '-p', 'fluidanimate'])

def parsec(nbx = 1, nby = 1, nbz = 1, steps = 1, input = None, output = None,
           **_ignored):
    sp.check_call([_parsec_fluid, str(nbx*nby*nbz), str(steps),
                   str(input), str(output)])

_legion_fluid = None
def prep_legion():
    global _legion_fluid
    _legion_fluid = os.path.join(_root_dir, 'fluid3d')
    sp.check_call(['make'], cwd=_root_dir)

def legion(nbx = 1, nby = 1, nbz = 1, steps = 1, input = None, output = None,
           legion_logging = 4,
           **_ignored):
    sp.check_call(
        [_legion_fluid,
         '-ll:csize', '16384', '-ll:gsize', '2000',
         '-ll:l1size', '16384', '-ll:cpu', str(nbx*nby*nbz),
         '-level', str(legion_logging),
         '-nbx', str(nbx), '-nby', str(nby), '-nbz', str(nbz), '-s', str(steps),
        ])
    shutil.copyfile(os.path.join(_root_dir, 'output.fluid'),
                    output)

_input_filename = None
def prep_input():
    global _input_filename
    _input_filename = os.path.join(_root_dir, 'init.fluid')
    shutil.copyfile(os.path.join(_root_dir, 'in_5K.fluid'),
                    _input_filename)

def get_input():
    return _input_filename

prep = [prep_parsec, prep_legion, prep_input]
programs = [legion, parsec]

def regress(**params):
    for program in programs:
        program(input = get_input(),
                output = '%s.fluid' % program.__name__,
                **params)
    results = map(lambda p: read_file('%s.fluid' % p.__name__), programs)
    print
    print 'For %s:' % ', '.join(['%s %s' % kv for kv in params.iteritems()])
    for r1 in results:
        for r2 in results:
            if results.index(r1) < results.index(r2):
                epsilons = compare(r1, r2)
                if epsilons is not None:
                    for k in epsilons.iterkeys():
                        print '  %s:\t%s' % (k, epsilons[k])

if __name__ == '__main__':
    for thunk in prep: thunk()
    regress(nbx = 1, nby = 1, nbz = 1, steps = 1)
    regress(nbx = 1, nby = 1, nbz = 1, steps = 2)
    regress(nbx = 2, nby = 1, nbz = 1, steps = 1)
