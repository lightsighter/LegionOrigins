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

def fresh_file(filename):
    for i in xrange(10000):
        check = '%s.%s' % (filename, i)
        if not os.path.exists(check): return check

def call_silently(command, filename):
    with open(filename, 'wb') as f:
        return sp.call(command, stdout = f, stderr = sp.STDOUT)

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
        print

def parsec(nbx = 1, nby = 1, nbz = 1, steps = 1, input = None, output = None,
           **_ignored):
    cmd_out = fresh_file('parsec.out')
    retcode = call_silently(
        [_parsec_fluid, str(nbx*nby*nbz), str(steps),
         str(input), str(output)],
        cmd_out)
    return (retcode, cmd_out)

_legion_fluid = None
def prep_legion():
    global _legion_fluid
    _legion_fluid = os.path.join(_root_dir, 'fluid3d')
    if sp.call(['make', '--question', '--silent'], cwd=_root_dir) != 0:
        sp.check_call(['make'], cwd=_root_dir)
        print

def legion(nbx = 1, nby = 1, nbz = 1, steps = 1, input = None, output = None,
           legion_logging = 4,
           **_ignored):
    cmd_out = fresh_file('legion.out')
    retcode = call_silently(
        [_legion_fluid,
         '-ll:csize', '16384', '-ll:gsize', '2000',
         '-ll:l1size', '16384', '-ll:cpu', str(nbx*nby*nbz),
         '-level', str(legion_logging),
         '-nbx', str(nbx), '-nby', str(nby), '-nbz', str(nbz), '-s', str(steps),
        ],
        cmd_out)
    shutil.copyfile(os.path.join(_root_dir, 'output.fluid'),
                    output)
    return (retcode, cmd_out)

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

def read_result(ps):
    program, status = ps
    if status[0] != 0: return None
    return read_file('%s.fluid' % program.__name__)

_max_epsilon = 1.0e-5
def validate(epsilons):
    return all(map(lambda e: e < _max_epsilon, epsilons.itervalues()))

def summarize(params, epsilons):
    return '%s ==> %s' % (
        ', '.join(['%s %s' % kv for kv in params.iteritems()]),
        ', '.join(['%s %.1e' % kv for kv in epsilons.iteritems()]),
        )

_red="\033[1;31m"
_green="\033[1;32m"
_clear="\033[0m"
_pass="[ %sPASS%s ]" % (_green, _clear)
_fail="[ %sFAIL%s ]" % (_red, _clear)

def regress(**params):
    statuses = [program(input = get_input(),
                        output = '%s.fluid' % program.__name__,
                        **params)
                for program in programs]
    results = map(read_result, zip(programs, statuses))
    for i1, s1, r1 in zip(range(len(results)), statuses, results):
        for i2, s2, r2 in zip(range(len(results)), statuses, results):
            if i1 < i2:
                es = compare(r1, r2)
                if es is None:
                    print '%s (see %s %s)' % (_fail, s1[1], s2[1])
                else:
                    passes = validate(es)
                    pass_str = _pass if passes else _fail
                    summary = summarize(params, es)
                    see = '' if passes else ' (see %s %s)' % (s1[1], s2[1])
                    print '%s %s%s' % (pass_str, summary, see)

if __name__ == '__main__':
    for thunk in prep: thunk()
    regress(nbx = 1, nby = 1, nbz = 1, steps = 1)
    regress(nbx = 1, nby = 1, nbz = 1, steps = 2)
    regress(nbx = 2, nby = 1, nbz = 1, steps = 1)
