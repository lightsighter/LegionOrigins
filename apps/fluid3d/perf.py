#!/usr/bin/env python

import os, re, shutil, subprocess as sp, sys
_root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_root_dir)
from compare import read_file, compare

def check_output(command):
    proc = sp.Popen(command, stdout = sp.PIPE)
    (out, err) = proc.communicate()
    return out

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
    return check_output(
        [_legion_fluid,
         '-ll:csize', '16384', '-ll:gsize', '2000',
         '-ll:l1size', '16384', '-ll:cpu', str(nbx*nby*nbz),
         '-level', str(legion_logging),
         '-nbx', str(nbx), '-nby', str(nby), '-nbz', str(nbz), '-s', str(steps),
        ])

_input_filename = None
def prep_input():
    global _input_filename
    _input_filename = os.path.join(_root_dir, 'init.fluid')
    shutil.copyfile(os.path.join(_root_dir, 'in_5K.fluid'),
                    _input_filename)

def get_input():
    return _input_filename

_solution_filename = None
def prep_solution():
    global _solution_filename
    _solution_filename = os.path.join(_root_dir, 'out_5K.fluid')

prep = [prep_legion, prep_input]
programs = [legion]

_re_timing = re.compile(r'^ELAPSED TIME\s=\s+(\d+\.\d+)\ss$', re.MULTILINE)
def parse_timing(output):
    match = re.search(_re_timing, output)
    if match is None:
        return {'total': 'not available'}
    return {'total': '%s s' % match.group(1)}

def summarize_params(nbx = 1, nby = 1, nbz = 1, steps = 1, **others):
    return '%sx%sx%s (%s step%s)%s' % (
        nbx, nby, nbz, steps, '' if steps == 1 else 's',
        ', '.join(['%s %s' % kv for kv in others.iteritems()]))

def summarize(params, results):
    return '%s ==> %s' % (
        summarize_params(**params),
        ', '.join(['%s %s' % kv for kv in results.iteritems()])
        )

def regress(**params):
    outputs = [program(input = get_input(), **params)
               for program in programs]
    for output in outputs:
        timing = parse_timing(output)
        summary = summarize(params, timing)
        print summary

if __name__ == '__main__':
    for thunk in prep: thunk()
    divs = (1, 2, 4)
    for nbx in divs:
        for nby in divs:
            for nbz in divs:
                regress(nbx = nbx, nby = nby, nbz = nbz, steps = 4)
