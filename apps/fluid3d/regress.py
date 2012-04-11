#!/usr/bin/env python

import os, re, shutil, subprocess as sp, sys
_root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_root_dir)
from compare import read_file, compare

def newer (filename1, filename2):
    if not os.path.exists(filename1):
        return False
    if not os.path.exists(filename2):
        return True
    return os.path.getmtime(filename1) > os.path.getmtime(filename2)

def fresh_file(filepath, ext):
    for i in xrange(10000):
        check = '%s.%s.%s' % (filepath, i, ext)
        if not os.path.exists(check): return check

def call_silently(command, filename):
    with open(filename, 'wb') as f:
        return sp.call(command, stdout = f, stderr = sp.STDOUT)

_legion_fluid = None
_legion_use_gasnet = True
def prep_legion():
    global _legion_fluid
    _legion_fluid = os.path.join(_root_dir, 'fluid3d')
    if sp.call(['make', '--question', '--silent'], cwd=_root_dir) != 0:
        sp.check_call(['make'], cwd=_root_dir)
        print

def legion(nbx = 1, nby = 1, nbz = 1, steps = 1, input = None, output = None,
           legion_logging = 1,
           **_ignored):
    cmd_out = fresh_file('legion', 'log')
    retcode = call_silently(
        (['gasnetrun_ibv', '-n', str(1)] if _legion_use_gasnet else []) +
        [_legion_fluid,
         '-ll:csize', '16384', '-ll:gsize', '2000',
         '-ll:cpu', str(1),#str(nbx*nby*nbz),
         '-level', str(legion_logging),
         '-nbx', str(nbx), '-nby', str(nby), '-nbz', str(nbz), '-s', str(steps),
        ],
        cmd_out)
    shutil.copyfile(os.path.join(_root_dir, 'output.fluid'),
                    output)
    return (retcode, cmd_out)

_input_filename = None
def prep_input(size = 5):
    global _input_filename
    _input_filename = os.path.join(_root_dir, 'init.fluid')
    shutil.copyfile(os.path.join(_root_dir, 'in_%dK.fluid' % size),
                    _input_filename)

def get_input():
    return _input_filename

_solution_filename = None
def prep_solution(size = 5):
    global _solution_filename
    _solution_filename = os.path.join(_root_dir, 'out_%dK.fluid' % size)

def get_solution():
    return _solution_filename

def get_output_for_program(program):
    return '%s.fluid' % program.__name__

_output_re = re.compile(r'.*\.log')
def cleanup_output():
    for path in os.listdir(_root_dir):
        if (os.path.isfile(path) and
            re.match(_output_re, os.path.basename(path)) is not None):
            os.remove(path)

prep = [prep_legion, prep_input, prep_solution, cleanup_output]
programs = [legion]

def read_result(ps):
    program, status = ps
    if status[0] != 0: return None
    return read_file(get_output_for_program(program))

_max_epsilon = 1.0e-5
def validate(epsilons):
    return all(map(lambda e: e < _max_epsilon, epsilons.itervalues()))

def summarize_params(nbx = 1, nby = 1, nbz = 1, steps = 1, **others):
    return '%sx%sx%s (%s step%s)%s' % (
        nbx, nby, nbz, steps, '' if steps == 1 else 's',
        ', '.join(['%s %s' % kv for kv in others.iteritems()]))

_status_table = [
'SIGHUP',
'SIGINT',
'SIGQUIT',
'SIGILL',
'SIGTRAP',
'SIGABRT',
'SIGBUS',
'SIGFPE',
'SIGKILL',
'SIGUSR1',
'SIGSEGV',
'SIGUSR2',
'SIGPIPE',
'SIGALRM',
'SIGTERM',
'SIGCHLD',
'SIGCONT',
'SIGSTOP',
'SIGTSTP',
'SIGTTIN',
'SIGTTOU',
'SIGURG',
'SIGXCPU',
'SIGXFSZ',
'SIGVTALRM',
'SIGPROF',
'SIGPOLL',
'SIGSYS',
]
def summarize_status(status):
    if status > 0:
        return 'Exited with status %s' % status
    elif status < 0:
        return 'Killed by signal %s' % _status_table[-status-1]
    else:
        return 'Exited normally'

def summarize(params, epsilons, status):
    return '%s ==> %s' % (
        summarize_params(**params),
        (', '.join(['%s %.1e' % kv for kv in epsilons.iteritems()])
         if epsilons is not None else summarize_status(status)),
        )

_red="\033[1;31m"
_green="\033[1;32m"
_clear="\033[0m"
_pass="[ %sPASS%s ]" % (_green, _clear)
_fail="[ %sFAIL%s ]" % (_red, _clear)

def regress(**params):
    statuses = [program(input = get_input(),
                        output = get_output_for_program(program),
                        **params)
                for program in programs]
    results = map(read_result, zip(programs, statuses))
    solution = read_file(get_solution())
    for status, result in zip(statuses, results):
        es = compare(result, solution)
        summary = summarize(params, es, status[0])
        if es is None:
            print '%s %s (see %s)' % (_fail, summary, status[1])
        else:
            passes = validate(es)
            pass_str = _pass if passes else _fail
            see = '' if passes else ' (see %s)' % status[1]
            print '%s %s%s' % (pass_str, summary, see)

if __name__ == '__main__':
    for thunk in prep: thunk()

    print 'Testing small (5K) input.'
    divs = (1, 2, 4)
    for nbx in divs:
        for nby in divs:
            for nbz in divs:
                regress(nbx = nbx, nby = nby, nbz = nbz, steps = 1)

    print
    print "Note: The following are expected to fail, but should NOT crash."
    divs = (1, 2)
    for nbx in divs:
        for nby in divs:
            for nbz in divs:
                regress(nbx = nbx, nby = nby, nbz = nbz, steps = 4)

    prep_input(300)
    prep_solution(300)
    print
    print 'Testing large (300K) input. (This might take a while.)'
    divs = (1, 2, 4)
    for nbx in divs:
        for nby in divs:
            for nbz in divs:
                regress(nbx = nbx, nby = nby, nbz = nbz, steps = 1)
