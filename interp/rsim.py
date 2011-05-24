#! /usr/bin/python

# region simulator

import sys
import re
from getopt import getopt
from Runtime import Runtime
from Tasks import UnhandledSubtaskException

def help():
    print "Syntax: " + sys.argv[0] + " [-v] [-m machine desc file] [module args*]+"
    print "  -v : verbose output"
    print "  -m : read specified machine description instead of 'machine.desc'"
    print ""
    print "  module args* : name of python module containing test app and optional args"
    exit(2)

opts, cmds = getopt(sys.argv[1:], 'hm:v')
opts = dict(opts)

if ("-h" in opts) or (len(cmds) == 0): help()

# default machine description is 'machine.desc'
machine_desc = opts.get("-m", "machine.desc")
verbose = "-v" in opts

# create a runtime to run these apps
runtime = Runtime(machine_desc)

sys.path.insert(0, "./apps")

errors = 0

for cmd in cmds:
    args = cmd.split(' ')
    srcfile = args.pop(0)
    srcmod = __import__(srcfile)

    # any argument that looks like a number is converted to a number
    args = map(lambda x: (int(x) if re.match(r'^-?\d+$', x) else
                          float(x) if re.match(r'^-?\d+\.\d*(e-?\d+)?$', x) else x), args)

    if verbose: print srcfile + ": running main function from " + srcmod.__file__ + " with args: " + str(args)

    try:
        runtime.run_application(srcmod.main, *args)
        if verbose: print srcfile + ": application completed successfully"
    except UnhandledSubtaskException:
        print srcfile + ": application died due to an unhandled exception!"
        errors = errors + 1

exit(errors)
