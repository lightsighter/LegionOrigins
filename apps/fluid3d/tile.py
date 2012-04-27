#!/usr/bin/env python

# This script takes an input fluid file and makes a resulting fluid
# file which is 8 times larger, by tiling the file twice in each
# dimension.

import math, struct, sys

def read_int32(f):
    return struct.unpack('<i', f.read(4))[0]

def write_int32(f, v):
    f.write(struct.pack('<i', v))

def read_float32(f):
    return struct.unpack('<f', f.read(4))[0]

def write_float32(f, v):
    f.write(struct.pack('<f', v))

def read_vec3(f):
    return {'x': read_float32(f), 'y': read_float32(f), 'z': read_float32(f)}

def write_vec3(f, v):
    write_float32(f, v['x'])
    write_float32(f, v['y'])
    write_float32(f, v['z'])

def read_particle(f):
    return {'p': read_vec3(f), 'hv': read_vec3(f), 'v': read_vec3(f)}

def write_particle(f, v):
    write_vec3(f, v['p'])
    write_vec3(f, v['hv'])
    write_vec3(f, v['v'])

def read_file(filename):
    content = {}
    with open(filename, 'rb') as f:
        content['rest_particles_per_meter'] = read_float32(f)
        content['orig_num_particles'] = read_int32(f)
        content['particles'] = [read_particle(f)
                                for p in xrange(content['orig_num_particles'])]
    return content

def write_file(filename, content):
    with open(filename, 'wb') as f:
        write_float32(f, content['rest_particles_per_meter'])
        write_int32(f, content['orig_num_particles'])
        for p in content['particles']:
            write_particle(f, p)

def neg(v):
    return dict((k, -v[k]) for k in v.iterkeys())

def add(a, b):
    return dict((k, a[k] + b[k]) for k in a.iterkeys())

def sub(a, b):
    return dict((k, a[k] - b[k]) for k in a.iterkeys())

def mul(v, s):
    return dict((k, v[k] * s) for k in v.iterkeys())

def scale(factor):
    def _scale(particle):
        return dict([(k, mul(v, factor)) for k, v in particle.iteritems()])
    return _scale

def translate(offset):
    def _translate(particle):
        result = {'v': particle['v'], 'hv': particle['hv']}
        result['p'] = add(particle['p'], offset)
        return result
    return _translate

_domain_min = {'x': -0.065, 'y': -0.08, 'z': -0.065}
_domain_max = {'x': 0.065, 'y': 0.1, 'z': 0.065}
_center = add(_domain_min, _domain_max)
def tile(original):
    result = {}
    result['rest_particles_per_meter'] = original['rest_particles_per_meter'] * 2.0
    result['orig_num_particles'] = original['orig_num_particles'] * 8
    # this'll make a smaller copy, but still centered on the center
    shrunk = map(translate(_center),
                 map(scale(0.5),
                     map(translate(neg(_center)),
                         original['particles'])))
    # tile each of the 8 copies
    parts = []
    offset = mul(sub(_domain_max, _domain_min), 0.25)
    for x in (-offset['x'], offset['x']):
        for y in (-offset['y'], offset['y']):
            for z in (-offset['z'], offset['z']):
                parts.extend(map(translate({'x': x, 'y': y, 'z': z}), shrunk))
    result['particles'] = parts
    return result

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: %s <input> <output>' % sys.argv[0]
        sys.exit()
    infile, outfile = sys.argv[1], sys.argv[2]
    particles = read_file(infile)
    write_file(outfile, tile(particles))
