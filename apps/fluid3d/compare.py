#!/usr/bin/env python

import math, struct, sys

verbose = False

def read_int32(f):
    return struct.unpack('<i', f.read(4))[0]

def read_float32(f):
    return struct.unpack('<f', f.read(4))[0]

def read_vec3(f):
    return {'x': read_float32(f), 'y': read_float32(f), 'z': read_float32(f)}

def read_particle(f):
    return {'p': read_vec3(f), 'hv': read_vec3(f), 'v': read_vec3(f)}

def read_file(filename):
    content = {}
    with open(filename, 'rb') as f:
        content['rest_particles_per_meter'] = read_float32(f)
        content['orig_num_particles'] = read_int32(f)
        content['particles'] = [read_particle(f)
                                for p in xrange(content['orig_num_particles'])]
    return content

def summarize(filename, parts):
    print 'File %s:' % filename
    print '  Rest parts per meter:\t\t%s' % parts['rest_particles_per_meter']
    print '  Number of particles:\t\t%s' % parts['orig_num_particles']
    print

def distance(vec1, vec2):
    return math.sqrt((vec1['x']-vec2['x'])**2
                     + (vec1['y']-vec2['y'])**2
                     + (vec1['z']-vec2['z'])**2)

def compare(parts1, parts2):
    if parts1 is None or parts2 is None: return None
    if parts1['rest_particles_per_meter'] != parts2['rest_particles_per_meter']:
        print 'Error: Number of particles differ'
        return
    if parts1['orig_num_particles'] != parts2['orig_num_particles']:
        print 'Error: Number of particles differ'
        return
    max_epsilon = {'p': 0, 'hv': 0, 'v': 0}
    for i, (p1, p2) in enumerate(zip(parts1['particles'], parts2['particles'])):
        for k in p1.iterkeys():
            dist = distance(p1[k], p2[k])
            if verbose and (dist > 1e-2):
                print (('Diff: [%d].%s %f (%f,%f,%f) (%f,%f,%f)' %
                        (i, k, dist, 
                         p1[k]['x'], p1[k]['y'], p1[k]['z'],
                         p2[k]['x'], p2[k]['y'], p2[k]['z'])))
            if dist > max_epsilon[k]:
                max_epsilon[k] = dist
    return max_epsilon

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: %s <file1> <file2>' % sys.argv[0]
        sys.exit()
    if sys.argv[1] == '-v':
        sys.argv.pop(1)
        verbose = True
    filename1, filename2 = sys.argv[1], sys.argv[2]
    parts1, parts2 = read_file(filename1), read_file(filename2)
    summarize(filename1, parts1)
    summarize(filename2, parts2)
    epsilons = compare(parts1, parts2)
    if epsilons is not None:
        print 'Maximum error:'
        for k in epsilons.iterkeys():
            print '  %s:\t%s' % (k, epsilons[k])
