import re

class Memory(object):
    def __init__(self, name, capacity):
        self.name = name
        self.total_bytes = capacity
        self.avail_bytes = capacity
        self.allocations = dict()
        self.proc_affinity = dict()

    def reserve_space(self, allocator, min_bytes, max_bytes = None):
        if(min_bytes > self.avail_bytes):
            return 0
        bytes = min_bytes
        if max_bytes <> None:
            if max_bytes > self.avail_bytes:
                bytes = self.avail_bytes
            else:
                bytes = max_bytes
        self.avail_bytes = self.avail_bytes - bytes
        self.allocations[allocator] = self.allocations.get(allocator, 0) + bytes
        return bytes

    def release_space(self, allocator, num_bytes):
        if self.allocations.get(allocator, 0) < num_bytes:
            raise UnexpectedFree(self, allocator, num_bytes)
        self.avail_bytes = self.avail_bytes + bytes
        self.allocations[allocator] = self.allocations.get(allocator, 0) - bytes

class Processor(object):
    def __init__(self, name, proc_type, proc_speed):
        self.name = name
        self.proc_type = proc_type
        self.proc_speed = proc_speed
        self.memory_affinity = dict()
        self.pending_tasks = []
        self.running_tasks = []

class Machine(object):
    def __init__(self, name):
        self.name = name
        self.processors = dict()
        self.memories = dict()

    def read_config_file(self, filename):
        f = open(filename, "r")
        for line in f:
            if re.search(r"\S", line) is None:
                continue

            m = re.match(r"\s*processor\s+(?P<name>\S+)\s+(?P<proc_type>\S+)\s+(?P<proc_speed>\d+)", line)
            if m <> None:
                p = Processor(**m.groupdict())
                self.processors[m.group('name')] = p
                continue

            m = re.match(r"\s*memory\s+(?P<name>\S+)\s+(?P<size>\d+)(?P<mult>[kmg]?)", line)
            if m <> None:
                bytes = int(m.group('size')) * dict({ "": 1, "k": 1024, "m": 1024 * 1024, "g": 1024 * 1024 * 1024 })[m.group('mult')]
                mm = Memory(m.group('name'), bytes)
                self.memories[m.group('name')] = mm
                continue

            m = re.match(r"\s*affinity\s+(?P<pname>\S+)\s+(?P<mname>\S+)\s+(?P<bandwidth>\d+)\s+(?P<latency>\d+)", line)
            if m <> None:
                p = self.processors[m.group("pname")]
                mm = self.memories[m.group("mname")]
                a = { "proc": p, "memory": mm, "bandwidth": m.group("bandwidth"), "latency": m.group("latency") }
                p.memory_affinity[mm.name] = a
                mm.proc_affinity[p.name] = a
                continue

            print "What to do with: " + line + "?"
        f.close()

