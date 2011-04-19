#!/usr/bin/python

from regions import *
from pprint import pprint

bt = BasicDataType("int")
rt = RegionType(bt)
rv = RegionValue(rt)
ri = rv.get_instance()
p = ri.allocate(7)

pprint(p)
print(p)
print(p.my_type)
print(p.__dict__)
print(rv)
