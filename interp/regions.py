class Type(object):
    '''Base class for all kinds of types: basic data, pointers, structs, regions, and functions'''
    def create_value(self, *args, **kwargs):
        v = self.value_class(self, *args, **kwargs)
        return v
    pass

class BasicDataType(Type):
    '''Catch-all class for all 'boring' data types: ints, floats, strings, bools, etc.'''
    #value_class = BasicDataValue
    def __init__(self, type_desc):
        '''Creates a new basic data type - the 'type_desc' parameter is just a text string that will be equality-tested for simple type-checking for now.'''
        self.type_desc = type_desc
    def __repr__(self):
        return "Basic(" + str(self.type_desc) + ")"
    pass

class PointerType(Type):
    '''A pointer is an opaque handle that can be used to get and put values from/to a region.  A pointer knows the type of the value it points to and which logical region (a.k.a. RegionValue!) it points into.
    TODO: I think generic pointer types (i.e. the existential types) come from not actually knowing which RegionValue is pointed into and having a placeholder to fill in later.  We'll need to have multiple pointers be able to use the same placeholder.'''
    next_unknown_region = 1
    def __init__(self, data_type, logical_region = None):
        self.data_type = data_type
        if logical_region <> None:
            self.logical_region = logical_region
        else:
            self.logical_region = "unknown_region #" + next_unknown_region
            next_unknown_region = next_unknown_region + 1
    def __repr__(self):
        return str(self.data_type) + "@" + str(self.logical_region)
    pass

class StructType(Type):
    '''A "structure" is probably more accurately called a dictionary - fields have names and types, but there's no standard ordering of the fields in a structure.  (This is to make it easier for me to do "structure slicing" later.)'''
    pass

class RegionType(Type):
    '''A region is a container for values of a certain type, allowing all those values to be manipulated/described concisely.  Multiple regions can exist that hold the same type, so the "type" of a region just needs to know the type of the contained values.'''
    def __init__(self, element_type):
        self.element_type = element_type
    pass

class FunctionType(Type):
    '''Functions are simplified to have a single input value (which can be an anonymous structure with fields named for the formal arguments) and a single output value (which can also be complex).
    TODO: Where does the output value go?  Do we need an output region/pointer pair instead?  It'd be nice to be able to talk about "pure" functions that have no region entanglements, but the output has to go somewhere...'''
    pass


class Value(object):
    '''The base class for all values.  Every value knows its type - the rest is type-dependent.'''
    def __init__(self, my_type):
        self.my_type = my_type
    pass

class BasicDataValue(Value):
    '''A basic data value for now will hold anything a python variable - we'll try to stick to "plain" data types except when we need to cheat for something.'''
    def __init__(self, my_type, value = None):
        super(BasicDataValue, self).__init__(my_type)
        if value <> None:
            self.value = value
    def set_value(new_value):
        self.value = new_value
    def get_value():
        return self.value
    pass

BasicDataType.value_class = BasicDataValue

class PointerValue(Value):
    '''A pointer's type already remember which RegionValue it points into, so the only additional thing we have to store for each individual pointer is a "tag" that lets us find the right element in that region.  We'll use a tag of 'None' for a null pointer.'''
    def __init__(self, my_type, tag = None):
        super(PointerValue, self).__init__(my_type)
        if tag <> None:
            self.tag = tag
    def __repr__(self):
        if self.tag <> None:
            return str(self.tag) + "@" + str(self.my_type.logical_region)
        else:
            return "(null)"
    pass

PointerType.value_class = PointerValue

class StructureValue(Value):
    '''A structure's value is just a mapping of field names to the values of each field.  (Remember, we're not worrying about the layout of structures in memory yet.'''
    def __init__(self, my_type, **my_fields):
        super(StructureValue, self).__init__(my_type)
        self.fields = my_fields
    def set_field(field_name, field_value):
        self.fields[field_name] = field_value
    def get_field(field_name):
        return self.fields[field_name]
    pass

class RegionValue(Value):
    '''A region value is a particular region created at run-time, usually as part of "concrete-izing" a type with abstract region values in it.  A region value is a logical region - there will be one (or possibly more) region instances associated with the logical region that actually hold data.  (When more than one instance exists, some clever consistency mechanisms will be used to keep them in sync. :)  Region instances will handle the normal alloc/free/peek/poke operations.  Operations performed on the logical region are things like partioning.''' 
    next_region_id = 1
    def __init__(self, my_type):
        super(RegionValue, self).__init__(my_type)
        self.region_id = RegionValue.next_region_id
        RegionValue.next_region_id = RegionValue.next_region_id + 1
        self.master_instance = RegionInstance(self)
        self.pointer_type = PointerType(my_type.element_type, self) # now that we're a specific region, we have a specific pointer type
    def get_instance(self, location = None):
        return self.master_instance # for now, we just hand out the master instance to everyone
    def __repr__(self):
        return "R" + str(self.region_id)
    pass

class RegionInstance(object):
    '''A region instance tracks an actual version of a logical region.  It provides alloc/free functionality as well as methods for copying values into and out of the region.'''
    def __init__(self, logical_region):
        self.logical_region = logical_region
        self.next_tag = 1
        self.elements = dict()
    def allocate(self, *args, **kwargs):
        tag = self.next_tag
        self.next_tag = self.next_tag + 1
        self.elements[tag] = self.logical_region.my_type.element_type.create_value(*args, **kwargs)
        return self.logical_region.pointer_type.create_value(tag)
    def free(ptr):
        if ptr.region == self.logical_region:
            del self.elements[ptr.tag]
        else:
            raise RegionMismatchError
    def set_element(ptr, new_value):
        if ptr.region == self.logical_region:
            self.elements[ptr.tag] = new_value
        else:
            raise RegionMismatchError
    def get_element(ptr):
        if ptr.region == self.logical_region:
            return self.elements[ptr.tag]
        else:
            raise RegionMismatchError
    pass
