
#ifndef COMMON_H
#define COMMON_H

// This file contains declarations of objects
// that need to be globally visible to all layers
// of the program including the application code
// as well as both of the runtimes.

typedef unsigned int ReductionID;
// A mapping tag ID is a way of telling the
// mapper what context a function is being called in
// so it can optimize for certain cases.
typedef unsigned int MappingTagID;

template<typename T>
struct ptr_t { unsigned value; };

class Context
{

};

#endif // COMMON_H