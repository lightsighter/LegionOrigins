
#ifndef COMMON_H
#define COMMON_H

// This file contains declarations of objects
// that need to be globally visible to all layers
// of the program including the application code
// as well as both of the runtimes.

// A mapping tag ID is a way of telling the
// mapper what context a function is being called in
// so it can optimize for certain cases.
typedef unsigned int MappingTagID;

template<typename T>
struct ptr_t 
{ 
public:
  unsigned value; 
public:
  bool operator==(const ptr_t<T> &ptr) const { return (ptr.value == this->value); }
  bool operator!=(const ptr_t<T> &ptr) const { return (ptr.value != this->value); }
  bool operator< (const ptr_t<T> &ptr) const { return (ptr.value <  this->value); }
  operator bool(void) const { return (value != (unsigned)-1); }
  bool operator!(void) const { return (value == (unsigned)-1); }

  static ptr_t<T> nil(void) { ptr_t<T> p; p.value = (unsigned)-1; return p; }
};

#endif // COMMON_H
