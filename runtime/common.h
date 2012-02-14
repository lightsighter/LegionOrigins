
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

struct utptr_t
{ 
public:
  unsigned value; 
public: 
  bool operator==(const utptr_t &ptr) const { return (ptr.value == this->value); }
  bool operator!=(const utptr_t &ptr) const { return (ptr.value != this->value); }
  bool operator< (const utptr_t &ptr) const { return (ptr.value <  this->value); }
  operator bool(void) const { return (value != (unsigned)-1); }
  bool operator!(void) const { return (value == (unsigned)-1); }

  static utptr_t nil(void) { utptr_t p; p.value = (unsigned)-1; return p; }
};

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
  operator utptr_t(void) const { utptr_t ptr; ptr.value = value; return ptr; }

  static ptr_t<T> nil(void) { ptr_t<T> p; p.value = (unsigned)-1; return p; }
};

#endif // COMMON_H
