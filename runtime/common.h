
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

// Forware declaration
template<typename T> struct ptr_t;

struct utptr_t
{ 
public:
  //utptr_t(void) : value(0) { }
  //utptr_t(const utptr_t &p) : value(p.value) { }
public:
  unsigned value; 
public: 
#ifdef __CUDACC__
  __host__ __device__
#endif
  utptr_t& operator=(const utptr_t &ptr) { value = ptr.value; return *this; }

  template<typename T>
#ifdef __CUDACC__
  __host__ __device__
#endif
  utptr_t& operator=(const ptr_t<T> &ptr) { value = ptr.value; return *this; }

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
  //ptr_t(void) : value(0) { }
  //ptr_t(const utptr_t &p) : value(p.value) { }
  //ptr_t(const ptr_t<T> &p) : value(p.value) { }
public:
  unsigned value; 
public:
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t<T>& operator=(const ptr_t<T> &ptr) { value = ptr.value; return *this; }
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t<T>& operator=(const utptr_t &ptr)  { value = ptr.value; return *this; }
  bool operator==(const ptr_t<T> &ptr) const { return (ptr.value == this->value); }
  bool operator!=(const ptr_t<T> &ptr) const { return (ptr.value != this->value); }
  bool operator< (const ptr_t<T> &ptr) const { return (ptr.value <  this->value); }
  operator bool(void) const { return (value != (unsigned)-1); }
  bool operator!(void) const { return (value == (unsigned)-1); }
  operator utptr_t(void) const { utptr_t ptr; ptr.value = value; return ptr; }

  static ptr_t<T> nil(void) { ptr_t<T> p; p.value = (unsigned)-1; return p; }
};

#endif // COMMON_H
