
#ifndef __LEGION_UTILITIES_H__
#define __LEGION_UTILITIES_H__

#include "legion_types.h"

namespace RegionRuntime {
  namespace HighLevel {

// Useful macros
#define IS_NO_ACCESS(req) ((req).privilege == NO_ACCESS)
#define IS_READ_ONLY(req) (((req).privilege == NO_ACCESS) || ((req).privilege == READ_ONLY))
#define HAS_WRITE(req) (((req).privilege == READ_WRITE) || ((req).privilege == REDUCE) || ((req).privilege == WRITE_ONLY))
#define IS_WRITE(req) (((req).privilege == READ_WRITE) || ((req).privilege == WRITE_ONLY))
#define IS_WRITE_ONLY(req) ((req).privilege == WRITE_ONLY)
#define IS_REDUCE(req) ((req).privilege == REDUCE)
#define IS_EXCLUSIVE(req) ((req).prop == EXCLUSIVE)
#define IS_ATOMIC(req) ((req).prop == ATOMIC)
#define IS_SIMULT(req) ((req).prop == SIMULTANEOUS)
#define IS_RELAXED(req) ((req).prop == RELAXED)

    /////////////////////////////////////////////////////////////
    // AutoLock 
    /////////////////////////////////////////////////////////////
    // An auto locking class for taking a lock and releasing it when
    // the object goes out of scope
    class AutoLock {
    private:
      const bool is_low;
      Lock low_lock;
      ImmovableLock immov_lock;
    public:
      AutoLock(Lock l, unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT)
        : is_low(true), low_lock(l)
      {
        Event lock_event = l.lock(mode,exclusive,wait_on);
        lock_event.wait(true/*block*/);
      }
      AutoLock(ImmovableLock l)
        : is_low(false), immov_lock(l)
      {
        l.lock();
      }
      ~AutoLock(void)
      {
        if (is_low)
        {
          low_lock.unlock();
        }
        else
        {
          immov_lock.unlock();
        }
      }
    };

    /////////////////////////////////////////////////////////////
    // Serializer 
    /////////////////////////////////////////////////////////////
    class Serializer {
    public:
      Serializer(size_t buffer_size);
      ~Serializer(void) 
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(index == long(total_bytes)); // We should have used the whole buffer
#endif
        free(buffer);
      }
    public:
      template<typename T>
      inline void serialize(const T &element);
      inline void serialize(const void *src, size_t bytes);
      inline void grow(size_t more_bytes);
      inline const void* get_buffer(void) const 
      { 
#ifdef DEBUG_HIGH_LEVEL
        assert(index == long(total_bytes));
#endif
        return buffer; 
      }
    private:
      size_t total_bytes;
      char *buffer;
      off_t index;
    };

    /////////////////////////////////////////////////////////////
    // Deserializer 
    /////////////////////////////////////////////////////////////
    class Deserializer {
    public:
      friend class HighLevelRuntime;
      friend class TaskContext;
      Deserializer(const void *buffer, size_t buffer_size);
      ~Deserializer(void)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remaining_bytes == 0); // should have used the whole buffer
#endif
      }
    public:
      template<typename T>
      inline void deserialize(T &element);
      inline void deserialize(void *dst, size_t bytes);
      inline size_t get_remaining_bytes(void) const { return remaining_bytes; }
      inline const void* get_pointer(void);
      inline void advance_pointer(size_t bytes);
    private:
      const char *location;
      size_t remaining_bytes;
    };

    /////////////////////////////////////////////////////////////
    // Fraction 
    /////////////////////////////////////////////////////////////
    template<typename T>
    class Fraction {
    public:
      Fraction(void);
      Fraction(T num, T denom);
      Fraction(const Fraction<T> &f);
    public:
      void divide(T factor);
      void add(const Fraction<T> &rhs);
      void subtract(const Fraction<T> &rhs);
      // Return a fraction that can be taken from this fraction 
      // such that it leaves at least 1/ways parts local after (ways-1) portions
      // are taken from this instance
      Fraction<T> get_part(T ways);
    public:
      bool is_whole(void) const;
      bool is_empty(void) const;
    public:
      Fraction<T>& operator=(const Fraction<T> &rhs);
    private:
      T numerator;
      T denominator;
    };

    /////////////////////////////////////////////////////////////
    // Bit Mask 
    /////////////////////////////////////////////////////////////
    template<typename T, unsigned int MAX>
    class BitMask {
    public:
      BitMask(void);
      BitMask(const BitMask &rhs);
    public:
      inline bool operator==(const BitMask &rhs) const;
      inline bool operator<(const BitMask &rhs) const;
    public:
      inline const T& operator[](const unsigned &idx) const;
      inline T& operator[](const unsigned &idx);
      inline BitMask& operator=(const BitMask &rhs);
    public:
      inline BitMask operator~(void) const;
      inline BitMask operator|(const BitMask &rhs) const;
      inline BitMask operator&(const BitMask &rhs) const;
      inline BitMask operator^(const BitMask &rhs) const;
    public:
      inline BitMask& operator|=(const BitMask &rhs);
      inline BitMask& operator&=(const BitMask &rhs);
      inline BitMask& operator^=(const BitMask &rhs);
    public:
      // Use * for disjointness testing
      inline bool operator*(const BitMask &rhs) const;
      // Set difference
      inline BitMask operator-(const BitMask &rhs) const;
      inline BitMask& operator-=(const BitMask &rhs);
      // Test to see if everything is zeros
      inline bool operator!(void) const;
    private:
      T bit_vector[MAX/(8*sizeof(T))];
    };

    //--------------------------------------------------------------------------
    // Give the implementations here so the templates get instantiated
    //--------------------------------------------------------------------------
    
    //--------------------------------------------------------------------------
    Serializer::Serializer(size_t num_bytes)
      : total_bytes(num_bytes), buffer((char*)malloc(num_bytes)), index(0) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Serializer::serialize(const T &element)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((index + sizeof(T)) <= total_bytes); // Check to make sure we don't write past the end
#endif
      *((T*)(&(buffer[index]))) = element;
      index += sizeof(T);
    }

    //--------------------------------------------------------------------------
    inline void Serializer::serialize(const void *src, size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((index + bytes) <= total_bytes);
#endif
      memcpy(&(buffer[index]),src,bytes);
      index += bytes;
    }

    //--------------------------------------------------------------------------
    inline void Serializer::grow(size_t more_bytes)
    //--------------------------------------------------------------------------
    {
      total_bytes += more_bytes;
      buffer = (char*)realloc(buffer,total_bytes); 
    }

    //--------------------------------------------------------------------------
    Deserializer::Deserializer(const void *buffer, size_t buffer_size)
      : location((const char*)buffer), remaining_bytes(buffer_size)
    //--------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------- 
    template<typename T>
    inline void Deserializer::deserialize(T &element)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes >= sizeof(T)); // Check to make sure we don't read past the end
#endif
      element = *((const T*)location);
      location += sizeof(T);
      remaining_bytes -= sizeof(T);
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::deserialize(void *dst, size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes >= bytes);
#endif
      memcpy(dst,location,bytes);
      location += bytes;
      remaining_bytes -= bytes;
    }

    //--------------------------------------------------------------------------
    inline const void* Deserializer::get_pointer(void)
    //--------------------------------------------------------------------------
    {
      return location;
    }

    //--------------------------------------------------------------------------
    inline void Deserializer::advance_pointer(size_t bytes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remaining_bytes >= bytes);
#endif
      location += bytes;
      remaining_bytes -= bytes;
    }

    // There is an interesting design decision about how to break up the 32 bit
    // address space for fractions.  We'll assume that there will be some
    // balance between the depth and breadth of the task tree so we can split up
    // the fractions efficiently.  We assume that there will be large fan-outs
    // in the task tree as well as potentially large numbers of task calls at
    // each node.  However, we'll assume that the tree is not very deep.
#define MIN_FRACTION_SPLIT    256
    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>::Fraction(void)
      : numerator(256), denominator(256)
    //-------------------------------------------------------------------------
    {
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>::Fraction(T num, T denom)
      : numerator(num), denominator(denom)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(denom > 0);
#endif
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>::Fraction(const Fraction<T> &f)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(f.denominator > 0);
#endif
      numerator = f.numerator;
      denominator = f.denominator;
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void Fraction<T>::divide(T factor)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(factor != 0);
      assert(denominator > 0);
#endif
      T new_denom = denominator * factor;
#ifdef DEBUG_HIGH_LEVEL
      assert(new_denom > 0); // check for integer overflow
#endif
      denominator = new_denom;
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void Fraction<T>::add(const Fraction<T> &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(denominator > 0);
#endif
      if (denominator == rhs.denominator)
      {
        numerator += rhs.numerator;
      }
      else
      {
        // Denominators are different, make them the same
        // Check if one denominator is divisible by another
        if ((denominator % rhs.denominator) == 0)
        {
          // Our denominator is bigger
          T factor = denominator/rhs.denominator; 
          numerator += (rhs.numerator*factor);
        }
        else if ((rhs.denominator % denominator) == 0)
        {
          // Rhs denominator is bigger
          T factor = rhs.denominator/denominator;
          numerator = (numerator*factor) + rhs.numerator;
          denominator *= factor;
#ifdef DEBUG_HIGH_LEVEL
          assert(denominator > 0); // check for integer overflow
#endif
        }
        else
        {
          // One denominator is not divisible by the other, compute a common denominator
          T lhs_num = numerator * rhs.denominator;
          T rhs_num = rhs.numerator * denominator;
          numerator = lhs_num + rhs_num;
          denominator *= rhs.denominator;
#ifdef DEBUG_HIGH_LEVEL
          assert(denominator > 0); // check for integer overflow
#endif
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(numerator <= denominator); // Should always be less than or equal to 1
#endif
    }

    //-------------------------------------------------------------------------
    template<typename T>
    void Fraction<T>::subtract(const Fraction<T> &rhs)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(denominator > 0);
#endif
      if (denominator == rhs.denominator)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(numerator >= rhs.numerator); 
#endif
        numerator -= rhs.numerator;
      }
      else
      {
        if ((denominator % rhs.denominator) == 0)
        {
          // Our denominator is bigger
          T factor = denominator/rhs.denominator;
#ifdef DEBUG_HIGH_LEVEL
          assert(numerator >= (rhs.numerator*factor));
#endif
          numerator -= (rhs.numerator*factor);
        }
        else if ((rhs.denominator % denominator) == 0)
        {
          // Rhs denominator is bigger
          T factor = rhs.denominator/denominator;
#ifdef DEBUG_HIGH_LEVEL
          assert((numerator*factor) >= rhs.numerator);
#endif
          numerator = (numerator*factor) - rhs.numerator;
          denominator *= factor;
#ifdef DEBUG_HIGH_LEVEL
          assert(denominator > 0); // check for integer overflow
#endif
        }
        else
        {
          // One denominator is not divisible by the other, compute a common denominator
          T lhs_num = numerator * rhs.denominator;
          T rhs_num = rhs.numerator * denominator;
#ifdef DEBUG_HIGH_LEVEL
          assert(lhs_num >= rhs_num);
#endif
          numerator = lhs_num - rhs_num;
          denominator *= rhs.denominator; 
#ifdef DEBUG_HIGH_LEVEL
          assert(denominator > 0); // check for integer overflow
#endif
        }
      }
      // Check to see if the numerator has gotten down to one, if so bump up the
      // fraction split
      if (numerator == 1)
      {
        numerator *= MIN_FRACTION_SPLIT;
        denominator *= MIN_FRACTION_SPLIT;
#ifdef DEBUG_HIGH_LEVEL
        assert(denominator > 0); // check for integer overflow
#endif
      }
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T> Fraction<T>::get_part(T ways)
    //-------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ways > 0);
      assert(denominator > 0);
#endif
      // Check to see if we have enough parts in the numerator, if not
      // multiply both numerator and denominator by ways
      // and return one over denominator
      if (ways > numerator)
      {
        // Check to see if the ways is at least as big as the minimum split factor
        if (ways < MIN_FRACTION_SPLIT)
        {
          ways = MIN_FRACTION_SPLIT;
        }
        numerator *= ways;
        T new_denom = denominator * ways;
#ifdef DEBUG_HIGH_LEVEL
        assert(new_denom > 0); // check for integer overflow
#endif
        denominator = new_denom;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(numerator >= ways);
#endif
      return Fraction(1,denominator);
    }

    //-------------------------------------------------------------------------
    template<typename T>
    bool Fraction<T>::is_whole(void) const
    //-------------------------------------------------------------------------
    {
      return (numerator == denominator);
    }

    //-------------------------------------------------------------------------
    template<typename T>
    bool Fraction<T>::is_empty(void) const
    //-------------------------------------------------------------------------
    {
      return (numerator == 0);
    }

    //-------------------------------------------------------------------------
    template<typename T>
    Fraction<T>& Fraction<T>::operator=(const Fraction<T> &rhs)
    //-------------------------------------------------------------------------
    {
      numerator = rhs.numerator;
      denominator = rhs.denominator;
      return *this;
    }
#undef MIN_FRACTION_SPLIT

#define BIT_ELMTS (MAX/(8*sizeof(T)))
    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    BitMask<T,MAX>::BitMask(void)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = 0;
      }
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    BitMask<T,MAX>::BitMask(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = rhs[idx];
      }
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline const T& BitMask<T,MAX>::operator[](const unsigned &idx) const
    //-------------------------------------------------------------------------
    {
      return bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline T& BitMask<T,MAX>::operator[](const unsigned &idx) 
    //-------------------------------------------------------------------------
    {
      return bit_vector[idx];
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline bool BitMask<T,MAX>::operator==(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] != rhs[idx]) 
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline bool BitMask<T,MAX>::operator<(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      // Only be less than if the bits are a subset of the rhs bits
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] & ~(bit_vector[idx] & rhs[idx]))
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX>& BitMask<T,MAX>::operator=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] = rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX> BitMask<T,MAX>::operator~(void) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = ~bit_vector[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX> BitMask<T,MAX>::operator|(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] | rhs[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX> BitMask<T,MAX>::operator&(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] & rhs[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX> BitMask<T,MAX>::operator^(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] ^ rhs[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX>& BitMask<T,MAX>::operator|=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] |= rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX>& BitMask<T,MAX>::operator&=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] &= rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX>& BitMask<T,MAX>::operator^=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] ^= rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline bool BitMask<T,MAX>::operator*(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] & rhs[idx])
          return false;
      }
      return true;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX> BitMask<T,MAX>::operator-(const BitMask &rhs) const
    //-------------------------------------------------------------------------
    {
      BitMask<T,MAX> result;
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        result[idx] = bit_vector[idx] & ~rhs[idx];
      }
      return result;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline BitMask<T,MAX>& BitMask<T,MAX>::operator-=(const BitMask &rhs)
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        bit_vector[idx] &= ~rhs[idx];
      }
      return *this;
    }

    //-------------------------------------------------------------------------
    template<typename T, unsigned int MAX>
    inline bool BitMask<T,MAX>::operator!(void) const
    //-------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < BIT_ELMTS; idx++)
      {
        if (bit_vector[idx] != 0)
          return false;
      }
      return true;
    }
#undef BIT_ELMTS

  }; // namespace HighLevel
}; // namespace RegionRuntime

#endif // __LEGION_UTILITIES_H__
