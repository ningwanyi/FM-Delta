#ifndef PC_MAP_H
#define PC_MAP_H

#include <climits>
#include <cstring>
#include "types.h"

template <typename T, typename U = void>
struct PCmap;

// specialized for integer-to-integer map
template <typename T>
struct PCmap<T, void> {
  typedef T Domain;
  typedef T Range;
  static const uint bits = CHAR_BIT * sizeof(T);
  Range forward(Domain d) const { return d; }
  Domain inverse(Range r) const { return r; }
  Domain identity(Domain d) const { return d; }
};

// specialized for float type
template <>
struct PCmap<float, void> {
  typedef float  Domain;
  typedef uint32 Range;
  static const uint bits = CHAR_BIT * sizeof(Range);
  Range fcast(Domain d) const {
    Range r;
    memcpy(&r, &d, sizeof(r));
    return r;
  }
  Domain icast(Range r) const {
    Domain d;
    memcpy(&d, &r, sizeof(d));
    return d;
  }
  Range forward(Domain d) const {   // map float to uint32 while preserving order
    Range r = fcast(d);
    r = ~r;
    r ^= -(r >> (bits - 1)) >> 1;
    return r;
  }
  Domain inverse(Range r) const {   // inverse of forward mapping
    r ^= -(r >> (bits - 1)) >> 1;
    r = ~r;
    return icast(r);
  }
  Domain identity(Domain d) const { return d; }
};

// specialized for double type
template <>
struct PCmap<double, void> {
  typedef double Domain;
  typedef uint64 Range;
  static const uint bits = CHAR_BIT * sizeof(Range);
  Range fcast(Domain d) const {
    Range r;
    memcpy(&r, &d, sizeof(r));
    return r;
  }
  Domain icast(Range r) const {
    Domain d;
    memcpy(&d, &r, sizeof(d));
    return d;
  }
  Range forward(Domain d) const {
    Range r = fcast(d);
    r = ~r;
    r ^= -(r >> (bits - 1)) >> 1;
    return r;
  }
  Domain inverse(Range r) const {
    r ^= -(r >> (bits - 1)) >> 1;
    r = ~r;
    return icast(r);
  }
  Domain identity(Domain d) const { return d; }
};

// specialized for short type
template <>
struct PCmap<short, void> {
  typedef short  Domain;
  typedef uint16 Range;
  static const uint bits = CHAR_BIT * sizeof(Range);
  Range fcast(Domain d) const {
    Range r;
    memcpy(&r, &d, sizeof(r));
    return r;
  }
  Domain icast(Range r) const {
    Domain d;
    memcpy(&d, &r, sizeof(d));
    return d;
  }
  Range forward(Domain d) const {
    Range r = fcast(d);
    r = ~r;
    Range move = -(r >> (bits - 1));
    move >>= 1;
    r ^= move;
    return r;
  }
  Domain inverse(Range r) const {
    Range move = -(r >> (bits - 1));
    move >>= 1;
    r ^= move;
    r = ~r;
    return icast(r);
  }
  Domain identity(Domain d) const { return d; }
};

#endif
