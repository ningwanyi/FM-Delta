#include <cstddef>
#include "rcencoder.h"

// finalize encoder
void RCencoder::finish()
{
  put(4); // output the last 4 bytes of low
  flush(); 
}

// encode a symbol s using probability modeling
void RCencoder::encode(uint s, RCmodel* rm)
{
  uint l, r;
  rm->encode(s, l, r);
  rm->normalize(range);
  low += range * l;
  range *= r;
  normalize();
}

// encode a number s : 0 <= s < 2^n <= 2^16
void RCencoder::encode_shift(uint s, uint n)  // n is bits width
{
  range >>= n;  // equal to range /= 2^n
  low += range * s; // [low, low+range) is the interval for symbol s
  normalize();
}

// encode a number s : 0 <= s < n <= 2^16
void RCencoder::encode_ratio(uint s, uint n)
{
  range /= n;
  low += range * s;
  normalize();
}

// normalize the range and output data
void RCencoder::normalize()
{
  while (!((low ^ (low + range)) >> 24)) {  // if the top 8 bits are the same, output them
    // top 8 bits are fixed; output them
    put(1); // output the top 8 bits of low
    range <<= 8;  // expand range
  }
  if (!(range >> 16)) { 
    // top 8 bits are not fixed but range is small;
    // fudge range to avoid carry and output 16 bits
    put(2); // output the top 16 bits of low
    range = -low; // set range to complement of low. expand range
  }
}
