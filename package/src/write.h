#ifndef FMD_WRITE_H
#define FMD_WRITE_H

#include "types.h"

// #define subsize(T,n) n
// (int)(CHAR_BIT * sizeof(T)*n/32)
#define subsize(T,n) (CHAR_BIT * sizeof(T) * (n) / 16)

// memory writer for compressed data
class RCmemencoder : public RCencoder {
public:
  RCmemencoder(void* buffer, size_t size) : RCencoder(), ptr((uchar*)buffer), begin(ptr), end(ptr + size) {}
  void putbyte(uint byte)
  {
    if (ptr == end) {
      error = true;
      fmd_errno = fmdErrorBufferOverflow;
    }
    else
      *ptr++ = (uchar)byte;
  }
  size_t bytes() const { return ptr - begin; }
private:
  uchar* ptr;
  const uchar* const begin;
  const uchar* const end;
};

#endif
