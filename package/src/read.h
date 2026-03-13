#ifndef FMD_READ_H
#define FMD_READ_H

#include "types.h"

class RCmemdecoder : public RCdecoder {
public:
  RCmemdecoder(const void* buffer) : RCdecoder(), ptr((const uchar*)buffer), begin(ptr) {}
  uint getbyte() { return *ptr++; }
  size_t bytes() const { return ptr - begin; }
private:
  const uchar* ptr;
  const uchar* const begin;
};

#endif
