#ifndef PC_DECODER_H
#define PC_DECODER_H

#include "types.h"
#include "pcmap.h"
#include "rcdecoder.h"
#include "rcmodel.h"

template <typename T, class M = PCmap<T>>
class PCdecoder {
public:
  PCdecoder(RCdecoder* rd, RCmodel* rm) : rd(rd), rm(rm) {}
  ~PCdecoder() {}
  T decode(T pred);
  static const uint symbols = 2 * M::bits + 1;
private:
  static const uint bias = M::bits;
  M                 map;
  RCdecoder*const   rd;
  RCmodel*          rm;
};

#include "pcdecoder.inl"

#endif
