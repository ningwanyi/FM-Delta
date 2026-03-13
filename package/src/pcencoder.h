#ifndef PC_ENCODER_H
#define PC_ENCODER_H

#include "types.h"
#include "pcmap.h"
#include "rcencoder.h"
#include "rcmodel.h"

template <typename T, class M = PCmap<T>>
class PCencoder {
public:
  PCencoder(RCencoder* re, RCmodel* rm) : re(re), rm(rm) {}
  void encode(T base, T finetuned);
  static const uint symbols = 2 * M::bits + 1;
private:
  static const uint bias = M::bits;
  M                 map;
  RCencoder*const   re;
  RCmodel*          rm;
};

template <typename U> uint bsr(U x);

#include "pcencoder.inl"

#endif
