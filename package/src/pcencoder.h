#ifndef PC_ENCODER_H
#define PC_ENCODER_H

#include "types.h"
#include "pcmap.h"
#include "rcencoder.h"
#include "rcmodel.h"

template <typename T, class M = PCmap<T>, bool wide = (M::bits > 8)>
class PCencoder {
public:
  PCencoder(RCencoder* re, RCmodel*const* rm);

  // encode a value with prediction and optional context
  T encode(T real, T pred, uint context = 0);

  // number of symbols (needed by probability modeler)
  static const uint symbols;
};

template <typename U> uint bsr(U x);

#include "pcencoder.inl"

#endif
