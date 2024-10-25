#ifndef RC_MODEL_H
#define RC_MODEL_H

#include "types.h"

class RCmodel {
public:
  RCmodel(uint symbols) : symbols(symbols) {}
  virtual ~RCmodel() {}

  virtual void encode(uint s, uint& l, uint& r) = 0;

  // get symbol frequency r and return symbol corresponding to
  // cumulative frequency l
  virtual uint decode(uint& l, uint& r) = 0;

  // divide range r by sum of all frequency counts
  virtual void normalize(uint &r) = 0;

  const uint symbols; // number of symbols
};

#endif
