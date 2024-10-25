template <uint width>
uint32
PCmap<float, width, void>::fcast(float d) const
{
  Range r;
  memcpy(&r, &d, sizeof(r));
  return r;
}

template <uint width>
float
PCmap<float, width, void>::icast(uint32 r) const
{
  Domain d;
  memcpy(&d, &r, sizeof(d));
  return d;
}

template <uint width>
uint32
PCmap<float, width, void>::forward(float d) const
{
  Range r = fcast(d);
  r = ~r;
  r >>= shift;
  r ^= -(r >> (bits - 1)) >> (shift + 1);
  return r;
}

template <uint width>
float
PCmap<float, width, void>::inverse(uint32 r) const
{
  r ^= -(r >> (bits - 1)) >> (shift + 1);
  r = ~r;
  r <<= shift;
  return icast(r);
}

template <uint width>
float
PCmap<float, width, void>::identity(float d) const
{
  Range r = fcast(d);
  r >>= shift;
  r <<= shift;
  return icast(r);
}

template <uint width>
uint64
PCmap<double, width, void>::fcast(double d) const
{
  Range r;
  memcpy(&r, &d, sizeof(r));
  return r;
}

template <uint width>
double
PCmap<double, width, void>::icast(uint64 r) const
{
  Domain d;
  memcpy(&d, &r, sizeof(d));
  return d;
}

template <uint width>
uint64
PCmap<double, width, void>::forward(double d) const
{
  Range r = fcast(d);
  r = ~r;
  r >>= shift;
  r ^= -(r >> (bits - 1)) >> (shift + 1);
  return r;
}

template <uint width>
double
PCmap<double, width, void>::inverse(uint64 r) const
{
  r ^= -(r >> (bits - 1)) >> (shift + 1);
  r = ~r;
  r <<= shift;
  return icast(r);
}

template <uint width>
double
PCmap<double, width, void>::identity(double d) const
{
  Range r = fcast(d);
  r >>= shift;
  r <<= shift;
  return icast(r);
}

// for short type
template <uint width>
uint16
PCmap<short, width, void>::fcast(short d) const
{
  Range r;
  memcpy(&r, &d, sizeof(r));
  return r;
}

template <uint width>
short
PCmap<short, width, void>::icast(uint16 r) const
{
  Domain d;
  memcpy(&d, &r, sizeof(d));
  return d;
}

template <uint width>
uint16
PCmap<short, width, void>::forward(short d) const
{
  Range r = fcast(d);
  r = ~r;
  r >>= shift;
  Range move = -(r >> (bits - 1));
  move >>= (shift + 1);
  r ^= move;
  return r;
}

template <uint width>
short
PCmap<short, width, void>::inverse(uint16 r) const
{
  Range move = -(r >> (bits - 1));
  move >>= (shift + 1);
  r ^= move;
  r = ~r;
  r <<= shift;
  return icast(r);
}

template <uint width>
short
PCmap<short, width, void>::identity(short d) const
{
  Range r = fcast(d);
  r >>= shift;
  r <<= shift;
  return icast(r);
}
