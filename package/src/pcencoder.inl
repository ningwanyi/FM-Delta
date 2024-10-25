// specialization for small alphabets -----------------------------------------

template <typename T, class M>
class PCencoder<T, M, false> {
public:
  PCencoder(RCencoder* re, RCmodel*const* rm) : re(re), rm(rm) {} 
  T encode(T base_data, T finetuned_data, uint context = 0);
  static const uint symbols = 2 * (1 << M::bits) - 1; //bits=32
private:
  static const uint bias = (1 << M::bits) - 1; // perfect prediction symbol
  M                 map;                       // maps T to integer type
  RCencoder*const   re;                        // entropy encoder
  RCmodel*const*    rm;                        // probability modeler(s)
};

// encode narrow range type
template <typename T, class M>
T PCencoder<T, M, false>::encode(T base, T finetuned, uint context)
{
  // map type T to unsigned integer type
  typedef typename M::Range U;
  U r = map.forward(base);
  U p = map.forward(finetuned);

  re->encode(static_cast<uint>(bias + r - p), rm[context]);
  return r;
}

// for float and double -----------------------------------------

template <typename T, class M>
class PCencoder<T, M, true> {
public:
  PCencoder(RCencoder* re, RCmodel*const* rm) : re(re), rm(rm) {}   // rm: RCqsmodel; re: RCencoder
  T encode(T base, T finetuned, uint context = 0);
  static const uint symbols = 2 * M::bits + 1;
private:
  static const uint bias = M::bits; // perfect prediction symbol
  M                 map;            // maps T to integer type
  RCencoder*const   re;             // entropy encoder
  RCmodel*const*    rm;             // probability modeler(s)
};

// encode wide range type
template <typename T, class M>
T PCencoder<T, M, true>::encode(T base, T finetuned, uint context)
{
  // map type T to unsigned integer type
  typedef typename M::Range U;
  U p = map.forward(base);     
  U r = map.forward(finetuned);
  
  if (p < r) {      // underprediction
    U d = r - p;
    uint k = bsr(d);  
    re->encode(bias + 1 + k, rm[context]);  
    re->encode(d - (U(1) << k), k);
  }
  else if (p > r) { // overprediction
    U d = p - r;
    uint k = bsr(d);
    re->encode(bias - 1 - k, rm[context]);
    re->encode(d - (U(1) << k), k);
  }
  else             // perfect prediction
    re->encode(bias, rm[context]); 
  return map.inverse(r);
}

template <typename U>
uint bsr(U x)
{
  uint k;
  k = 0;
  do k++; while (x >>= 1);
  k--;
  return k;
}
