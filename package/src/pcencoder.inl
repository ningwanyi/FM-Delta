template <typename T, class M>
void PCencoder<T, M>::encode(T base, T finetuned)
{
  // map type T to unsigned integer type
  typedef typename M::Range U;
  U p = map.forward(base);          // map base value to unsigned integer
  U r = map.forward(finetuned);     // map finetuned value to unsigned integer
  
  if (p < r) {      // underprediction
    U d = r - p;
    uint k = bsr(d);  // k \in [0, bits-1]
    re->encode(bias + 1 + k, rm);
    re->encode(d - (U(1) << k), k);
  }
  else if (p > r) { // overprediction
    U d = p - r;
    uint k = bsr(d);
    re->encode(bias - 1 - k, rm);
    re->encode(d - (U(1) << k), k);
  }
  else             // perfect prediction
    re->encode(bias, rm);
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
