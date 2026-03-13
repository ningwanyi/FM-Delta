template <typename T, class M>
T PCdecoder<T, M>::decode(T pred)
{
  typedef typename M::Range U;
  uint s = rd->decode(rm);
  if (s > bias) {      // underprediction
    uint k = s - bias - 1;
    U d = (U(1) << k) + rd->template decode<U>(k);
    U p = map.forward(pred);
    U r = p + d;
    return map.inverse(r);
  }
  else if (s < bias) { // overprediction
    uint k = bias - 1 - s;
    U d = (U(1) << k) + rd->template decode<U>(k);
    U p = map.forward(pred);
    U r = p - d;
    return map.inverse(r);
  }
  else                 // perfect prediction
    return map.identity(pred);
}
