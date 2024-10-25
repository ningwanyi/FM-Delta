#include <cstdio>
#include <cstdlib>
#include "pcdecoder.h"
#include "rcqsmodel.h"
#include "front.h"
#include "fmd.h"
#include "read.h"

// array meta data and decoder
struct FMDinput : public FMD {
  RCdecoder* rd;
};

// allocate input stream
static FMDinput*
allocate_input()
{
  FMDinput* stream = new FMDinput;
  stream->type = FMD_TYPE_FLOAT;
  stream->nx = stream->ny = stream->nz = stream->nf = 1;
  stream->rd = 0;
  return stream;
}

// decompress 3D array at specified precision using floating-point arithmetic
template <typename T, uint bits>
static void
decompress3d(
  RCdecoder* rd,   // entropy decoder
  T*         data, // flattened 3D array to decompress to
  T*         base_data,
  uint       nx,   // number of x samples
  uint       ny,   // number of y samples
  uint       nz    // number of z samples
)
{
  // initialize decompressor
  typedef PCmap<T, bits> Map;
  RCmodel* rm = new RCqsmodel(false, PCdecoder<T, Map>::symbols);
  PCdecoder<T, Map>* fd = new PCdecoder<T, Map>(rd, &rm);
  Front<T> f(nx, ny);

  // decode difference between predicted (p) and actual (a) value
  uint x, y, z;
  for (z = 0, f.advance(0, 0, 1); z < nz; z++)
    for (y = 0, f.advance(0, 1, 0); y < ny; y++)
      for (x = 0, f.advance(1, 0, 0); x < nx; x++) {
        // volatile 
        T p = *base_data++;
        T a = fd->decode(p);
        *data++ = a;
        f.push(a);
      }

  delete fd;
  delete rm;
}


// decompress p-bit float, 2p-bit double
#define decompress_case(p)\
  case subsize(T, p):\
    decompress3d<T, subsize(T, p)>(stream->rd, data, base_data, stream->nx, stream->ny, stream->nz);\
    break

// decompress 4D array
template <typename T>
static bool
decompress4d(
  FMDinput* stream, // input stream
  T*        data,    // flattened 4D array to decompress to
  T*        base_data
)
{
  // decompress one field at a time
  for (int i = 0; i < stream->nf; i++) {
    int bits = (int)(CHAR_BIT * sizeof(T));
    switch (bits){
      decompress_case( 2);
      decompress_case( 3);
      decompress_case( 4);
      decompress_case( 5);
      decompress_case( 6);
      decompress_case( 7);
      decompress_case( 8);
      decompress_case( 9);
      decompress_case(10);
      decompress_case(11);
      decompress_case(12);
      decompress_case(13);
      decompress_case(14);
      decompress_case(15);
      decompress_case(16);
      default:
        fmd_errno = fmdErrorBadPrecision;
        return false;
    }
    data += stream->nx * stream->ny * stream->nz;
    base_data += stream->nx * stream->ny * stream->nz;
  }
  return true;
}


// read compressed stream from memory buffer
FMD*
fmd_read_from_buffer(
  const void* buffer // pointer to compressed data
)
{
  fmd_errno = fmdSuccess;
  FMDinput* stream = allocate_input();
  stream->rd = new RCmemdecoder(buffer);
  stream->rd->init();
  return static_cast<FMD*>(stream);
}

// close stream for reading and clean up
void
fmd_read_close(
  FMD* fpz // stream handle
)
{
  FMDinput* stream = static_cast<FMDinput*>(fpz);
  delete stream->rd;
  delete stream;
}

// read meta data
int
fmd_read_header(
  FMD* fpz // stream handle
)
{
  fmd_errno = fmdSuccess;

  FMDinput* stream = static_cast<FMDinput*>(fpz);
  RCdecoder* rd = stream->rd;

  // type
  stream->type = rd->decode<uint>(8);

  // array dimensions
  stream->nx = rd->decode<uint>(32);
  stream->ny = rd->decode<uint>(32);
  stream->nz = rd->decode<uint>(32);
  stream->nf = rd->decode<uint>(32);

  return 1;
}

// decompress a single- or double-precision 4D array
size_t
fmd_read(
  FMD*  fpz, // stream handle
  void* data, // array to read
  void* base_data
)
{
  fmd_errno = fmdSuccess;
  size_t bytes = 0;
  try {
    FMDinput* stream = static_cast<FMDinput*>(fpz);
    bool success;
    switch (stream->type) {
      case 0:{
        decompress4d(stream, static_cast<float*>(data), static_cast<float*>(base_data));
        success = true;
        break;}
      case 1:{
        decompress4d(stream, static_cast<double*>(data), static_cast<double*>(base_data));
        success = true;
        break;}
      case 2:{
        decompress4d(stream, static_cast<short*>(data), static_cast<short*>(base_data));
        success = true;
        break;}
      default:{
        success = false;
        break;}
      }
    if (success) {
      RCdecoder* rd = stream->rd;
      if (rd->error) {
        if (fmd_errno == fmdSuccess)
          fmd_errno = fmdErrorReadStream;
      }
      else
        bytes = rd->bytes();
    }
  }
  catch (...) {
    // exceptions indicate unrecoverable internal errors
    fmd_errno = fmdErrorInternal;
  }
  return bytes;
}
