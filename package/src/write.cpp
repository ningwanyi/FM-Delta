#include <cstdio>
#include <cstdlib>
#include "pcencoder.h"
#include "rcqsmodel.h"
#include "front.h"
#include "fmd.h"
#include "write.h"

// array meta data and encoder
struct FMDoutput : public FMD {
  RCencoder* re;  // range coding encoder
};

// allocate output stream
static FMDoutput*
allocate_output()
{
  FMDoutput* stream = new FMDoutput;
  stream->type = FMD_TYPE_FLOAT;
  stream->nx = stream->ny = stream->nz = stream->nf = 1;
  stream->re = 0;
  return stream;
}

// compress 3D array at specified precision using floating-point arithmetic
template <typename T, uint bits>
static void
compress3d(
  RCencoder* re,   // entropy encoder
  const T*   base_data, // flattened 3D array to compress
  const T*   finetuned_data,
  uint       nx,   // number of x samples
  uint       ny,   // number of y samples
  uint       nz    // number of z samples
)
{
  // initialize compressor
  typedef PCmap<T, bits> Map;
  RCmodel* rm = new RCqsmodel(true, PCencoder<T, Map>::symbols);  
  PCencoder<T, Map>* fe = new PCencoder<T, Map>(re, &rm); 
  Front<T> f(nx, ny);

  //encode difference between base and finetuned value
  uint x, y, z;
  // uint d;
  for (z = 0, f.advance(0, 0, 1); z < nz; z++)
    for (y = 0, f.advance(0, 1, 0); y < ny; y++)
      for (x = 0, f.advance(1, 0, 0); x < nx; x++) {
        T base = *base_data++;
        T finetuned = *finetuned_data++;
        finetuned = fe->encode(base,finetuned);
        f.push(finetuned);
      }

  delete fe;
  delete rm;
}

// compress p-bit float, 2p-bit double
#define compress_case(p)\
  case subsize(T, p):\
    compress3d<T, subsize(T, p)>(stream->re, base_data, finetuned_data, stream->nx, stream->ny, stream->nz);\
    break

// compress 4D array
template <typename T>
static bool
compress4d(
  FMDoutput* stream, // output stream
  const T*   base_data,    // flattened 4D array to compress
  const T*   finetuned_data
)
{
  // compress one field at a time
  for (int i = 0; i < stream->nf; i++) {
    int bits = (int)(CHAR_BIT * sizeof(T));
    switch (bits) {
      compress_case( 2);
      compress_case( 3);
      compress_case( 4);
      compress_case( 5);
      compress_case( 6);
      compress_case( 7);
      compress_case( 8);
      compress_case( 9);
      compress_case(10);
      compress_case(11);
      compress_case(12);
      compress_case(13);
      compress_case(14);
      compress_case(15);
      compress_case(16);
      default:
        fmd_errno = fmdErrorBadPrecision;
        return false;
    }
    base_data += stream->nx * stream->ny * stream->nz;
    finetuned_data += stream->nx * stream->ny * stream->nz;
  }
  return true;
}

// write compressed stream to memory buffer
FMD*
fmd_write_to_buffer(
  void*  buffer, // pointer to compressed data
  size_t size    // size of buffer
)
{
  fmd_errno = fmdSuccess;
  FMDoutput* stream = allocate_output();   
  stream->re = new RCmemencoder(buffer, size);  
  return static_cast<FMD*>(stream);
}

// close stream for writing and clean up
void
fmd_write_close(
  FMD* fpz // stream handle
)
{
  FMDoutput* stream = static_cast<FMDoutput*>(fpz);
  delete stream->re;
  delete stream;
}

// write meta data
int
fmd_write_header(
  FMD* fpz // stream handle
)
{
  fmd_errno = fmdSuccess;

  FMDoutput* stream = static_cast<FMDoutput*>(fpz);
  RCencoder* re = stream->re;
  
  // type
  re->encode<uint>(stream->type, 8);

  re->encode<uint>(stream->nx, 32);
  re->encode<uint>(stream->ny, 32);
  re->encode<uint>(stream->nz, 32);
  re->encode<uint>(stream->nf, 32);

  if (re->error) {
    fmd_errno = fmdErrorWriteStream;
    return 0;
  }

  return 1;
}

// compress a single- or double-precision 4D array
size_t
fmd_write(
  FMD*        fpz, // stream handle
  const void* base_data, // array to write
  const void* finetuned_data // array to write
)
{
  fmd_errno = fmdSuccess;
  size_t bytes = 0;
  try {
    FMDoutput* stream = static_cast<FMDoutput*>(fpz);
    bool success;
    switch (stream->type) {
      case 0:{
        compress4d(stream, static_cast<const float*>(base_data), static_cast<const float*>(finetuned_data));
        success = true;
        break;}
      case 1:{
        compress4d(stream, static_cast<const double*>(base_data), static_cast<const double*>(finetuned_data));
        success = true;
        break;}
      case 2:{
        compress4d(stream, static_cast<const short*>(base_data), static_cast<const short*>(finetuned_data));
        success = true;
        break;}
      default:{
        success = false;
        break;}
      }
    if (success) {
      RCencoder* re = stream->re;
      re->finish();
      if (re->error) {
        if (fmd_errno == fmdSuccess)
          fmd_errno = fmdErrorWriteStream;
      }
      else
        bytes = re->bytes();
    }
  }
  catch (...) {
    // exceptions indicate unrecoverable internal errors
    fmd_errno = fmdErrorInternal;
  }
  return bytes;
}
