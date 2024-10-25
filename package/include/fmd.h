#ifndef FMD_H
#define FMD_H

#define FMD_TYPE_FLOAT  0 /* single-precision data (see FMD.type) */
#define FMD_TYPE_DOUBLE 1 /* double-precision data */
#define FMD_TYPE_HALF 2

#ifdef __cplusplus
#include <cstdio>
extern "C" {
#else
#include <stdio.h>
#endif

/* array meta data and stream handle */
typedef struct {
  int type; /* single (0) or double (1) or half (2) precision */
  int nx;   /* number of x samples */
  int ny;   /* number of y samples */
  int nz;   /* number of z samples */
  int nf;   /* number of fields */
} FMD;

/* associate memory buffer with compressed input stream */
FMD*                  /* compressed stream */
fmd_read_from_buffer(
  const void* buffer  /* pointer to compressed input data */
);

/* read FMD meta data (use only if previously written) */
int                   /* nonzero upon success */
fmd_read_header(
  FMD* fpz            /* compressed stream */
);

/* decompress array */
size_t                /* number of compressed bytes read (zero = error) */
fmd_read(
  FMD*  fpz,          /* compressed stream */
  void* data,          /* uncompressed floating-point data */
  void* base_data
);

/* close input stream and deallocate fpz */
void
fmd_read_close(
  FMD* fpz            /* compressed stream */
);

/* associate memory buffer with compressed output stream */
FMD*                  /* compressed stream */
fmd_write_to_buffer(
  void*  buffer,      /* pointer to compressed output data */
  size_t size         /* size of allocated storage for buffer */
);

/* write FMD meta data */
int                   /* nonzero upon success */
fmd_write_header(
  FMD* fpz            /* compressed stream */
);

/* 压缩数组，compress array */
size_t                /* number of compressed bytes written (zero = error) */
fmd_write(
  FMD*        fmd,    /* compressed stream */
  const void* base_data,    /* uncompressed floating-point data */
  const void* finetuned_data
);

/* close output stream and deallocate fpz */
void
fmd_write_close(
  FMD* fpz            /* compressed stream */
);

/*
** Error codes.
*/

typedef enum {
  fmdSuccess             = 0, /* no error */
  fmdErrorReadStream     = 1, /* cannot read stream */
  fmdErrorWriteStream    = 2, /* cannot write stream */
  fmdErrorBufferOverflow = 3, /* compressed buffer overflow */
  fmdErrorInternal       = 4,  /* exception thrown */
  fmdErrorBadPrecision   = 5
} fmdError;

extern fmdError fmd_errno; /* error code */
extern const char* const fmd_errstr[]; /* error message indexed by fmd_errno */

#ifdef __cplusplus
}
#endif

#endif
