#ifndef FMD_TYPES_H
#define FMD_TYPES_H

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

// C++11 and later: use standard integer types
#include <cstdint>
#include <cinttypes>
#define INT64C(x) INT64_C(x)
#define UINT64C(x) UINT64_C(x)
#define INT64PRId PRId64
#define INT64PRIi PRIi64
#define UINT64PRIo PRIo64
#define UINT64PRIu PRIu64
#define UINT64PRIx PRIx64
#define INT64SCNd SCNd64
#define INT64SCNi SCNi64
#define UINT64SCNo SCNo64
#define UINT64SCNu SCNu64
#define UINT64SCNx SCNx64
typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;
#endif
