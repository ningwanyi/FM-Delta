#include "fmd.h"

fmdError fmd_errno;

const char* const fmd_errstr[] = {
  "success",
  "cannot read stream",
  "cannot write stream",
  "memory buffer overflow",
  "internal error",
};
