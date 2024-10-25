from cpython cimport array 
import array
import sys
import torch
cimport numpy as numpy
from tqdm.contrib import tzip
from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint8_t
import numpy as np
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import os
import pickle

__VERSION__ = '0.0'
__version__ = __VERSION__

FMD_ERROR_STRINGS = [
  "success",
  "cannot read stream",
  "cannot write stream",
  "memory buffer overflow",
  "internal error",
  "precision not supported"
]

cdef extern from "fmd.h":
  ctypedef struct FMD:
    int type
    int nx
    int ny
    int nz
    int nf

  cdef FMD* fmd_read_from_buffer(void* buffer) 
  cdef int fmd_read_header(FMD* fpz)
  cdef size_t fmd_read(FMD* fpz, void* data, void* base_data)
  cdef void fmd_read_close(FMD* fpz)
  
  cdef FMD* fmd_write_to_buffer(void* buffer, size_t size)
  cdef int fmd_write_header(FMD* fpz)
  cdef int fmd_write(FMD* fpz, const void* base_data, const void* finetuned_data)
  cdef void fmd_write_close(FMD* fpz)

  ctypedef enum fmdError:
    fmdSuccess             = 0, # no error 
    fmdErrorReadStream     = 1, # cannot read stream 
    fmdErrorWriteStream    = 2, # cannot write stream 
    fmdErrorBufferOverflow = 3,  # compressed buffer overflow 
    fmdErrorInternal       = 4,
    fmdErrorBadPrecision   = 5

  cdef fmdError fmd_errno = 0

class FmdError(Exception):
  pass

class FmdWriteError(FmdError):
  pass

class FmdReadError(FmdError):
  pass

cpdef allocate(typecode, ct):
  cdef array.array array_template = array.array(chr(typecode), [])
  # create an array with 3 elements with same type as template
  return array.clone(array_template, ct, zero=True)

def validate_order(order):
  order = order.upper()
  if order not in ('C', 'F'):
    raise ValueError("Undefined order parameter '{}'. Options are 'C' or 'F'".format(order))
  return order


def compress(base_model, finetuned_model, save_path=None, order='C'):
    """
    Compresses the parameters of a fine-tuned model relative to a base model.

    This function iterates over the parameters of both the base and fine-tuned models, 
    compresses the fine-tuned parameters, and stores them in a dictionary. It also 
    handles optional saving of compressed chunks to disk if specified.

    Parameters:
    - base_model (torch.nn.Module): The base model containing the original parameters.
    - finetuned_model (torch.nn.Module): The fine-tuned model with updated parameters.
    - save_path (str, optional): Directory path to save compressed parameter chunks. 
                                  If None, the compressed data is not saved to disk.
    - order (str): Memory layout order for the input data ('C' for row-major or 'F' for column-major).

    Returns:
    - dict: A dictionary containing the compressed parameters of the fine-tuned model.
    """

    compressed_finetuned_model_dict = {}
    sum_bytes_mb = 0  # Total size of compressed data in megabytes
    chunk_count = 0  # Count of chunks saved to disk

    # Iterate through the parameters of both models
    for (base_items, finetuned_items) in tzip(base_model.named_parameters(), finetuned_model.named_parameters()):
        base_layer, base_param = base_items
        finetuned_layer, finetuned_param = finetuned_items

        # Ensure that the layer names match between the two models
        assert base_layer == finetuned_layer, 'The layer in the base model is different from the fine-tuned model!'

        # Check if the shapes of the parameters match
        if base_param.shape != finetuned_param.shape:
            print('Shape mismatch!')  # Notify of shape mismatch
            compressed_finetuned_model_dict[finetuned_layer] = finetuned_param.detach().numpy()  # Store raw fine-tuned parameter
            continue

        # Convert tensors to numpy arrays with shared memory for efficiency
        base_param_np = base_param.detach().numpy()
        finetuned_param_np = finetuned_param.detach().numpy()

        # Compress the fine-tuned parameters using the base parameters
        compressed_bytes = compress_param(base_param_np, finetuned_param_np, order)
        compressed_finetuned_model_dict[finetuned_layer] = compressed_bytes  # Store compressed data

        # Accumulate the size of compressed data in megabytes
        sum_bytes_mb += len(compressed_bytes) / 1024 / 1024

        # Save chunks to disk if specified and the accumulated size exceeds 200 MB
        if save_path is not None and sum_bytes_mb > 200:
            with open(os.path.join(save_path, f"chunk{chunk_count}.pkl"), 'wb') as f:
                pickle.dump(compressed_finetuned_model_dict, f)  # Save the compressed chunk
            sum_bytes_mb = 0  # Reset accumulated size
            chunk_count += 1  # Increment chunk counter
            compressed_finetuned_model_dict = {}  # Clear the dictionary for the next chunk
    
    # Save any remaining compressed parameters as the last chunk
    if save_path is not None and len(compressed_finetuned_model_dict) > 0:
        with open(os.path.join(save_path, f"chunk{chunk_count}.pkl"), 'wb') as f:
            pickle.dump(compressed_finetuned_model_dict, f)

    return compressed_finetuned_model_dict  # Return the dictionary of compressed parameters


def compress_param(base_data, finetuned_data, order='C'):
    """
    FMD Compression API Entry Point.

    This function compresses a numpy array of floats or doubles (up to 4 dimensions) 
    and returns the compressed data as a byte string. 

    Parameters:
    - base_data (np.ndarray): The base model parameters to be compressed.
    - finetuned_data (np.ndarray): The fine-tuned model parameters to be compressed.
    - order (str): Memory layout order for the input array. 
                   Acceptable values are 'C' (row-major) or 'F' (column-major), 
                   which should match the underlying array's orientation.

    Returns:
    - bytes: Compressed data in bytes format.

    Raises:
    - ValueError: If the data type of base_data is not a floating-point type.
    - FmdWriteError: If there is an error during the header writing or compression process.
    """

    if base_data.dtype not in (np.float32, np.float64, np.float16):
        raise ValueError("base_data type {} must be a floating type.".format(base_data.dtype))
  
    # If the data type is float16, first convert it into bytes, then to int16 for compression.
    if base_data.dtype == np.float16:
        base_data_bytes = base_data.tobytes()
        base_data = np.frombuffer(base_data_bytes, dtype=np.int16).copy()
        finetuned_data_bytes = finetuned_data.tobytes()
        finetuned_data = np.frombuffer(finetuned_data_bytes, dtype=np.int16).copy()

    order = validate_order(order)

    # Ensure base_data and finetuned_data are at least 4D, adding new axes as needed.
    while len(base_data.shape) < 4:
        if order == 'C':
            base_data = base_data[np.newaxis, ...]
            finetuned_data = finetuned_data[np.newaxis, ...]
        else:  # F
            base_data = base_data[..., np.newaxis]
            finetuned_data = finetuned_data[..., np.newaxis]

    # Create copies in the specified order if the arrays are not contiguous.
    if not base_data.flags['C_CONTIGUOUS'] and not base_data.flags['F_CONTIGUOUS']:
        base_data = np.copy(base_data, order=order)
        finetuned_data = np.copy(finetuned_data, order=order)

    header_bytes = 28  # Size for header, plus additional bytes.

    # Determine the floating point type for compression.
    cdef char fptype
    if base_data.dtype == np.float32:
        fptype = b'f'
    elif base_data.dtype == np.float64:
        fptype = b'd'
    elif base_data.dtype == np.int16:
        fptype = b'h'
    
    # Allocate a buffer for compression.
    cdef array.array compression_buf = allocate(fptype, base_data.size + header_bytes)

    cdef FMD* fpz_ptr
    # Write the base data to the compression buffer based on the floating point type.
    if fptype == b'f':
        fpz_ptr = fmd_write_to_buffer(compression_buf.data.as_floats, base_data.nbytes + header_bytes)
    elif fptype == b'h':
        fpz_ptr = fmd_write_to_buffer(compression_buf.data.as_shorts, base_data.nbytes + header_bytes)
    else:
        fpz_ptr = fmd_write_to_buffer(compression_buf.data.as_doubles, base_data.nbytes + header_bytes)

    # Set the type in the header.
    if base_data.dtype == np.float32:
        fpz_ptr[0].type = 0  # Type: float
    elif base_data.dtype == np.float64:
        fpz_ptr[0].type = 1  # Type: double
    else:
        fpz_ptr[0].type = 2  # Type: half

    shape = list(base_data.shape)

    # Reverse the shape if the order is 'C' to accommodate memory layout.
    if order == 'C':
        shape.reverse()

    # Assign shape values to the header.
    fpz_ptr[0].nx = shape[0]
    fpz_ptr[0].ny = shape[1]
    fpz_ptr[0].nz = shape[2]
    fpz_ptr[0].nf = shape[3]

    # Write the header and check for errors.
    if fmd_write_header(fpz_ptr) == 0:
        fmd_write_close(fpz_ptr)
        del compression_buf
        raise FmdWriteError("Cannot write header. %s" % FMD_ERROR_STRINGS[fmd_errno])

    # Prepare memory views for data writing.
    cdef float[:,:,:,:] arr_memviewf_base
    cdef float[:,:,:,:] arr_memviewf_finetuned
    cdef double[:,:,:,:] arr_memviewd_base
    cdef double[:,:,:,:] arr_memviewd_finetuned
    cdef short[:,:,:,:] arr_memviews_base
    cdef short[:,:,:,:] arr_memviews_finetuned
    cdef size_t outbytes

    cdef float[:] bufviewf
    cdef double[:] bufviewd
    cdef short[:] bufviews

    # Handle the case where base_data is empty.
    if base_data.size == 0:
        fmd_write_close(fpz_ptr)
        if base_data.dtype == np.float32:
            bufviewf = compression_buf
            bytes_out = bytearray(bufviewf[:header_bytes])
        elif base_data.dtype == np.int16:
            bufviews = compression_buf
            bytes_out = bytearray(bufviews[:header_bytes])
        else:
            bufviewd = compression_buf
            bytes_out = bytearray(bufviewd[:header_bytes])
        del compression_buf
        return bytes(bytes_out)
  
    # Perform the actual writing of base and fine-tuned data to the compression buffer.
    if base_data.dtype == np.float32:
        arr_memviewf_base = base_data
        arr_memviewf_finetuned = finetuned_data
        outbytes = fmd_write(fpz_ptr, <void*>&arr_memviewf_base[0,0,0,0], <void*>&arr_memviewf_finetuned[0,0,0,0])
        bufviewf = compression_buf
        bytes_out = bytearray(bufviewf[:outbytes])[:outbytes]
    elif base_data.dtype == np.int16:
        arr_memviews_base = base_data
        arr_memviews_finetuned = finetuned_data
        outbytes = fmd_write(fpz_ptr, <void*>&arr_memviews_base[0,0,0,0], <void*>&arr_memviews_finetuned[0,0,0,0])
        bufviews = compression_buf
        bytes_out = bytearray(bufviews[:outbytes])[:outbytes]
    else:  # float64
        arr_memviewd_base = base_data
        arr_memviewd_finetuned = finetuned_data
        outbytes = fmd_write(fpz_ptr, <void*>&arr_memviewd_base[0,0,0,0], <void*>&arr_memviewd_finetuned[0,0,0,0])
        bufviewd = compression_buf
        bytes_out = bytearray(bufviewd[:outbytes])[:outbytes]

    # Clean up and finalize compression.
    del compression_buf
    fmd_write_close(fpz_ptr)
  
    if outbytes == 0:
        raise FmdWriteError("Compression failed. %s" % FMD_ERROR_STRINGS[fmd_errno])

    return bytes(bytes_out)


def decompress_param(bytes encoded, base_data, order='C'):
  """
  fmd.decompress(encoded, order='C')

  Accepts an fmd encoded bytestring (e.g. b'fpy)....') and 
  returns the original array as a 4d numpy array.

  order is 'C' or 'F' (row major vs column major memory layout) and 
  should correspond to the byte order of the originally compressed
  array.
  """

  if base_data.dtype not in (np.float32, np.float64, np.float16):
    raise ValueError("base_data type {} must be a floating type.".format(base_data.dtype))

  order = validate_order(order)

  if base_data.dtype == np.float16:
    base_data_bytes = base_data.tobytes()
    base_data = np.frombuffer(base_data_bytes, dtype=np.int16).copy()

  while len(base_data.shape) < 4:
    if order == 'C':
      base_data = base_data[np.newaxis, ...]
    else: # F
      base_data = base_data[..., np.newaxis ]

  if not base_data.flags['C_CONTIGUOUS'] and not base_data.flags['F_CONTIGUOUS']:
    base_data = np.copy(base_data, order=order)

  cdef float[:,:,:,:] arr_memviewf_base
  cdef double[:,:,:,:] arr_memviewd_base
  cdef short[:,:,:,:] arr_memviews_base
  if base_data.dtype == np.float32:
    arr_memviewf_base = base_data
  elif base_data.dtype == np.float64:
    arr_memviewd_base = base_data
  else:
    arr_memviews_base = base_data

  # line below necessary to convert from PyObject to a naked pointer
  cdef unsigned char *encodedptr = <unsigned char*>encoded 
  cdef FMD* fpz_ptr = fmd_read_from_buffer(<void*>encodedptr)

  if fmd_read_header(fpz_ptr) == 0:
    raise FmdReadError("cannot read header: %s" % FMD_ERROR_STRINGS[fmd_errno])

  cdef char fptype
  if fpz_ptr[0].type == 0:
    fptype = b'f'
  elif fpz_ptr[0].type == 1:
    fptype = b'd'
  else:
    fptype = b'h'
  nx, ny, nz, nf = fpz_ptr[0].nx, fpz_ptr[0].ny, fpz_ptr[0].nz, fpz_ptr[0].nf

  cdef array.array buf = allocate(fptype, nx * ny * nz * nf)

  cdef size_t read_bytes = 0;
  if fptype == b'f':
    read_bytes = fmd_read(fpz_ptr, buf.data.as_floats, <void*>&arr_memviewf_base[0,0,0,0])
  elif fptype == b'd':
    read_bytes = fmd_read(fpz_ptr, buf.data.as_doubles, <void*>&arr_memviewd_base[0,0,0,0])
  else:
    read_bytes = fmd_read(fpz_ptr, buf.data.as_shorts, <void*>&arr_memviews_base[0,0,0,0])

  if read_bytes == 0:
    raise FmdReadError("decompression failed: %s" % FMD_ERROR_STRINGS[fmd_errno])

  fmd_read_close(fpz_ptr)

  dtype = np.float32 
  if fptype == b'f':
    dtype = np.float32 
  elif fptype == b'd':
    dtype = np.float64
  else:
    dtype = np.float16

  if order == 'C':
    return np.frombuffer(buf, dtype=dtype).reshape( (nf, nz, ny, nx), order='C')
  elif order == 'F':
    return np.frombuffer(buf, dtype=dtype).reshape( (nx, ny, nz, nf), order='F')
  else:
    raise ValueError(f"Undefined order parameter '{order}'. Options are 'C' or 'F'")


def decompress(compressed_dict={}, base_model=None, save_path=None, order='C'):
  """
  Decompresses the compressed_dict using the base_model's structure. 
  If save_path is provided, loads chunks from that directory.
  
  Parameters:
  - compressed_dict (dict): In-memory compressed model dictionary
  - base_model (torch.nn.Module): Base model for structure and shape information
  - save_path (str, optional): Directory path to load compressed model chunks
  - order (str): Memory layout order ('C' or 'F') for decompression
  
  Returns:
  - dict: Decompressed model parameters as a dictionary
  """

  # Dictionary to hold decompressed model parameters
  decompressed_model_dict = {}
  base_mode_named_parameters_dict = dict(base_model.named_parameters())

  # In-memory decompression if compressed_dict is not empty
  if compressed_dict != {}:
    for name, base_p in base_mode_named_parameters_dict.items():
      if name in compressed_dict:
        if isinstance(compressed_dict[name], bytes):
          base_p_np = base_p.detach().numpy()
          decompressed_p_np = decompress_param(compressed_dict[name], base_p_np, order=order)
          decompressed_p_np = np.squeeze(decompressed_p_np)  # Remove any extra dimensions
          decompressed_param = torch.from_numpy(decompressed_p_np).reshape(base_p.shape)
          decompressed_model_dict[name] = decompressed_param
        elif isinstance(compressed_dict[name], np.ndarray):
          decompressed_model_dict[name] = torch.from_numpy(compressed_dict[name])
      else:
        raise ValueError(f"Layer '{name}' not found in the provided compressed dictionary.")
    return decompressed_model_dict

  # Incrementally decompress using chunk files if save_path is provided
  elif save_path is not None:
    chunk_count = 0
    while True:
      chunk_path = os.path.join(save_path, f"chunk{chunk_count}.pkl")
      if not os.path.exists(chunk_path):
        break  # Exit loop if no more chunk files

      # Load and process each chunk
      with open(chunk_path, 'rb') as f:
        chunk_dict = pickle.load(f)

      # Decompress parameters in this chunk
      for name, compressed_param in chunk_dict.items():
        base_p = base_mode_named_parameters_dict.get(name)
        if base_p is None:
          raise ValueError(f"Layer '{name}' in chunk file not found in base model structure.")

        if isinstance(compressed_param, bytes):
          base_p_np = base_p.detach().numpy()
          decompressed_p_np = decompress_param(compressed_param, base_p_np, order=order)
          decompressed_p_np = np.squeeze(decompressed_p_np)
          decompressed_param = torch.from_numpy(decompressed_p_np).reshape(base_p.shape)
          decompressed_model_dict[name] = decompressed_param
        elif isinstance(compressed_param, np.ndarray):
          decompressed_model_dict[name] = torch.from_numpy(compressed_param)

      # Move to the next chunk
      chunk_count += 1

    return decompressed_model_dict

  else:
    raise ValueError("Either compressed_dict or save_path must be provided.")
