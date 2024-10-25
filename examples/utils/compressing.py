from ctypes import *
import bz2
import lzma
import gzip, zlib
import pickle
import fmd
import fpzip
import torch

def decompress(compressed_finetuned_model_dict, compressor, dtype):
    decompressed_finetuned_model_dict = {}
    # iterate model layer tensors
    for name, comp_bytes_shape in compressed_finetuned_model_dict.items():
        comp_bytes, p_shape = comp_bytes_shape
        if compressor == 'fpzip':
            if isinstance(comp_bytes,torch.Tensor):
                decompressed_finetuned_model_dict[name] = comp_bytes
            else:
                comp_p_np_flatten = general_decompress(comp_bytes, compressor) 
                decompressed_finetuned_model_dict[name] = torch.from_numpy(comp_p_np_flatten).reshape(p_shape)
        else:
            decomp_bytes = general_decompress(comp_bytes, compressor) 
            decompressed_finetuned_model_dict[name] = (pickle.loads(decomp_bytes)).reshape(p_shape)
    return decompressed_finetuned_model_dict

def general_decompress(comp_bytes, compressor):
    if compressor=='lzma':
        decomp_param = lzma.decompress(comp_bytes)
    elif compressor=='gzip':
        decomp_param = gzip.decompress(comp_bytes)
    elif compressor=='zlib':
        decomp_param = zlib.decompress(comp_bytes)
    elif compressor=='fpzip':
        decomp_param = fpzip.decompress(comp_bytes)
    elif compressor=='bz2':
        decomp_param = bz2.decompress(comp_bytes)
    return decomp_param

def compress(base_model, finetuned_model, args, compressor):
    compressed_finetuned_model_dict = {}
    total_len, total_compress_len = 0, 0
    # iterate model layer tensors
    for name, finetuned_param in finetuned_model.named_parameters():
        param_shape = finetuned_param.shape
        compressed_bytes = general_compress(finetuned_param, compressor)
        compressed_finetuned_model_dict[name] = (compressed_bytes, param_shape)
        total_compress_len += len(compressed_bytes)
        total_len+=len(pickle.dumps(finetuned_param))
    return compressed_finetuned_model_dict, total_compress_len/(1024**3), total_compress_len/total_len

def general_compress(param, compressor):
    if compressor=='lzma':
        compressed_bytes = lzma.compress(pickle.dumps(param))
    elif compressor=='gzip':
        compressed_bytes = gzip.compress(pickle.dumps(param))
    elif compressor=='zlib':
        compressed_bytes = zlib.compress(pickle.dumps(param))
    elif compressor=='fpzip':
        try:
            compressed_bytes = fpzip.compress(param.detach().numpy().flatten())
        except:
            print("memorybuffer overflow, continue.")
            compressed_bytes = param
    elif compressor=='bz2':
        compressed_bytes = bz2.compress(pickle.dumps(param))
    return compressed_bytes

