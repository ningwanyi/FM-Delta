import numpy as np
import fmd


def test_fmd():
    base = np.array([1.,2.,3.,4.], dtype=np.float16)
    finetuned = np.array([1.1,2.2,3.3,4.4], dtype=np.float16)
    bytes = fmd.compress_param(base,finetuned)
    decompressed_finetuned = fmd.decompress_param(bytes,base)
    print(len(bytes),len(finetuned.tobytes()))
    print("float16:", decompressed_finetuned, finetuned, decompressed_finetuned==finetuned)

    base = np.array([1.,2.,3.,4.], dtype=np.float32)
    finetuned = np.array([1.1,2.2,3.3,4.4], dtype=np.float32)
    bytes = fmd.compress_param(base,finetuned)
    print(len(bytes))
    decompressed_finetuned = fmd.decompress_param(bytes,base)
    print("float32:", decompressed_finetuned, finetuned, decompressed_finetuned==finetuned)
    
    base = np.array([1.2,2.3,3.4,4.5], dtype=np.float64)
    finetuned = np.array([1.,2.,3.,4.], dtype=np.float64)
    bytes = fmd.compress_param(base,finetuned)
    print(len(bytes))
    decompressed_finetuned = fmd.decompress_param(bytes,base)
    print("float64:", decompressed_finetuned, finetuned, decompressed_finetuned==finetuned)


if __name__=='__main__':
    test_fmd()