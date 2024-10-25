import argparse
from utils.compressing import compress, decompress
import pickle
import fmd
import torch
from utils.models import read_models
import copy
import pickle
import time
import os


def model_compress(args, base_model, finetuned_model):
    if args.compressor == 'fmd':
        compressed_finetuned_model = fmd.compress(base_model, finetuned_model, save_path=args.save_path) if args.save_with_chunk else fmd.compress(base_model, finetuned_model) 
    else:
        compressed_finetuned_model,_,_ = compress(base_model, finetuned_model, args, args.compressor)
    return compressed_finetuned_model

def model_decompress(args, compressed_finetuned_model, base_model):
    decompressed_finetuned_model = copy.deepcopy(base_model)
    if args.compressor == 'fmd':
        decompressed_model_dict = fmd.decompress(base_model=base_model, save_path=args.save_path) if args.save_with_chunk else fmd.decompress(compressed_finetuned_model, base_model)
    else:
        decompressed_model_dict = decompress(compressed_finetuned_model, args.compressor, args.dtype)
    for n,p in decompressed_finetuned_model.named_parameters():
        p.data = decompressed_model_dict[n].clone()
    return decompressed_finetuned_model

def check_equal(finetuned_model, decompressed_finetuned_model):
    for p1,p2 in zip(finetuned_model.parameters(), decompressed_finetuned_model.parameters()):
        assert p1.equal(p2), "quit checking: finetuned_model != decompressed_finetuned_model."
    print("finish checking: finetuned_model == decompressed_finetuned_model.")


def main(args):
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dtype_dict = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'fp64': torch.float64,
    }
    base_model, finetuned_model = read_models(args.base_model, args.finetuned_model, dtype=dtype_dict[args.dtype])

    # compress
    tick = time.time()
    compressed_finetuned_model = model_compress(args, base_model, finetuned_model)
    compress_time = time.time()-tick
    print('> compress_time:', compress_time)

    # decompress
    tick = time.time()
    decompressed_finetuned_model = model_decompress(args, compressed_finetuned_model, base_model)
    decompress_time = time.time()-tick
    print('> decompress_time:', decompress_time)
    comp_rate = len(pickle.dumps(compressed_finetuned_model))/len(pickle.dumps(finetuned_model))
    print('> compression rate:', comp_rate)
    check_equal(finetuned_model, decompressed_finetuned_model)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fine-tuned model compression.')
    parser.add_argument('--compressor', type=str, default='fmd', metavar='S',
                        help='option: [lzma, bz2, gzip, zlib, fpzip, fmd]')
    parser.add_argument('--base-model', type=str, default='bert-large-uncased', metavar='S',
                        help='pretrained model name')
    parser.add_argument('--finetuned-model', type=str, default='Jorgeutd/bert-large-uncased-finetuned-ner', metavar='S',
                        help='finetuned model name')
    parser.add_argument('--dtype', type=str, default='fp32', metavar='S',
                        help='finetuned model name')
    parser.add_argument('--stdout', type=str, default='stdout')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--save_with_chunk', action='store_true', help='use with --save_path to save fmdelta compressed model with chunk')

    args = parser.parse_args()
    print(args)
    main(args)
