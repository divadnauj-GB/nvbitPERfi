import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.common_types as ct
import torchvision
import numpy as np
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import os, sys
import nvbitfi_DNN
import struct
import argparse
import h5py


def float_view_hex(f):
    h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
    return h[2:len(h)]

def float_view_int(f):
    h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
    return int(h[2:len(h)],16)


def hex_view_float(h):
    return float(struct.unpack(">f",struct.pack(">I",int(h,16)))[0])


def int_view_float(h):
    return float(struct.unpack(">f",struct.pack(">I",h))[0])


def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('-ln','--layer_number', required=True, type=int, help='golden')
    parser.add_argument('-bs','--batch_size', required=True, type=int, help='golden')
    parser.add_argument('-v','--verbose', required=False, type=int, help='golden')
    return parser

def main(args):
    layer_number = args.layer_number
    batch_size = args.batch_size
    verbose = args.verbose
    #current_path = os.getcwd()
    current_path = os.path.dirname(__file__)

    dataset_file = os.path.join(
        current_path,
        f"Golden_Output_layer.h5")

    with h5py.File(dataset_file, "r") as hf:
        Output_dataset = np.array(hf["layer_output"])

    dataset_file = os.path.join(
        current_path,
        f"Output_layer.h5")
    
    if os.path.isfile(dataset_file):
        with h5py.File(dataset_file, "r") as hf:
            results_dataset = np.array(hf["layer_output"])


        #print(len(np.reshape(Output_dataset,(-1))))
        #print(len(np.reshape(results_dataset,(-1)))) 

        #print(np.not_equal(np.reshape(Output_dataset,(-1)),np.reshape(results_dataset,(-1))))

        #results_dataset[10,0,0,0]=1.0
        #results_dataset[20,0,2,0]=1.0
        #results_dataset[256,1,3,0]=1.0
        if verbose: print(len(Output_dataset))
        if verbose: print(Output_dataset.shape)
        if verbose: print(len(results_dataset))
        if verbose: print(results_dataset.shape)
        if(len(results_dataset)!=len(Output_dataset)):
            print(f"ERROR: Faulty != Golden \n Faulty: {len(results_dataset)}; Golden: {len(Output_dataset)}")
        
        max_batches = float(float(len(results_dataset)) / float(batch_size))
        for batch in range(0, int(np.ceil(max_batches))):
            tensor_golden = Output_dataset[
                batch * batch_size : batch * batch_size + batch_size
            ]
            tensor_faulty = results_dataset[
                batch * batch_size : batch * batch_size + batch_size
            ]

            cmp_result = np.not_equal(tensor_golden,tensor_faulty)

            sdc_index=np.argwhere(cmp_result)
            if len(sdc_index)>0:
                print(Output_dataset[0].shape)
                
            for k in sdc_index:
                original_value=tensor_golden[tuple(k)]
                faulty_value=tensor_faulty[tuple(k)]
                int_view_original_val = float_view_int(original_value)
                int_view_faulty_val = float_view_int(faulty_value)
                xor_result = (int_view_original_val) ^ (int_view_faulty_val)
                sdc_line = f"{batch}: {tuple(k)}: {int_view_original_val:08X}; {int_view_faulty_val:08X}; {xor_result:08X}"
                print(sdc_line)

        os.remove(dataset_file)


if __name__=="__main__":
    argparser = get_argparser()
    args, unknown = argparser.parse_known_args()
    main(args)