import os
import nvbitfi_DNN as nvbitfi_DNN
import argparse


def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('-ln','--layer_number', required=True, type=int, help='golden')
    parser.add_argument('-bs','--batch_size', required=True, type=int, help='golden')
    parser.add_argument('-sz','--shape',required=False, nargs='+', type=int ,help="shape of the output layer")
    parser.add_argument('-onnx','--onnx', required=False, action='store_true', help='golden')
    parser.add_argument('-trt','--run_trt', required=False, action='store_true', help='golden')
    return parser

def main(args):
    Layer_number = args.layer_number
    batch_size = args.batch_size
    if args.shape:
        shape = tuple(args.shape)

    if args.run_trt:
        Layer = nvbitfi_DNN.TRT_load_embeddings(layer_number=Layer_number,batch_size=batch_size, layer_output_shape=shape)
        Layer.TRT_layer_inference()
    else:
        Layer = nvbitfi_DNN.load_embeddings(layer_number=Layer_number,batch_size=batch_size)
        Layer.layer_inference()
        if args.onnx:
            Layer.TRT_create_onnx_model()


if __name__=="__main__":
    argparser = get_argparser()
    main(argparser.parse_args())