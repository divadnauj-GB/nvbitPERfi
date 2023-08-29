import nvbitfi_DNN
import argparse


def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('-ln','--layer_number', required=True, type=int, help='golden')
    parser.add_argument('-bs','--batch_size', required=True, type=int, help='golden')
    return parser

def main(args):
    Layer_number = args.layer_number
    batch_size = args.batch_size

    Layer = nvbitfi_DNN.load_embeddings(layer_number=Layer_number,batch_size=batch_size)
    Layer.layer_inference()



if __name__=="__main__":
    argparser = get_argparser()
    main(argparser.parse_args())