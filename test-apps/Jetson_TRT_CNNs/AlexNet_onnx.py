import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.common_types as ct
import torchvision
from torchvision import models
import numpy as np
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import os, sys
import argparse
import nvbitfi_DNN as nvbitDNN
import torch.onnx


def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('--golden', required=False, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, default=0, help='golden')
    parser.add_argument('-bs','--batch_size', required=False, type=int, default=1, help='golden')
    return parser


def main(args):

    path = os.path.dirname(__file__)
    # os.environ["CUDA_VISIBLE_DEVICES"]=""

    transform = transforms.Compose([            #[1]
        transforms.Resize(256),                    #[2]
        transforms.CenterCrop(224),                #[3]
        transforms.ToTensor(),                     #[4]
        transforms.Normalize(                      #[5]
        mean=[0.485, 0.456, 0.406],                #[6]
        std=[0.229, 0.224, 0.225]                  #[7]
        )])


    # Define relevant variables for the ML task
    batch_size = args.batch_size #512 
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10

    LeNet_dict = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }

    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # device = 'cpu'
    # Loading the dataset and preprocessing
    train_dataset = torchvision.datasets.ImageFolder(
        root="~/dataset/ilsvrc2012/train",
        transform=transform
    )

    test_dataset = train_dataset = torchvision.datasets.ImageFolder(
        root="~/dataset/ilsvrc2012/val",
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    #print(args.golden)

    if args.golden:
        # torch.backends.cudnn.allow_tf32=True
        # torch.backends.cudnn.enabled=False

        # print(torch.backends.cuda.matmul.allow_tf32)
        # print(torch.backends.cudnn.allow_tf32)
        model = models.alexnet(pretrained=True)

        model = model.to(device)
        model.eval()
        
        #print(model)

        #Embeddings = nvbitDNN.extract_embeddings_nvbit(
        #    model=model, lyr_type=[nn.Conv2d], lyr_num=args.layer_number, batch_size=batch_size
        #)

        t = time.time()
        tot_imgs=0
        gacc1=0
        gacc5=0
        with torch.no_grad():
            print(f"label; class; pred")
            for batch, (images, labels) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                dummy_input = images
                # if(labels[0].item()==0):
                outputs = model(images)
                # sorted, indices=torch.sort(outputs.data)
                pred, clas=outputs.cpu().topk(5,1,True,True)
                clas = clas.t()
                pred = pred.t()
                size = pred.shape
                
                for idx,label in enumerate(labels):
                    for pred_top in range(size[0]):
                        print(f"{label}; {clas[pred_top][idx]}; {pred[pred_top][idx]}")
                Res = clas.eq(labels[None].cpu())
                
                acc1 = Res[:1].sum(dim=0,dtype=torch.float32)
                acc5 = Res[:5].sum(dim=0,dtype=torch.float32)
                tot_imgs+=batch_size
                gacc1 += Res[:1].flatten().sum(dtype=torch.float32)
                gacc5 += Res[:5].flatten().sum(dtype=torch.float32)
                if batch*batch_size+batch_size>=1:
                    break
            elapsed = time.time() - t
            print(
                "Accuracy of the network on the {} test images: acc1 {} % acc5 {} % in {} sec".format(
                    tot_imgs, 100 * gacc1 / tot_imgs,  100 * gacc5 / tot_imgs, elapsed
                )
            )

        #Embeddings.extract_embeddings_target_layer()
        onnx_model_name = "AlexNet_pytorch.onnx"
        TRT_model_name = "AlexNet_pytorch.rtr"
        torch.onnx.export(model, dummy_input, onnx_model_name, verbose=False)

        USE_FP16 = False
        target_dtype = np.float16 if USE_FP16 else np.float32
        if USE_FP16:
            cmd=f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_model_name} --saveEngine={TRT_model_name}  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
        else:
            cmd=f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_model_name} --saveEngine={TRT_model_name}  --explicitBatch"
        os.system(cmd)

    else:
        Target_layer = nvbitDNN.load_embeddings(1, batch_size)
        Target_layer.layer_inference()


if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

