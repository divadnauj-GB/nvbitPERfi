import torch
import torchvision
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse 
import numpy as np
import os, time


def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('--golden', required=False, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, default=0, help='golden')
    parser.add_argument('-bs','--batch_size', required=False, type=int, default=1, help='golden')
    return parser


def main(args):

    BATCH_SIZE = args.batch_size
    target_dtype = np.float32
    path = os.path.dirname(__file__)

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

    TRT_model_name = "vit_b_16_noTF32.rtr"
    Num_outouts = 10000
    
    with open(os.path.join(path, TRT_model_name), "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        output = np.empty([BATCH_SIZE, Num_outouts], dtype = target_dtype) 
        # allocate device memory
        for batch, (images, labels) in enumerate(test_loader):
            sample_images = np.array(images, dtype=np.float32)
            break

        d_input = cuda.mem_alloc(1 * sample_images.nbytes)
        d_output = cuda.mem_alloc(1 * output.nbytes)
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()

        t = time.time()
        tot_imgs=0
        gacc1=0
        gacc5=0
        dummy_input=None
        for batch, (images, labels) in enumerate(test_loader):
            images = np.array(images, dtype=np.float32)

            cuda.memcpy_htod_async(d_input, images, stream)
            # execute model
            context.execute_async_v2(bindings, stream.handle, None)
            # transfer predictions back
            cuda.memcpy_dtoh_async(output, d_output, stream)
            # syncronize threads
            stream.synchronize()
            outputs = torch.from_numpy(output)
            pred, clas=outputs.cpu().topk(5,1,True,True)
            clas = clas.t()
            pred = pred.t()
            size = pred.shape
            
            for idx,label in enumerate(labels):
                for pred_top in range(size[0]):
                    print(f"{batch*BATCH_SIZE+idx}; {pred_top}; {label}; {clas[pred_top][idx]}; {pred[pred_top][idx]}")
            Res = clas.eq(labels[None].cpu())

            acc1 = Res[:1].sum(dim=0,dtype=torch.float32)
            acc5 = Res[:5].sum(dim=0,dtype=torch.float32)            
            gacc1 += Res[:1].flatten().sum(dtype=torch.float32)
            gacc5 += Res[:5].flatten().sum(dtype=torch.float32)
            tot_imgs+=BATCH_SIZE
            if batch*BATCH_SIZE+BATCH_SIZE>=1:
                break
        
        elapsed = time.time() - t
        print(
            "Accuracy of the network on the {} test images: acc1 {} % acc5 {} % in {} sec".format(
                tot_imgs, 100 * gacc1 / tot_imgs,  100 * gacc5 / tot_imgs, elapsed
            )
        )


if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

