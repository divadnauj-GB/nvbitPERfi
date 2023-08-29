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
import argparse
import nvbitfi_DNN as nvbitDNN



def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('--golden', required=False, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, default=0, help='golden')
    parser.add_argument('-bs','--batch_size', required=False, type=int, default=1, help='golden')
    return parser


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


def main(args):

    path = os.path.dirname(__file__)
    # os.environ["CUDA_VISIBLE_DEVICES"]=""

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
    train_dataset = torchvision.datasets.MNIST(
        root=os.path.join(path, "data"),
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        ),
        download=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        root=os.path.join(path, "data"),
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
            ]
        ),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    print(args.golden)

    if args.golden:
        # torch.backends.cudnn.allow_tf32=True
        # torch.backends.cudnn.enabled=False

        # print(torch.backends.cuda.matmul.allow_tf32)
        # print(torch.backends.cudnn.allow_tf32)
        
        model = torch.load(
            os.path.join(path, "./checkpoint/LeNet"), map_location=torch.device("cpu")
        )
        model = model.to(device)
        model.eval()
        
        print(model)

        Embeddings = nvbitDNN.extract_embeddings_nvbit(
            model=model, lyr_type=[nn.Conv2d], lyr_num=args.layer_number, batch_size=batch_size
        )

        t = time.time()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                # if(labels[0].item()==0):
                outputs = Embeddings.DNN_inference(images,labels)
                # sorted, indices=torch.sort(outputs.data)
                sorted, indices = outputs.data.sort(descending=True)

                for i in range(0, labels.size(0)):
                    print(f"\n\nImage {LeNet_dict[labels[i].item()]}.png")
                    for j in range(0, 5):
                        print(f"{LeNet_dict[indices[i][j].item()]}:{sorted[i][j]}")
                    # _, predicted = torch.max(outputs.data, 1)
                    correct += (indices[i][0] == labels[i].item()).sum().item()
                total += labels.size(0)
                # break
                if batch*batch_size+batch_size>=100:
                    break
            elapsed = time.time() - t
            print(
                "Accuracy of the network on the {} test images: {} % in {} sec".format(
                    total, 100 * correct / total, elapsed
                )
            )

        Embeddings.extract_embeddings_target_layer()

    else:
        Target_layer = nvbitDNN.load_embeddings(1, batch_size)
        Target_layer.layer_inference()


if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

