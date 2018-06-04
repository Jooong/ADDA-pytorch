import torch
from torchvision import transforms
from torchvision import datasets as dsets


def _rgb2gray(x):
    w = [0.2989, 0.5870, 0.1140]
    r = x[0,:,:]
    g = x[1,:,:]
    b = x[2,:,:]
    x = w[0]*r+w[1]*g+w[2]*b
    return torch.unsqueeze(x,dim=0)


def _transform(dset="MNIST"):
    params = {"mean":[0.5,], "std":[0.5]}
    
    if dset == "SVHN":
        return transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),_rgb2gray,transforms.Normalize(**params)])
    
    params = {"mean":[0.5,], "std":[0.5,]}
    return transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize(**params)])


def get_loader(type="MNIST",train=True,batch_size=100):
    if type=="MNIST":
        dset = dsets.MNIST(root="../data/MNIST",train=train,transform=_transform("MNIST"),download=True)
    else:
        split = "train" if train else "test"
        dset = dsets.SVHN(root="../data/SVHN",split=split,transform=_transform("SVHN"),download=True)
        
    loader= torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
    return loader