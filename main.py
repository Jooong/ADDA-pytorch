import torch
from models import LeNet, Discriminator
from preprocessing import get_loader
from trainer import train_source,evaluate,adapt_target_domain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get loaders

source_train_loader = get_loader(type="SVHN",train=True,batch_size=100)
source_test_loader = get_loader(type="SVHN",train=False,batch_size=100)
target_loader = get_loader(type="MNIST",train=False,batch_size=100)

# build source classifier
model_src = LeNet().to(device)

# train source classifier

print("-"*50)
print("\ntrain source classifier\n")

train_source(model_src,source_train_loader,source_test_loader,initialize=True,device=device)
      
print("-"*50)
print("\ntest accurracy in source domain: %f\n" %(evaluate(model_src,source_test_loader)))


# initialize target classifer with source classifer
model_trg = torch.load(open("./pretrained/lenet-source.pth","rb"))

# build discriminator
D = Discriminator()

# adaptation process
print("-"*50)
print("\nstart adpatation process\n")
adapt_target_domain(
                    D,model_src,model_trg,
                    source_train_loader,target_loader,
                    lr_d=0.002,lr_t=0.002,
                    initialize=True,num_epoch=100,
                    device=device
                   )

print("-"*50)
print("\naccurracy in target domain: %f\n" %(evaluate(model_trg,target_loader)))