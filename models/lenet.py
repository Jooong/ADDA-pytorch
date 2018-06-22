from torch import nn

class LeNet(nn.Module):
    """
    Different from the original structure of LeNet5
    The same strucure as the lenet implemented in Caffe-Lenet
    https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    """
    
    def __init__(self,ngpu):
        super(LeNet,self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=20,out_channels=50,kernel_size=5,stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU()
        )
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(4*4*50,500),
            nn.ReLU()
        )
        
        self.fc2 = nn.Linear(500,10)
        
    def forward(self,x):
        if x.is_cuda and self.ngpu>1:
            x = nn.parallel.data_parallel(self.conv, x, range(self.ngpu))
            x = x.view(-1,4*4*50)
            embedding = nn.parallel.data_parallel(self.fc1,x,range(self.ngpu))
            out = nn.parallel.data_parallel(self.fc2,embedding,range(self.ngpu))
        else:
            x = self.conv(x)
            x = x.view(-1,4*4*50)
            embedding = self.fc1(x)
            out = self.fc2(embedding)

        return embedding,out