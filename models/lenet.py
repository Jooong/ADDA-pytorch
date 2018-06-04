from torch import nn

class LeNet(nn.Module):
    """
    Different from the original structure of LeNet5
    The same strucure as the lenet implemented in Caffe-Lenet
    https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    """
    
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,4*4*50)
        embedding = self.fc1(x)
        out = self.fc2(embedding)
        return embedding,out
    
    
class CNN(nn.Module):
    """
    Different from the original structure of LeNet5
    The same strucure as the lenet implemented in Caffe-Lenet
    https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    """
    
    def __init__(self):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=2,stride=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=2,stride=1),
            nn.LeakyReLU(0.1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(8*8*64,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
        
    def forward(self,x):
        features = self.conv(x)
        pooled = self.pool(features)
        pooled = pooled.view(-1,8*8*64)
        logits = self.classifier(pooled)
        return features,logits
    
            