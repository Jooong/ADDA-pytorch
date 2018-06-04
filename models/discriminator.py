from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        layers = []
        for _ in range(2):
            layers.append(nn.Linear(in_features=500,out_features=500))
            layers.append(nn.LeakyReLU(0.1))
            
        layers.append(nn.Linear(in_features=500,out_features=2))
        self.model = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.model(x)
    
class Discriminator_CNN(nn.Module):
    def __init__(self):
        super(Discriminator_CNN,self).__init__()
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=2),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=2),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv2d(in_channels=64,out_channels=10,kernel_size=4,stride=2,padding=2)
        )
        
    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,1)
        return x