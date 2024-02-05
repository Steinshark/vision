import torch 
import math


class AE_Model(torch.nn.Module):

    def __init__(self,compression_dimension:int=64,img_size:int=128):

        #Call super
        super(AE_Model,self).__init__()

        self.compression_dimension  = compression_dimension

        #Define standards
        self.compression_fn = torch.nn.ReLU
        self.encode_fn      = torch.nn.ReLU
        self.upsample_fn    = torch.nn.ReLU

        #Create modules for downsample
        self.lower_dim1     = torch.nn.Sequential(torch.nn.Conv2d(3,16,3,1,1),self.compression_fn(),torch.nn.MaxPool2d(2))     #64
        self.lower_dim2     = torch.nn.Sequential(torch.nn.Conv2d(16,32,3,1,1),self.compression_fn(),torch.nn.MaxPool2d(2))    #32
        self.lower_dim3     = torch.nn.Sequential(torch.nn.Conv2d(32,64,3,1,1),self.compression_fn(),torch.nn.MaxPool2d(2))   #16
        self.lower_dim4     = torch.nn.Sequential(torch.nn.Conv2d(64,16,3,1,1),self.compression_fn(),torch.nn.MaxPool2d(2))    #8
       
        self.adaptive1      = torch.nn.AdaptiveMaxPool2d((8,8))

        #Create the intermediate modules
        self.encode1        = torch.nn.Sequential(torch.nn.Linear(8*8*16,compression_dimension),torch.nn.Dropout(p=.1),self.encode_fn())
        self.encode2        = torch.nn.Sequential(torch.nn.Linear(compression_dimension,compression_dimension),torch.nn.Dropout(p=.1),self.encode_fn())
        self.unflatten      = torch.nn.Sequential(torch.nn.Linear(compression_dimension,4*compression_dimension),self.encode_fn(),torch.nn.Unflatten(dim=1,unflattened_size=(self.compression_dimension,2,2)),self.encode_fn())

        #Create modules for upsample 

        self.upsamp1        = torch.nn.Sequential(torch.nn.Upsample(size=(8,8)),torch.nn.Conv2d(self.compression_dimension,256,3,1,1),torch.nn.BatchNorm2d(256),self.upsample_fn())
        self.upsamp2        = torch.nn.Sequential(torch.nn.Upsample(size=(16,16)),torch.nn.Conv2d(256,128,3,1,1),torch.nn.BatchNorm2d(128),self.upsample_fn(),torch.nn.Conv2d(128,128,3,1,1),torch.nn.BatchNorm2d(128),self.upsample_fn())
        self.upsamp3        = torch.nn.Sequential(torch.nn.Upsample(size=(32,32)),torch.nn.Conv2d(128,64,3,1,1),torch.nn.BatchNorm2d(64),self.upsample_fn(),torch.nn.Conv2d(64,64,3,1,1),torch.nn.BatchNorm2d(64),self.upsample_fn())
        self.upsamp4        = torch.nn.Sequential(torch.nn.Upsample(size=(64,64)),torch.nn.Conv2d(64,32,3,1,1),torch.nn.BatchNorm2d(32),self.upsample_fn(),torch.nn.Conv2d(32,32,3,1,1),torch.nn.BatchNorm2d(32),self.upsample_fn())
        self.upsamp5        = torch.nn.Sequential(torch.nn.Upsample(size=(img_size,img_size)),torch.nn.Conv2d(32,3,3,1,1),torch.nn.Tanh()) 



    def forward(self,x:torch.Tensor):

        #Decrease dimension 
        x                   = self.lower_dim1(x)
        x                   = self.lower_dim2(x)
        x                   = self.lower_dim3(x)
        x                   = self.lower_dim4(x)
        x                   = self.adaptive1(x)

        #Flatten 
        x                   = x.flatten(1)


        #Send through encoder 
        x                   = self.encode1(x)
        x                   = self.encode2(x)

        #Unflatten 
        x                   = self.unflatten(x)

        #Send through upsampler 
        x                   = self.upsamp1(x)
        x                   = self.upsamp2(x)
        x                   = self.upsamp3(x)
        x                   = self.upsamp4(x)
        x                   = self.upsamp5(x)

        return x 

class UpSampler(torch.nn.Module):

    def __init__(self,final_size):

        super(UpSampler,self).__init__()

        conv1_fn    = torch.nn.LeakyReLU
        conv2_fn    = torch.nn.LeakyReLU

        #Assume input of lr
        self.Conv1  = torch.nn.Sequential(

            #   lrxlr -> lrxlr
            torch.nn.Conv2d(3,8,3,1,1,bias=False),
            torch.nn.BatchNorm2d(8),
            conv1_fn(),

            #   lrxlr -> lrxlr
            torch.nn.Conv2d(8,16,3,1,1,bias=False),
            torch.nn.BatchNorm2d(16),
            conv1_fn(),

            #   lrxlr -> lrxlr
            torch.nn.Conv2d(16,16,3,1,1,bias=False),
            torch.nn.BatchNorm2d(16),
            conv1_fn()
        )

        self.Upsample1  = torch.nn.Sequential(

            #   lrxlr -> (lr*2)x(lr*2)
            #torch.nn.ConvTranspose2d(16,16,4,2,1),
            torch.nn.Upsample(size=final_size)
        )

        self.Conv2      = torch.nn.Sequential(

            #   (lr*2)x(lr*2) -> (lr*2)x(lr*2)
            torch.nn.Conv2d(16,16,3,1,1,bias=False),
            torch.nn.BatchNorm2d(16),
            conv2_fn(),

            #   (lr*2)x(lr*2) -> (lr*2)x(lr*2)
            torch.nn.Conv2d(16,8,3,1,1,bias=False),
            torch.nn.BatchNorm2d(8),
            conv2_fn(),

            #   (lr*2)x(lr*2) -> (lr*2)x(lr*2)
            torch.nn.Conv2d(8,3,3,1,1,bias=True),
            torch.nn.Tanh()
        )


    #Assumes x is [-1,1] and 3xlrxlr
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x       = self.Conv1(x)
        x       = self.Upsample1(x)
        x       = self.Conv2(x)

        return x

if __name__ == "__main__":
    model       = AE_Model(512,img_size=128)
    x           = torch.randn(size=(8,3,128,128))

    y           = model.forward(x)

