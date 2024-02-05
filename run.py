import torch 
from torch.utils.data import Dataset,DataLoader
from models import AE_Model
import os 
import random 
from matplotlib import pyplot as plt 
import dataclean
import time

def import_ds(root:str,subset=None)->list[torch.Tensor]:
    images          = [] 
    img_max         = 255.0
    
    for i,file in enumerate(os.listdir(root)):
        filename    = os.path.join(root,file)

        loaded_tsr  = torch.load(filename)

        #Rescale to -1,1 
        loaded_tsr  = loaded_tsr / img_max
        loaded_tsr  = loaded_tsr * 2
        loaded_tsr  = loaded_tsr - 1

        #Add to images 
        images.append(loaded_tsr)
        

        if not subset is None and i+1 == subset:
            break

    print(f"created {len(images)}")

    return images

class IMGDataset(Dataset):

    def __init__(self,imgs):
        self.data   = imgs 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        return self.data[i]
    
if __name__ == "__main__":

    bs          = 32
    lr          = .001
    wd          = 0


    #prep_data
    dataset     = IMGDataset(import_ds("C:/code/data/imgds_128",subset=4096))
    dataloader  = DataLoader(dataset,batch_size=bs,shuffle=True)

    #Build model 
    model       = AE_Model(compression_dimension=256)
    loss_fn     = torch.nn.MSELoss()
    optimizer   = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)


    for ep in range(1000):
        model.train()
        print(f"\tEPOCH {ep}")
        t0      = time.time()
        for batch_num,items in enumerate(dataloader):

            for param in model.parameters():
                param.grad = None

            #Convert to float
            x       = items.float()
            
            #Apply noise 
            x_noisy = torch.stack([dataclean.apply_noise(x_,noise_peak=.1,iter_noise=10) for x_ in x])

            #Forward pass through model 
            y       = model.forward(x)
            y_noisy = model.forward(x_noisy)

            #Loss x 
            loss    = loss_fn(y,x)
            loss.backward()

            #Loss x noisy
            loss    = loss_fn(y_noisy,x)
            loss.backward()

            #Optimize
            optimizer.step()

            if batch_num == 0 or  batch_num % int(len(dataloader)*.1) == 0:
                print(f"\t\tloss batch [{batch_num}/{len(dataloader)}]\twas {loss.item():.4f}")
        
        #Take example 
        model.eval()
        with torch.no_grad():
            chosen_img  = dataset.__getitem__(random.randint(0,len(dataset))).unsqueeze(dim=0)
            processed   = model.forward(chosen_img)[0] + 1 
            processed   = processed / 2

            #Smush images 
            finalimg    = torch.zeros(size=(3,128+128,128))
            finalimg[:,:128,:]      = (chosen_img + 1) / 2 
            finalimg[:,128:,:]      = processed        
            finalimg                = finalimg.permute(*torch.arange(finalimg.ndim - 1, -1, -1))
            print(f"\t\ttime={(time.time()-t0):.2f}s")
            #plt.imshow(numpy.transpose(finalimg.numpy(),(1,2,0)))
            plt.imshow(finalimg.numpy())
            plt.savefig(f"C:/code/vision/modelruns/{ep}_save.jpg")  

