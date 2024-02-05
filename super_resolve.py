import torch
import models 
import torchvision.io
from torch.utils.data import DataLoader,Dataset
import os 
import random
from matplotlib import pyplot as plt
import time 


class VideoDataSet(Dataset):

    def __init__(self,path:str,len=2048,ep_window=20):
        #Fnames will be [name]_[resolution].mp4

        self.data           = {144:[],240:[]} 
        self.fnames         = {144:[],240:[]}
        self.temp_frames    = {}
        self.readers        = {}
        self.len            = len
        self.ep_window      = ep_window

        for fname in [os.path.join(path,f) for f in os.listdir(path)]:
            fname   = fname.replace("\\","/")
            print(f"\tloading {fname}")
            if "144" in fname:
                self.fnames[144].append(fname)
                self.readers[fname] = torchvision.io.VideoReader(fname)
            elif "240" in fname:
                self.fnames[240].append(fname)
                self.readers[fname] = torchvision.io.VideoReader(fname)        
        self.build_temp_frames()
        print()
        print()

    def __getitem__(self,i) -> tuple[torch.Tensor,torch.Tensor]:
        
        #Load random second from a random fname
        fname_144   = random.choice(self.fnames[144])
        fname_240   = fname_144.replace('144','240')

        #load frames 
        frame_num   = random.randint(0,self.temp_frames[fname_144].shape[0]-1)
        # frame_144   = self.transform_frame(self.temp_frames[fname_144][frame_num])
        frame_144   = self.temp_frames[fname_144][frame_num]
        # frame_240   = self.transform_frame(self.temp_frames[fname_240][frame_num])
        frame_240   = self.temp_frames[fname_240][frame_num]
        
        return frame_144,frame_240

    def __len__(self):
        return self.len
    

    def transform_frame(self,img:torch.Tensor)->torch.Tensor:
        img     = 2 * (img / 255)
        img     = img - 1
        return img
    
    def untransform_frame(self,img:torch.Tensor,mode='float')->torch.Tensor:
        img     = img + 1 
        img     = img / 2

        if not mode == 'float':
            img     = img * 255
        return img

    def build_temp_frames(self):

        #Chose a random 20 second interval for this epoch
        for fname in self.fnames[144]:
                
            #Get 144
            reader          = self.readers[fname]
            vid_duration    = reader.get_metadata()['video']['duration'][0] - self.ep_window
            start_s         = random.randint(0,int(vid_duration-1))
            self.temp_frames[fname] = self.transform_frame(torchvision.io.read_video(fname,start_pts=start_s,end_pts=start_s+self.ep_window,pts_unit='sec',output_format='TCHW')[0])

            #Get 240 on same slice
            fname           = fname.replace('144','240')
            reader          = self.readers[fname]
            vid_duration    = reader.get_metadata()['video']['duration'][0] - self.ep_window
            self.temp_frames[fname] = self.transform_frame(torchvision.io.read_video(fname,start_pts=start_s,end_pts=start_s+self.ep_window,pts_unit='sec',output_format='TCHW')[0])

        self.temp_mean      = self.temp_frames[fname].float().mean()


def save_result(y:torch.Tensor,y_up:torch.Tensor,i):
    finalimg    = torch.zeros(size=(3,240*2,426))
    finalimg[:,:240,:]      = y.detach().cpu()
    finalimg[:,240:,:]      = y_up.detach().cpu()
    #finalimg                = finalimg.permute(*torch.arange(finalimg.ndim - 1, -1, -1))
    plt.subplots(figsize=(32,18))
    plt.imshow(finalimg.permute(1,2,0))

    if not os.path.isdir("C:/code/vision/lolruns"):
        os.mkdir("C:/code/vision/lolruns")
    plt.savefig(f"C:/code/vision/lolruns/{i}.jpg",bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    
    #Train vars
    bs              = 32
    ep              = 100
    lr              = .0002
    wd              = 0  
    betas           = (.5,.999)

    dataset         = VideoDataSet("C:/data/lol",len=512)
    dataloader      = DataLoader(dataset,batch_size=bs)

    model           = models.UpSampler(final_size=dataset.__getitem__(0)[1].shape[1:])
    optimizer       = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd,betas=betas)
    loss_fn         = torch.nn.MSELoss()

    for ep_num in range(ep):

        #Load data again
        t0          = time.time()
        dataset.build_temp_frames()

        print(f"\tEPOCH {ep_num} [mean={dataloader.dataset.temp_mean:.4f}]")
        for i,batch in enumerate(dataloader):

            #Zero grad 
            for param in model.parameters():
                param.grad  = None 
            
            #Snatch batch
            x,y           = batch

            #Get upsampled
            y_up        = model.forward(x)

            #Get error
            error       = loss_fn(y,y_up)
            error.backward()

            #Train
            optimizer.step()

            if i == 0 or int(len(dataloader)*.1) > 0 and i % int(len(dataloader)*.1) == 0:
                print(f"\t\tbatch [{i}/{len(dataloader)}]\tupscale loss is {error.mean().item():.4f}")

        #Save results
        print(f"\t\tbatch[{i}{len(dataloader)}]\tupscale loss is {error.mean().item():.4f}\n\t\ttime={(time.time()-t0):.2f}s\n")
        save_result(dataset.untransform_frame(y[0]),dataset.untransform_frame(y_up[0]),ep_num)
