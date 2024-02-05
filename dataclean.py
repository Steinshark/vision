import torch 
import torchvision
import numpy
from PIL import Image
import random 
import os 
from matplotlib import pyplot as plt 



def random_crop(img:torch.Tensor,crop_dim:int=256,n_crops:int=8)->list[torch.Tensor]:

    crops           = []

    #Get length and width 
    img_h,img_w     = img.shape[1],img.shape[2]

    for _ in range(n_crops):
        
        #Pick crop at least 25% of dimensions
        min_crop    = int(min(img_w,img_h) * .25)

        try:
            crop_len    = random.randint(min_crop,min(img_w,img_h))
        except ValueError as ve:
            print(f"err given on vals\n\tmin_crop:{min_crop}\n\tw,h:{img_w},{img_h}")

        #Pick random upper left 
        ll_w        = random.randint(0,img_w-crop_len)
        ll_h        = random.randint(0,img_h-crop_len)

        crops.append(torch.nn.functional.interpolate(img.clone()[:,ll_h:ll_h+crop_len,ll_w:ll_w+crop_len].unsqueeze_(dim=0),size=(crop_dim,crop_dim))[0])
    
    return crops

def view_img(img:torch.Tensor) -> None:

    #convert to numpy then imshow 
    as_numpy    = numpy.transpose(img.numpy(),(1,2,0))

    #imshow
    plt.imshow(as_numpy,origin='lower')
    plt.show()

def create_ds(root:str,ds_root:str):

    #Transformer
    pil_xfmr            = torchvision.transforms.PILToTensor()
    saved               = 0 

    for filename in os.listdir(root):

        filename    = os.path.join(root,filename)

        #Load and random crop
        pil_img             = Image.open(filename)
        img_tsr             = pil_xfmr(pil_img)
        cropped_imgs        = random_crop(img_tsr,crop_dim=128,n_crops=24)

        #Save imgs to ds
        for img in cropped_imgs:
            savename    = os.path.join(ds_root,f"{saved}.pt")
            img.type(torch.float16)
            torch.save(img,savename)
            saved += 1

def apply_noise(img:torch.Tensor,noise_peak=.1,iter_noise=5):
    #Get noise from shape of img 
    img_min     = img.min()
    img_max     = img.max()
    for i in range(iter_noise):
        noise   = torch.randn(size=img.shape,dtype=img.dtype,device=img.device)
        noise   = img.max() * noise * noise_peak / noise.max()
        img     = img + noise 
        img     = img.clip(img_min,img_max)
    
    return img


if __name__ == "__main__":

    data_root           = "C:/code/data/photos/"
    save_root           = "C:/code/data/imgds_128"
    #ds                  = create_ds(data_root,save_root)
    img                 = torch.load(save_root+"/12.pt").permute(1,2,0).float()
    img                 = apply_noise(img,noise_peak=.05,iter_noise=100) / 255
    plt.imshow(img)
    plt.show() 
