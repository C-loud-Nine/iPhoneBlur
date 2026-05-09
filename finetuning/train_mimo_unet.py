"""Fine-tune MIMO-UNet - 50 epochs - FIXED"""
import sys,os,torch,torch.nn as nn,torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import cv2,numpy as np,random

sys.path.insert(0,str(Path(__file__).parent.parent))
from models.mimo_unet import MIMOUNet

def find_dataset():
    for p in [Path('./data/iphoneblur'),Path('./dataset/data/iphoneblur')]:
        if p.exists() and (p/'train').exists():return p
    return None

def get_image_pairs(data_dir):
    data_dir=Path(data_dir)
    blur_files,sharp_files=[],[]
    for ext in ['*.jpg','*.JPG','*.png']:
        blur_files.extend(list(data_dir.rglob(f'blur/{ext}')))
        sharp_files.extend(list(data_dir.rglob(f'sharp/{ext}')))
    return sorted(blur_files),sorted(sharp_files)

class SimpleDataset:
    def __init__(self,blur_paths,sharp_paths,patch_size=256):
        self.blur_paths,self.sharp_paths,self.patch_size=blur_paths,sharp_paths,patch_size
    def __len__(self):return len(self.blur_paths)
    def __getitem__(self,idx):
        blur=cv2.cvtColor(cv2.imread(str(self.blur_paths[idx])),cv2.COLOR_BGR2RGB)
        sharp=cv2.cvtColor(cv2.imread(str(self.sharp_paths[idx])),cv2.COLOR_BGR2RGB)
        h,w,ps=blur.shape[0],blur.shape[1],self.patch_size
        if h>ps and w>ps:
            top,left=random.randint(0,h-ps),random.randint(0,w-ps)
            blur,sharp=blur[top:top+ps,left:left+ps],sharp[top:top+ps,left:left+ps]
        return torch.from_numpy(blur).float().permute(2,0,1)/255.0,torch.from_numpy(sharp).float().permute(2,0,1)/255.0

def train(epochs=50,batch_size=4,lr=1e-4,data_dir=None,ckpt_dir='./checkpoints/mimo_unet',pretrained=None):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    data_dir=find_dataset() if data_dir is None else Path(data_dir)
    if data_dir is None:print("❌ Not found!");return
    train_dir=data_dir/'train'
    blur_files,sharp_files=get_image_pairs(train_dir)
    if len(blur_files)==0:print("❌ No images!");return
    print(f"✅ {len(blur_files)} pairs")
    from torch.utils.data import DataLoader
    train_loader=DataLoader(SimpleDataset(blur_files,sharp_files),batch_size=batch_size,shuffle=True,num_workers=0)
    model=MIMOUNet(num_res=8).to(device)
    print(f"MIMO: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    if pretrained:
        ckpt=torch.load(pretrained,map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict',ckpt.get('params',ckpt)) if isinstance(ckpt,dict) else ckpt)
    
    # FIXED: Create optimizer and scheduler correctly
    criterion=nn.L1Loss()
    optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    
    os.makedirs(ckpt_dir,exist_ok=True)
    print(f"\nTraining {epochs} epochs...\n")
    best_loss=float('inf')
    for epoch in range(1,epochs+1):
        model.train();epoch_loss=0
        pbar=tqdm(train_loader,desc=f"Epoch {epoch}/{epochs}")
        for blur,sharp in pbar:
            blur,sharp=blur.to(device),sharp.to(device)
            outs=model(blur)
            loss=sum(criterion(o,sharp) for o in outs) if isinstance(outs,list) else criterion(outs,sharp)
            optimizer.zero_grad();loss.backward();optimizer.step()
            epoch_loss+=loss.item()
            pbar.set_postfix({'Loss':f"{loss.item():.4f}"})
        scheduler.step()
        avg=epoch_loss/len(train_loader)
        print(f"Epoch {epoch} - Loss: {avg:.4f}")
        if epoch%10==0:torch.save({'model_state_dict':model.state_dict()},f"{ckpt_dir}/mimo_epoch_{epoch}.pth")
        if avg<best_loss:best_loss=avg;torch.save({'model_state_dict':model.state_dict()},f"{ckpt_dir}/mimo_best.pth")
    torch.save({'model_state_dict':model.state_dict()},f"{ckpt_dir}/mimo_final.pth")
    print("✅ Done!")

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--data_dir',default=None)
    p.add_argument('--checkpoint_dir',default='./checkpoints/mimo_unet')
    p.add_argument('--pretrained',default=None)
    args=p.parse_args()
    train(args.epochs,args.batch_size,1e-4,args.data_dir,args.checkpoint_dir,args.pretrained)
