"""Fine-tune NAFNet - 50 epochs"""
import sys,os,torch,torch.nn as nn,torch.optim as optim
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).parent.parent))
from models.nafnet import NAFNet
from dataset.dataloader import get_dataloader

def train(epochs=50,batch_size=4,lr=1e-4,data_dir='./dataset/data/iphoneblur',ckpt_dir='./checkpoints/nafnet',pretrained=None):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_loader=get_dataloader(f"{data_dir}/train",patch_size=256,is_train=True,batch_size=batch_size,num_workers=0)
    
    model=NAFNet(width=64,enc_blk_nums=[1,1,1,28]).to(device)
    print(f"NAFNet: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    if pretrained:
        ckpt=torch.load(pretrained,map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict',ckpt.get('params',ckpt)) if isinstance(ckpt,dict) else ckpt)
        print(f"Loaded: {pretrained}")
    
    criterion=nn.L1Loss()
    optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    os.makedirs(ckpt_dir,exist_ok=True)
    
    print(f"\nTraining {epochs} epochs...\n")
    best_loss=float('inf')
    
    for epoch in range(1,epochs+1):
        model.train()
        epoch_loss=0
        pbar=tqdm(train_loader,desc=f"Epoch {epoch}/{epochs}")
        for blur,sharp in pbar:
            blur,sharp=blur.to(device),sharp.to(device)
            out=model(blur)
            loss=criterion(out,sharp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            pbar.set_postfix({'Loss':f"{loss.item():.4f}"})
        scheduler.step()
        avg=epoch_loss/len(train_loader)
        print(f"Epoch {epoch} - Loss: {avg:.4f}")
        if epoch%10==0:
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'loss':avg},f"{ckpt_dir}/nafnet_epoch_{epoch}.pth")
        if avg<best_loss:
            best_loss=avg
            torch.save({'model_state_dict':model.state_dict()},f"{ckpt_dir}/nafnet_best.pth")
    torch.save({'model_state_dict':model.state_dict()},f"{ckpt_dir}/nafnet_final.pth")
    print(f"✅ Done!")

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--data_dir',default='./dataset/data/iphoneblur')
    p.add_argument('--checkpoint_dir',default='./checkpoints/nafnet')
    p.add_argument('--pretrained',default=None)
    args=p.parse_args()
    train(args.epochs,args.batch_size,args.lr,args.data_dir,args.checkpoint_dir,args.pretrained)
