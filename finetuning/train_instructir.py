"""Fine-tune InstructIR - 30 epochs"""
import sys,os,torch,torch.nn as nn,torch.optim as optim,subprocess,yaml,argparse as ap
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).parent.parent))
from dataset.dataloader import get_dataloader

def dict2namespace(config):
    namespace=ap.Namespace()
    for key,value in config.items():
        setattr(namespace,key,dict2namespace(value) if isinstance(value,dict) else value)
    return namespace

def setup_instructir(device):
    """Setup InstructIR - inline to avoid import collision"""
    print("⚙️ Setting up InstructIR...")
    if not os.path.exists('InstructIR'):
        print("   Cloning InstructIR...")
        subprocess.run(['git','clone','-q','https://github.com/mv-lab/InstructIR.git','InstructIR'])
    sys.path.insert(0,os.path.abspath('InstructIR'))
    import torchvision.transforms.functional as TF
    sys.modules['torchvision.transforms.functional_tensor']=TF
    from models.instructir import create_model as create_instructir
    from text.models import LanguageModel,LMHead
    with open('InstructIR/configs/eval5d.yml','r') as f:
        cfg=dict2namespace(yaml.safe_load(f))
    lm=LanguageModel(model="TaylorAI/bge-micro-v2")
    lm.eval()
    for param in lm.parameters():param.requires_grad=False
    lm_head=LMHead(embedding_dim=cfg.llm.model_dim,hidden_dim=cfg.llm.embd_dim,num_classes=cfg.llm.nclasses).to(device)
    im=create_instructir(input_channels=cfg.model.in_ch,width=cfg.model.width,enc_blks=cfg.model.enc_blks,middle_blk_num=cfg.model.middle_blk_num,dec_blks=cfg.model.dec_blks,txtdim=cfg.model.textdim).to(device)
    print(f"✅ InstructIR ready")
    return im,lm_head,lm

def train(epochs=30,batch_size=2,lr=1e-4,data_dir='./dataset/data/iphoneblur',ckpt_dir='./checkpoints/instructir',pretrained=None):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_loader=get_dataloader(f"{data_dir}/train",patch_size=256,is_train=True,batch_size=batch_size,num_workers=0)
    
    im,lm_head,lm=setup_instructir(device)
    print(f"InstructIR: {sum(p.numel() for p in im.parameters())/1e6:.1f}M params")
    
    # FIXED: Text embedding with correct unpacking and device placement
    prompt="deblur"
    with torch.no_grad():
        lm_embd=lm(prompt)
        txt,_=lm_head(lm_embd.to(device))
    
    if pretrained:
        ckpt=torch.load(pretrained,map_location=device)
        im.load_state_dict(ckpt.get('model_state_dict',ckpt.get('params',ckpt)) if isinstance(ckpt,dict) else ckpt)
        print(f"Loaded: {pretrained}")
    
    criterion=nn.L1Loss()
    optimizer=optim.AdamW(im.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    os.makedirs(ckpt_dir,exist_ok=True)
    
    print(f"\nTraining {epochs} epochs...\n")
    best_loss=float('inf')
    
    for epoch in range(1,epochs+1):
        im.train()
        epoch_loss=0
        pbar=tqdm(train_loader,desc=f"Epoch {epoch}/{epochs}")
        for blur,sharp in pbar:
            blur,sharp=blur.to(device),sharp.to(device)
            outs=im(blur,txt)
            if isinstance(outs,list):outs=outs[0]
            loss=criterion(outs,sharp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            pbar.set_postfix({'Loss':f"{loss.item():.4f}"})
        scheduler.step()
        avg=epoch_loss/len(train_loader)
        print(f"Epoch {epoch} - Loss: {avg:.4f}")
        if epoch%10==0:
            torch.save({'epoch':epoch,'model_state_dict':im.state_dict(),'loss':avg},f"{ckpt_dir}/instructir_epoch_{epoch}.pth")
        if avg<best_loss:
            best_loss=avg
            torch.save({'model_state_dict':im.state_dict()},f"{ckpt_dir}/instructir_best.pth")
    torch.save({'model_state_dict':im.state_dict()},f"{ckpt_dir}/instructir_final.pth")
    print(f"✅ Done!")

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch_size',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--data_dir',default='./dataset/data/iphoneblur')
    p.add_argument('--checkpoint_dir',default='./checkpoints/instructir')
    p.add_argument('--pretrained',default=None)
    args=p.parse_args()
    train(args.epochs,args.batch_size,args.lr,args.data_dir,args.checkpoint_dir,args.pretrained)
