"""---------------------------------------------
Modules Used:
*   argparse    :   for command-line argument parsing
*   os          :   for directory handling
*   random      :   for reproducibility and tensor operations
*   numpy       :   for reproducibility and tensor operations
*   torch       :   for reproducibility and tensor operations
*   torch.nn    :   for neural network layers and loss functions
*   torch.optim :   for optimizers
*   model.VGG6  :   user-defined CNN architecture
*   utils.get_cifar10_loaders :
*                   utility for data loading and preprocessing
*   tqdm        :   for progress visualization
*   wandb       :   for experiment tracking (optional)
---------------------------------------------"""    
import argparse, os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from model import VGG6
from utils import get_cifar10_loaders
from tqdm import tqdm

# Try to import Weights & Biases for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

"""---------------------------------------------
* def name :   
*       set_seed
*
* purpose:
*        Sets random seeds across libraries for reproducible experiments
*        but for this assignemnt keeping default at 42
*
* Input parameters:
*       seed   :   Seed value to ensure deterministic behavior.
* return:
*       
---------------------------------------------"""
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""---------------------------------------------
* def name :   
*       train_one_epoch
*
* purpose:
*        Trains the model for a single epoch.
*
* Input parameters:
*       model   :   Trained model
*       loader  :   DataLoader for validation or test data
*       criterion:  Loss function
*       optimizer: Optimization algorithm.
*       device :    'cuda' or 'cpu'
* return:
*       average_loss
*       accuracy   
---------------------------------------------"""
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); running_loss=correct=total=0
    for x,y in tqdm(loader, desc='train', leave=False):
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step()
        running_loss+=loss.item()*x.size(0); preds=out.argmax(dim=1); correct+=(preds==y).sum().item(); total+=x.size(0)
    return running_loss/total, correct/total

"""---------------------------------------------
* def name :   
*       evaluate
*
* purpose:
*        Evaluates the model on validation or test data.
*
* Input parameters:
*       model   :   Trained model
*       loader  :   DataLoader for validation or test data
*       criterion:  Loss function
*       device :    'cuda' or 'cpu'
* return:
*       average_loss
*       accuracy
---------------------------------------------"""
def evaluate(model, loader, criterion, device):
    model.eval(); running_loss=correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); out=model(x); loss=criterion(out,y)
            running_loss+=loss.item()*x.size(0); preds=out.argmax(dim=1); correct+=(preds==y).sum().item(); total+=x.size(0)
    return running_loss/total, correct/total

"""---------------------------------------------
* def name :   
*       get_optimizer
*
* purpose:
*        Returns the optimizer instance based on its name.
*
* Input parameters:
*       name :      Optimizer name ('sgd', 'nesterov', 'adam', 'rmsprop', 'adagrad')
*       params:     Model parameters to optimize
*       lr:         Learning rate
*       momentum:   Momentum factor
*
* return:
*       torch.optim.Optimizer: Configured optimizer object     
---------------------------------------------"""
def get_optimizer(name, params, lr, momentum=0.9):
    name=name.lower()
    if name=='sgd': return optim.SGD(params, lr=lr, momentum=momentum)
    if name=='nesterov': return optim.SGD(params, lr=lr, momentum=momentum, nesterov=True)
    if name=='adam': return optim.Adam(params, lr=lr)
    if name=='nadam': return optim.NAdam(params, lr=lr)
    if name=='adagrad': return optim.Adagrad(params, lr=lr)
    if name=='rmsprop': return optim.RMSprop(params, lr=lr, momentum=momentum)
    raise ValueError(f'Unknown optimizer {name}')

"""---------------------------------------------
    Main training function that orchestrates data loading, model setup,
    training, validation, logging, and checkpoint saving.
---------------------------------------------"""
def main():
    # ---------------- Argument Parsing ----------------
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch_size',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.01)
    p.add_argument('--activation',type=str,default='relu')
    p.add_argument('--optimizer',type=str,default='sgd')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--project',type=str,default='vgg6_cifar10')
    p.add_argument('--save_dir',type=str,default='./checkpoints')
    p.add_argument('--no_wandb',action='store_true')
    a=p.parse_args()
    
    # ---------------- Setup ----------------
    set_seed(a.seed); device='cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.save_dir, exist_ok=True)

    # ---------------- Data, Model, and Optimizer ----------------
    train_loader,val_loader,test_loader=get_cifar10_loaders(batch_size=a.batch_size)
    model=VGG6(num_classes=10,activation=a.activation).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=get_optimizer(a.optimizer, model.parameters(), lr=a.lr)

     # ---------------- Initialize wandb (optional) ----------------
    if WANDB_AVAILABLE and not a.no_wandb:
        wandb.init(project=a.project, config=vars(a)); wandb.watch(model)

     # ---------------- Training Loop ----------------
    best_val=0; history={'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(1,a.epochs+1):
        tl,ta=train_one_epoch(model,train_loader,criterion,optimizer,device)
        vl,va=evaluate(model,val_loader,criterion,device)
        print(f'Epoch {epoch}/{a.epochs} train_acc:{ta:.4f} val_acc:{va:.4f}')
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va)
        if WANDB_AVAILABLE and not a.no_wandb:
            wandb.log({'epoch':epoch,'train_loss':tl,'train_acc':ta,'val_loss':vl,'val_acc':va})
            
        # Save best model checkpoint
        if va>best_val:
            best_val=va; save=os.path.join(a.save_dir,f'best_{a.activation}_{a.optimizer}_bs{a.batch_size}_lr{a.lr}.pth')
            torch.save({'model_state_dict':model.state_dict(),'val_acc':va},save)

    # ---------------- wandb Summary ----------------
    if WANDB_AVAILABLE and not a.no_wandb:
        wandb.run.summary['best_val_acc']=best_val

if __name__=='__main__': main()
