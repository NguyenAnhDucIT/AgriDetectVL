import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.datasets import TLUStatesDataset
from data.transforms import TrainTransform, ValTransform
from models.agri_detect_vl import AgriDetectVL
from losses.prototype_ce import PrototypeCrossEntropy
from losses.alignment import ImageTextAlignment
from losses.regularizer import PrototypeRegularizer
from utils import set_seed, accuracy, AverageMeter, save_checkpoint

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, train_loader, criterion_ce, criterion_align, criterion_reg, optimizer, device, config):
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        logits, features, vision_features = model(images)
        
        loss_ce = criterion_ce(logits, labels)
        
        prototypes = model.get_prototypes().to(device)
        loss_align = criterion_align(features, prototypes[labels], labels)
        loss_reg = criterion_reg(prototypes)
        
        loss = (config['training']['lambda_ce'] * loss_ce + 
                config['training']['lambda_align'] * loss_align + 
                config['training']['lambda_reg'] * loss_reg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = accuracy(logits, labels)[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.2f}%'})
    
    return loss_meter.avg, acc_meter.avg

def validate(model, val_loader, criterion_ce, device):
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            logits, _, _ = model(images)
            loss = criterion_ce(logits, labels)
            acc = accuracy(logits, labels)[0]
            
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))
            
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.2f}%'})
    
    return loss_meter.avg, acc_meter.avg

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'g-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('TLU-Fruit — Training vs. Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'g-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('TLU-Fruit — Training vs. Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'training_results.png'")
    plt.close()

def main():
    set_seed(42)
    
    config_default = load_config('configs/default.yaml')
    config_dataset = load_config('configs/dataset.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform = TrainTransform(
        image_size=config_dataset['dataset']['image_size'],
        mean=config_dataset['dataset']['mean'],
        std=config_dataset['dataset']['std']
    )
    val_transform = ValTransform(
        image_size=config_dataset['dataset']['image_size'],
        mean=config_dataset['dataset']['mean'],
        std=config_dataset['dataset']['std']
    )
    
    train_dataset = TLUStatesDataset(
        root=config_dataset['dataset']['root'],
        split='train',
        transform=train_transform
    )
    val_dataset = TLUStatesDataset(
        root=config_dataset['dataset']['root'],
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_default['training']['batch_size'],
        shuffle=True,
        num_workers=config_default['data']['num_workers'],
        pin_memory=config_default['data']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_default['training']['batch_size'],
        shuffle=False,
        num_workers=config_default['data']['num_workers'],
        pin_memory=config_default['data']['pin_memory']
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.class_names)}")
    
    model = AgriDetectVL(
        config=config_default,
        class_names=train_dataset.class_names,
        prompt_templates=config_dataset['prompts']
    ).to(device)
    
    criterion_ce = PrototypeCrossEntropy()
    criterion_align = ImageTextAlignment(temperature=config_default['model']['temperature'])
    criterion_reg = PrototypeRegularizer()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_default['training']['learning_rate'],
        weight_decay=config_default['training']['weight_decay'],
        betas=config_default['optimizer']['betas'],
        eps=config_default['optimizer']['eps']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config_default['training']['num_epochs'],
        eta_min=config_default['scheduler']['min_lr']
    )
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(1, config_default['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config_default['training']['num_epochs']}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_ce, criterion_align, criterion_reg,
            optimizer, device, config_default
        )
        val_loss, val_acc = validate(model, val_loader, criterion_ce, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'checkpoints/best_model.pth')
            print(f"Best model saved with val_acc: {best_val_acc:.2f}%")
    
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    print(f"\nTraining completed! Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
