import os 
import torch
import shutil
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
import timm, torchmetrics
from tqdm import tqdm

'''
Classifier source code inspired by Kaggle, but heavily modified for our purposes:

https://www.kaggle.com/code/killa92/f1-0-95-birds-species-vis-classification
'''
class BirdDataset(Dataset):

    def __init__(self, root, data):
        '''
        BirdDataset constructor.
        '''
        mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
        transformation = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean = mean, std = std)])

        self.transformations = transformation
        self.im_paths = sorted(glob(f"{root}/{data}/*/*"))
        self.cls_names, self.cls_counts, count, data_count = {}, {}, 0, 0

        # for all the image paths, populate the dictionary of class names
        for im_path in self.im_paths:
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names: 
                self.cls_names[class_name] = count 
                self.cls_counts[class_name] = 1 
                count += 1
            else: 
                self.cls_counts[class_name] += 1
        
    def get_class(self, path):
        '''
        Given a class path, return the class name.

        Example: ./bird-data//train/WHITE CRESTED HORNBILL/143.jpg
        Returns: WHITE CRESTED HORNBILL
        ''' 
        return os.path.dirname(path).split("/")[-1]
    
    def __len__(self): 
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = self.cls_names[self.get_class(im_path)]
        im = self.transformations(im)
        
        return im, gt
    
class ModelTrainer:
    '''
    Class that trains the model.
    '''
    def __init__(self, model, train_dl, val_dl, classes, save_prefix, save_dir):
        '''
        Class constructor. Takes in datasets and save data.
        '''
        self.model = model.to("cuda")
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.classes = classes
        self.save_prefix = save_prefix
        self.save_dir = save_dir
        self.device = "cuda"
        self.epochs = 10

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=3e-4)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=len(classes)).to(self.device)

        self.best_loss = float(torch.inf)
        self.best_acc = 0
        self.threshold = 0.01
        self.not_improved = 0
        self.patience = 5
    
    def get_predictions(self, ims):
        '''
        Returns the predictions of the current model. 
        '''
        return self.model(ims)
    
    def populate_metrics(self, preds, labels, epoch_metrics):
        '''
        Populates the accuracy and F1 metrics.
        '''
        acc = (torch.argmax(preds, dim=1) == labels).sum().item()
        f1 = self.f1_score(preds, labels)
        epoch_metrics['acc'] += acc
        epoch_metrics['f1'] += f1.item()

    def get_loss(self, preds, labels, epoch_metrics):
        '''
        Returns the loss, populates the epoch_metrics['loss'] variable.
        '''
        loss = self.loss_fn(preds, labels)
        epoch_metrics['loss'] += loss.item()
        return loss
    
    def train_epoch(self):
        '''
        Code to train one epoch.
        '''
        self.model.train()
        epoch_metrics = {'loss': 0, 'acc': 0, 'f1': 0}

        for batch in tqdm(self.train_dl):
            ims, labels = batch[0].to(self.device), batch[1].to(self.device)

            preds = self.get_predictions(ims)
            
            self.populate_metrics(preds, labels, epoch_metrics)

            loss = self.get_loss(preds, labels, epoch_metrics)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_metrics['loss'] / len(self.train_dl), epoch_metrics['acc'] / len(self.train_dl.dataset), epoch_metrics['f1'] / len(self.train_dl)
    
    def validate_epoch(self):
        '''
        Code to validate one epoch.
        '''
        self.model.eval()
        epoch_metrics = {'loss': 0, 'acc': 0, 'f1': 0}
        with torch.no_grad():
            for batch in self.val_dl:
                ims, labels = batch[0].to(self.device), batch[1].to(self.device)

                preds = self.get_predictions(ims)

                self.populate_metrics(preds, labels, epoch_metrics)

                _ = self.get_loss(preds, labels, epoch_metrics)

        return epoch_metrics['loss'] / len(self.val_dl), epoch_metrics['acc'] / len(self.val_dl.dataset), epoch_metrics['f1'] / len(self.val_dl)
    
    def save_model(self, epoch, val_loss):
        '''
        Saves the model to the save_dir path.
        '''
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.save_dir}/{self.save_prefix}_epoch{epoch}_val_loss{val_loss:.3f}.pth")
    
    def train(self):
        '''
        Training loop.
        '''
        for epoch in range(self.epochs):
            tr_loss, tr_acc, tr_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate_epoch()

            print(f"Epoch {epoch+1}:")
            print(f"Train Loss: {tr_loss:.3f}, Acc: {tr_acc:.3f}, F1: {tr_f1:.3f}")
            print(f"Val Loss: {val_loss:.3f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f}")

            if val_loss < self.best_loss - self.threshold:
                self.best_loss = val_loss
                self.save_model(epoch, val_loss)
                self.not_improved = 0
            else:
                self.not_improved += 1
                if self.not_improved >= self.patience:
                    print("Early stopping...")
                    break

def main():
    root = "./bird-data/"
    num_workers = 4
    batch_size = 32

    train_ds = BirdDataset(root = root, data = "train")
    val_ds = BirdDataset(root = root, data = "valid")
    test_ds = BirdDataset(root = root, data = "test")
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = False,  num_workers = num_workers)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    classes = train_ds.cls_names

    model = timm.create_model("rexnet_150", pretrained=True, num_classes=len(classes))
    trainer = ModelTrainer(model, train_dl, val_dl, classes, "birds", "saved_models")
    trainer.train()

if __name__ == "__main__":
    main()