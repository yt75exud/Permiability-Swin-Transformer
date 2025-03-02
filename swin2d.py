import time
import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision
import collections
from sklearn import preprocessing
import timm
from timm.optim import create_optimizer_v2
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")



# # Data Processing

df_data = pd.read_csv('/home/hpc/iwia/iwia105h/2D-porous-media-images-1/rock_perm.csv')
df_data.describe()



min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
x_minmax = min_max_scaler.fit_transform(df_data[['PERM']])
df_data['label'] = x_minmax
df_data.describe()



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir:str, labels:dict, transform=None):
        self.dataset = []
        for img_name, label in labels.items():
            img_path = os.path.join(imgs_dir, img_name)
            self.dataset.append((img_path, label))
        self.transform = transform
        # print(self.dataset)

    def __getitem__(self, IDX):
        img_path, label = self.dataset[IDX]
        img_bgr = cv2.imread(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)




dict_labels = {}
for i, x in df_data.iterrows():
    filename = str(int(x.loc['IDX']))+'.png'
    label = x['label']
    dict_labels[filename] = float(label)
len(dict_labels)



transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Resize([224, 224]),
])




dataset = MyDataset('/home/hpc/iwia/iwia105h/2D-porous-media-images-1/rock_images', dict_labels, transform=transform)
total_count = dataset.__len__()
test_count = int(0.2 * total_count)
valid_count = int(0.2 * total_count)
train_count = total_count - test_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, valid_count, test_count), generator=torch.Generator().manual_seed(42)
)
print('Total: {}, Train: {}, Vali: {}, Test: {}'.format(total_count,train_count,valid_count, test_count))




#batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=1,
    prefetch_factor=2
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=32,
    num_workers=1,
    prefetch_factor=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=1,
    prefetch_factor=2
)


# # Train & Test Function



def train_model(model, output_path, train_loader, valid_loader, num_epochs, optimizer, ce, patience=100):
    train_losses = []
    valid_losses = []
    valid_r2 = []
    epoch_times = [] # List to store the epoch times
    best_r2 = -100
    early_stopping = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for epoch in range(num_epochs):
        start_time = time.time()  # Record the start time of the epoch
        losses = []
    
        # Training phase
        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(torch.device(device))
            targets = targets.type(torch.float).to(torch.device(device))
            outputs = model(inputs).squeeze()
            # print(outputs.shape, targets.shape)
            loss = ce(outputs, targets)
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_losses.append((sum(losses)/len(losses)).cpu().tolist())
        
        # Validation phase
        labels = []
        preds  = []
        losses = []
        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader):
                inputs = inputs.to(torch.device(device))
                targets = targets.type(torch.float).to(torch.device(device))
                outputs = model(inputs).squeeze()
                loss = ce(outputs, targets)
                losses.append(loss)
                labels.append(targets)
                preds.append(outputs)
        valid_losses.append((sum(losses)/len(losses)).cpu().tolist())
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        r2 = r2_score(labels.cpu().numpy(), preds.cpu().numpy())
        valid_r2.append(r2)
    
        end_time = time.time() # Record the end time of the epoch
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)  # Store the epoch time
    
        if r2 > best_r2:
            best_r2 = r2
            early_stopping = 0
            torch.save(model.state_dict(), os.path.join(output_path, 'best.pth'))
        else:
            early_stopping += 1
        
        print('Epoch {}/{} | Train Loss: {} | Valid Loss: {} | R2 Score: {} | Epoch Time: {:.2f} seconds'.format(epoch+1, num_epochs, train_losses[-1], valid_losses[-1], r2, epoch_time))
        
        if early_stopping > patience:
            break

    df_train_log = pd.DataFrame.from_dict({'train_loss': train_losses, 'val_loss': valid_losses, 'val_r2': valid_r2, 'epoch_times':epoch_times})
    df_train_log.to_csv(os.path.join(output_path, 'train_log.csv'), IDX=False)




def test_model(model, output_path, test_loader):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_labels = []
    test_preds = []
    with torch.no_grad():
        start_time = time.time()  # Record the start time
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.to(torch.device(device))
            targets = targets.type(torch.float).to(torch.device(device))
            outputs = model(inputs)            
            test_labels.append(targets)
            test_preds.append(outputs)
        end_time = time.time() # Record the end time
        test_time = end_time - start_time
    test_labels = torch.cat(test_labels, dim=0).squeeze()
    test_preds = torch.cat(test_preds, dim=0).squeeze()
    r2 = r2_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())
    print('test r2: {},test time: {}'.format(r2, test_time))
    df_test_result = pd.DataFrame.from_dict({'labels': test_labels.cpu().tolist(), 'predicts': test_preds.cpu().tolist()})
    df_test_result.to_csv(os.path.join(output_path, 'test_result.csv'), IDX=False)
    

#model swin transformer

model_name = 'swin_base_patch4_window7_224'
model_swin = timm.create_model(model_name, pretrained=True, num_classes=1, global_pool='avg')
o = model_swin(torch.randn(2, 3, 224, 224))
print(f'OUT shape: {o.shape}')


new_fc = torch.nn.Sequential(collections.OrderedDict([
    ('relu', torch.nn.ReLU()),# GELU ReLU
    ('dropout', torch.nn.Dropout(0.1)),
    ('fc', torch.nn.Linear(1024, 1))
    ]))

model_swin.head.fc = new_fc
print(model_swin.head)


swin_path = './'+ model_name+'_'+ str(400)

path = os.path.join(swin_path, 'best.pth')
if os.path.exists(path):
    model_swin.load_state_dict(torch.load(path))
    print("model load successful")
    

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device('cuda:{}'.format(0))
model_swin = model_swin.to(device)

num_epochs = 20
optimizer = torch.optim.Adam(model_swin.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
ce = torch.nn.HuberLoss()


train_model(model_swin, swin_path, train_loader, valid_loader, num_epochs, optimizer, ce)

test_model(model_swin, swin_path, test_loader)

test_model(model_swin, swin_path, train_loader)   

