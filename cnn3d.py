import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
from sklearn.metrics import r2_score
import os

# -------------------------------
# Dataset Class for 3D Input Data
# -------------------------------
class DictDataset(Dataset):
    def __init__(self, path):
        self.h5f = h5py.File(path, 'r')
        self.inputs = self.h5f["input"]["fill"]  # Access input dataset
        self.target = self.h5f["output"]["p"]    # Access target dataset
        self.size = self.inputs.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Correct shape to [1, D, H, W]
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32).squeeze(0).unsqueeze(0)
        target_data = torch.tensor(self.target[idx], dtype=torch.float32).squeeze(0)
        return input_data, target_data

    def close(self):
        self.h5f.close()



# -------------------------------
# CNN Model for 3D Data
# -------------------------------
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, D, H, W)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),                 # Output: (16, D/2, H/2, W/2)

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1), # Output: (32, D/2, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),                 # Output: (32, D/4, H/4, W/4)

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (64, D/4, H/4, W/4)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),                 # Output: (64, D/8, H/8, W/8)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 8 * 8, 256),  # Adjust input size based on final output dimensions of the encoder
            nn.ReLU(),
            nn.Linear(256, 128 * 64 * 64),   # Output flattened 3D pressure difference
            nn.Unflatten(1, (128, 64, 64))   # Reshape to 3D output
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# -------------------------------
# Training and Testing Functions
# -------------------------------
def train_model(model, train_loader, valid_loader, optimizer, loss_fn, device, num_epochs=20):
    best_r2 = -100
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        valid_loss, labels, preds = 0, [], []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                valid_loss += loss.item()
                labels.append(targets.cpu().numpy())
                preds.append(outputs.cpu().numpy())
        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        r2 = r2_score(labels.flatten(), preds.flatten())
        print(f"Validation Loss: {valid_loss / len(valid_loader):.4f}, R2 Score: {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            #print("Best model saved.")


def test_model(model, test_loader, device):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            labels.append(targets.cpu().numpy())
            preds.append(outputs.cpu().numpy())
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    r2 = r2_score(labels.flatten(), preds.flatten())
    print(f"Test R2 Score: {r2:.4f}")
    return labels, preds


# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    path_to_data = "/home/hpc/iwia/iwia105h/2D-porous-media-images-1/shiftedSet3D/shiftedTest.h5"
    dataset = DictDataset(path_to_data)
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Train the model
    train_model(model, train_loader, valid_loader, optimizer, loss_fn, device, num_epochs=20)

    # Test the model
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    labels, preds = test_model(model, test_loader, device)

    # Example Permeability Calculation
    flow_rate = 1e-4  #  flow rate in m^3/s
    viscosity = 1e-3  # Fluid viscosity in Pa·s
    length = 0.1      # Length in meters
    area = 0.01       # Cross-sectional area in m^2
    pressure_diff = np.mean(preds)  # Average predicted pressure difference
    permeability = (flow_rate * viscosity * length) / (pressure_diff * area)
    print(f"Predicted Permeability: {permeability:.4e} m^2")

    dataset.close()
