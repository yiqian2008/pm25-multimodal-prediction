import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset import HoornImageDataset

# reproducibility
torch.manual_seed(42)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
dataset = HoornImageDataset("data/processed/dataset.csv")

# split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class TabularBaseline(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


model = TabularBaseline(input_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop
for epoch in range(3):
    model.train()
    total_train_loss = 0.0

    for _, tabular, labels in train_loader:
        tabular = tabular.to(device)
        labels = labels.to(device)

        preds = model(tabular)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

# evaluation
model.eval()
total_test_loss = 0.0

with torch.no_grad():
    for _, tabular, labels in test_loader:
        tabular = tabular.to(device)
        labels = labels.to(device)

        preds = model(tabular)
        loss = criterion(preds, labels)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss = {avg_test_loss:.4f}")