import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data.dataset import HoornImageDataset

# reproducibility
torch.manual_seed(42)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# dataset
dataset = HoornImageDataset(
    "data/processed/dataset.csv",
    transform=transform
)

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


class MultiModalNet(nn.Module):
    def __init__(self, tabular_dim=1):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 8),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(32 + 8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, image, tabular):
        image_feat = self.image_encoder(image)
        tabular_feat = self.tabular_encoder(tabular)

        fused = torch.cat([image_feat, tabular_feat], dim=1)
        out = self.fusion(fused)

        return out.squeeze(1)


model = MultiModalNet(tabular_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop
for epoch in range(3):
    model.train()
    total_train_loss = 0.0

    for images, tabular, labels in train_loader:
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        preds = model(images, tabular)
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
    for images, tabular, labels in test_loader:
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        preds = model(images, tabular)
        loss = criterion(preds, labels)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss = {avg_test_loss:.4f}")