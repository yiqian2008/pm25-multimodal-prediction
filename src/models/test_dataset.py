from src.data.dataset import HoornImageDataset

dataset = HoornImageDataset("data/processed/dataset.csv")

print(f"Dataset size: {len(dataset)}")

image, tabular, label = dataset[0]

print("\nSample inspection:")
print(f" - Image type: {type(image)}")
print(f" - Tabular features: {tabular}")
print(f" - Label (PM2.5): {label}")