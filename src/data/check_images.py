from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# read data
df = pd.read_csv("data/processed/dataset.csv")

bad_images = []
count_ok = 0


def check_image(image_path):
    if not image_path.exists():
        return False, "missing"

    try:
        with Image.open(image_path) as img:
            img.verify()

        # reopen
        with Image.open(image_path) as img:
            img.convert("RGB")

        return True, None

    except Exception as e:
        return False, str(e)


# main iteration
for row in tqdm(df.itertuples(), total=len(df)):
    image_path = Path(row.image_path)

    ok, error = check_image(image_path)

    if ok:
        count_ok += 1
    else:
        bad_images.append((str(image_path), error))


# output
print(f"Total images listed in CSV: {len(df)}")
print(f"Readable images: {count_ok}")
print(f"Bad or missing images: {len(bad_images)}")


# 
if bad_images:
    bad_df = pd.DataFrame(bad_images, columns=["image_path", "error"])
    bad_df.to_csv("data/processed/bad_images.csv", index=False)

    print("\nSaved bad image list to data/processed/bad_images.csv")

    print("\nExamples of bad images:")
    print(bad_df.head(10))