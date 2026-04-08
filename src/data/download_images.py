import pandas as pd
import requests
from pathlib import Path
from time import sleep

# read data
df = pd.read_csv("data/processed/dataset.csv")

# image
image_dir = Path("data/raw/images")
image_dir.mkdir(parents=True, exist_ok=True)


def download_image(lat, lon, save_path):
    """
    Download satellite image from Yandex Static Maps API
    """

    url = f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&size=256,256&z=15&l=sat"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                f.write(response.content)

            return True

        else:
            print(f"[ERROR] Failed ({response.status_code}): {lat}, {lon}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"[EXCEPTION] {lat}, {lon} -> {e}")
        return False


# iteration
for row in df.itertuples():
    lat = row.lat
    lon = row.lon
    path = row.image_path

    save_path = Path(path)

    if save_path.exists():
        continue

    success = download_image(lat, lon, save_path)

    if success:
        print(f"[OK] Downloaded: {save_path}")
    else:
        print(f"[SKIP] {lat}, {lon}")

    sleep(0.2)