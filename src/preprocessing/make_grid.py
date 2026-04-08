from pathlib import Path
import pandas as pd
import numpy as np


def generate_grid(lat_min, lat_max, lon_min, lon_max, step=0.005, random_seed=42):
    np.random.seed(random_seed)

    lats = np.arange(lat_min, lat_max + step, step)
    lons = np.arange(lon_min, lon_max + step, step)

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    rows = []
    idx = 0

    for lat in lats:
        for lon in lons:
            # Distance from grid point to city center
            distance_to_center = np.sqrt(
                (lat - center_lat) ** 2 + (lon - center_lon) ** 2
            )

            # Simulated population density: higher near city center
            population = 1000 * np.exp(-distance_to_center * 20)

            # Simulated PM2.5: correlated with population + random noise
            noise = np.random.normal(0, 5)
            pm25 = 0.05 * population + noise

            rows.append(
                {
                    "id": idx,
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                    "distance_to_center": round(float(distance_to_center), 6),
                    "population": round(float(population), 3),
                    "image_path": f"data/raw/images/{idx}.png",
                    "pm25": round(float(pm25), 3),
                }
            )

            idx += 1

    return pd.DataFrame(rows)


def main():
    lat_min, lat_max = 52.62, 52.68
    lon_min, lon_max = 5.02, 5.10
    step = 0.005

    df = generate_grid(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        step=step,
        random_seed=42,
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(f"Number of samples: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()