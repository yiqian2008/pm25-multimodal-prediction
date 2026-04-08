# PM2.5 Prediction using Multimodal Learning

## Overview

Air pollution, particularly PM2.5, poses significant risks to public health. This project develops a multimodal machine learning pipeline that integrates satellite imagery and structured environmental data to predict PM2.5 levels.

The goal is to explore how combining spatial information from images with structured variables (e.g., weather conditions) can improve predictive performance compared to single-modality approaches.

---

## Methodology

The pipeline consists of the following components:

1. **Data Collection**

   * Satellite image retrieval
   * Environmental and weather data integration

2. **Data Preprocessing**

   * Image cleaning and validation
   * Grid-based image structuring
   * Tabular data preprocessing

3. **Feature Extraction**

   * Image-based feature extraction using deep learning models
   * Structured feature engineering from environmental variables

4. **Modeling**

   * Baseline model (tabular data only)
   * Multimodal model (image + tabular fusion)

5. **Evaluation**

   * Performance comparison between baseline and multimodal models
   * Metrics such as RMSE, MAE, and R²

---

## Project Structure

```
pm25-multimodal-prediction/
│
├── src/
│   ├── data/
│   │   ├── download_images.py
│   │   ├── check_images.py
│   │   └── dataset.py
│   │
│   ├── preprocessing/
│   │   └── make_grid.py
│   │
│   ├── models/
│   │   ├── train_baseline.py
│   │   └── train_multimodal.py
│   │
│   └── utils/
│
├── notebooks/
├── results/
├── README.md
└── requirements.txt
```

---

## Key Features

* End-to-end machine learning pipeline
* Image-based feature extraction
* Multimodal learning (image + tabular data fusion)
* Baseline vs multimodal model comparison
* Modular and extensible project structure

---

## Results

*(Add your results here, for example:)*

* Baseline Model:

  * RMSE: XX
  * MAE: XX
  * R²: XX

* Multimodal Model:

  * RMSE: XX
  * MAE: XX
  * R²: XX

---

## Future Work

* Improve image representation learning
* Incorporate temporal modeling for time-series data
* Explore advanced multimodal fusion techniques
* Extend the framework to other domains such as medical imaging and clinical prediction

---

## Tech Stack

* Python
* PyTorch
* NumPy / Pandas
* Matplotlib / Seaborn

---

## Author

Your Name

---

## Notes

This project focuses on building a general multimodal prediction framework that can be adapted to different domains where both image and structured data are available.

