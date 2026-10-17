# Logistics Data Analytics — Week 2

## Task
**Data Collection, Cleaning, and Preprocessing for Logistics Analysis**

Project title: **Logistics Data Analytics for Delivery Time Prediction and Cost Optimization Using Python**

## What is included?
- `data/logistics_raw_sample.csv` — a small simulated logistics dataset with examples of missing values, an exact duplicate, an invalid date order, a negative delivery time, and an unusually large record.
- `preprocess_logistics.py` — Python script demonstrating data validation, duplicate removal, missing-value handling, IQR outlier flags, Min-Max scaling, and one-hot encoding.
- `requirements.txt` — required Python packages.
- `outputs/` — destination folder for generated results.

## Important
The included CSV is **simulated sample data**, not real company data. Do not claim that it represents actual deliveries. Replace it with an authorized dataset if your internship task requires real-data analysis. The script demonstrates a workflow; inspect its output and review flagged rows before using data for modelling.

## Run the project
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python preprocess_logistics.py
   ```
5. Find generated files in `outputs/`.

## Main preprocessing steps
1. Load and inspect the CSV using Pandas.
2. Parse date columns and convert numeric columns.
3. Remove exact duplicate rows.
4. Treat negative distance/time/cost values as missing for review.
5. Flag date sequences where delivery date precedes order date.
6. Fill numeric missing values with medians as a demonstration.
7. Flag potential outliers using the Interquartile Range (IQR) rule.
8. Scale numeric columns using MinMaxScaler.
9. One-hot encode vehicle type and delivery status.

## Suggested repository description
Week 2 internship task demonstrating data collection, cleaning, and preprocessing for logistics analysis using Python, Pandas, and Scikit-learn.
