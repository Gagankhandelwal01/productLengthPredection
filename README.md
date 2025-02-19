# productLengthPredection
Amazon ML Challenge: Predicting Product Length with BERT & LightGBM
This repository contains a solution for the Amazon ML Challenge on Kaggle. The goal is to predict the length of a product using a combination of natural language processing and gradient boosting techniques. The approach leverages BERT-based text embeddings for product descriptions and other text fields, along with LightGBM for regression.



Data Loading & Splitting: The data is loaded from train.csv and split into training and test subsets.
Text Cleaning & Feature Engineering: Multiple text columns (e.g., TITLE, DESCRIPTION, BULLET_POINTS) are cleaned using regex operations that preserve numeric values and measurement units. Additionally, a custom feature extraction is performed to capture product lengths mentioned in the text.
BERT Embeddings: Pre-trained BERT (using the distilbert-base-uncased model) embeddings are generated for the cleaned text.
Feature Combination: BERT embeddings are concatenated with additional numerical features (like PRODUCT_TYPE_ID and the extracted product length).
LightGBM Regression: A LightGBM regressor, set up to utilize GPU acceleration, is trained to predict the product length.
Post-processing & Submission: Model predictions are adjusted (ensuring non-negative values and blending with extracted length where applicable) before writing a submission CSV file.
Features
Text Preprocessing: Cleans and standardizes product text data while preserving measurements.
Feature Extraction: Uses regex to extract numeric length values from product descriptions.
BERT Embeddings: Employs the distilbert-base-uncased model for robust text representation.
GPU-Accelerated Training: Leverages LightGBM’s GPU support for faster model training.
Prediction Post-processing: Ensures non-negative output values and blends predictions with extracted values.
Setup & Installation
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/yourusername/amazon-ml-challenge.git
cd amazon-ml-challenge
Create a Conda Environment (Recommended):

bash
Copy
Edit
conda create -n amazon_ml python=3.8
conda activate amazon_ml
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Ensure that your requirements.txt includes packages such as:

torch
transformers
pandas
numpy
lightgbm
tqdm
category_encoders
GPU Setup:
The code automatically checks for a GPU. If available, computations (especially for BERT embeddings and LightGBM training) will utilize the GPU.

Data Preparation
Input Data:
The code expects a file named train.csv in the repository’s root. This file should contain columns like:

TITLE
DESCRIPTION
BULLET_POINTS
PRODUCT_TYPE_ID
PRODUCT_LENGTH
PRODUCT_ID
Data Splitting:
The first 20,000 rows are used as a test subset (for internal validation or submission blending), while the remainder of the data is used for training.

Preprocessing
Text Cleaning:
The clean_text function standardizes text by converting it to lowercase, removing URLs, HTML tags, and unwanted characters—all while preserving numeric values and measurement units (both imperial and metric).

Feature Extraction:
The extract_length function uses regex to capture numerical values associated with common units (cm, inches, etc.) and computes the median value when multiple measurements are found.

BERT Embeddings:
Using the Hugging Face transformers library, text data is tokenized and passed through the distilbert-base-uncased model. Mean pooling (with attention masks) is applied to generate fixed-length embeddings.

Modeling
Feature Combination:
The final feature set is created by concatenating:

BERT embeddings
PRODUCT_TYPE_ID
extracted_length
LightGBM Regressor:
A GPU-enabled LightGBM model is used with the following key hyperparameters:

num_leaves=255
learning_rate=0.05
n_estimators=1000
Feature and bagging fractions set at 0.7
Training:
The model is trained on the combined feature set to predict PRODUCT_LENGTH.

Usage
Run the Code:
Simply execute the main script (e.g., main.py) or run the cells in the provided Jupyter Notebook.

bash
Copy
Edit
python main.py
Monitoring Progress:
The code uses tqdm to display progress bars during embedding generation.

Submission File:
After training and prediction, a submission file (submission.csv) is generated. This file contains:

PRODUCT_ID
Predicted PRODUCT_LENGTH (with blending of the extracted lengths when available)
Note: There is a placeholder for a column labeled PRODUCT REAL LENGTH in the submission creation step. Ensure this is updated or removed if not required.

Results & Submission
Output:
The final predictions are saved to submission.csv, which can then be submitted to Kaggle.

Post-processing:
Predictions are post-processed to ensure that they are non-negative, and where available, the predictions are blended with the extracted lengths from the text.

Future Improvements
Hyperparameter Tuning:
Further tuning of the LightGBM parameters could improve performance.

Advanced Feature Engineering:
Consider exploring additional features from the text or incorporating external data sources.

