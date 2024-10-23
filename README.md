## Personalized Movie Recommendation System
## Overview
The Personalized Movie Recommendation System provides tailored movie recommendations based on user preferences. The system uses Generative Adversarial Networks (GANs) and autoencoders to learn user-item interactions and predict movie ratings. The project includes a FastAPI backend that delivers real-time recommendations by taking user IDs as input and returning personalized movie suggestions.

## Features
Generative Adversarial Network (GAN) for generating personalized movie recommendations.
Autoencoder for learning compressed user preferences and predicting movie ratings.
Data Preprocessing using scikit-learn pipelines to handle numerical and categorical data.
FastAPI backend for real-time API-based movie recommendations.
Model evaluation using MSE, RMSE, and MAE.

## Project Structure
bash
.
├── main.py                # FastAPI app file
├── utils.py               # Utilities for data loading, model predictions
├── models/
│   ├── generator_model.h5 # Trained GAN model
│   ├── latent_vectors.npy # Latent vectors for users
├── dataset/
│   ├── Dataset.csv        # MovieLens dataset
│   ├── Movie_Id_Titles.csv # Movie ID and Title mappings
├── notebooks/             # Jupyter Notebooks for model development
│   └── model_training.ipynb # GAN and autoencoder model training
├── all_pipeline.pkl       # Preprocessing pipeline
└── README.md              # Project readme (this file)
## Requirements
To run this project, you will need:

Python 3.7+
TensorFlow
Keras
FastAPI
Scikit-learn
Pandas
NumPy
Joblib
Uvicorn (for running FastAPI)
You can install all required packages by running:

## bash
pip install -r requirements.txt
## Dataset
The system uses the MovieLens dataset (small version), which contains movie ratings and user interactions. The dataset is preprocessed using scikit-learn pipelines.

You can download the dataset from MovieLens and place it in the dataset/ folder.

## Running the Project
## 1. Model Training
To train the recommendation models (GAN and autoencoder), use the provided Jupyter notebooks in the notebooks/ folder.

## bash
jupyter notebook notebooks/model_training.ipynb
## 2. API Deployment
After training the models, you can run the FastAPI server to deploy the recommendation system.

bash
uvicorn main:app --reload
Health Check: Access the API health check by navigating to http://127.0.0.1:8000/.
Movie Recommendations: Get movie recommendations by sending a POST request to /predict with a user ID.
Example:
POST /predict
Body:

## json
{
  "user_id": 1
}
The response will contain a list of recommended movies for the given user ID.

## 3. Evaluating Models
Use the evaluate_gan() function within the notebooks to evaluate the trained GAN model on the test data. Metrics such as MSE, RMSE, and MAE are calculated to assess model performance.

## File Descriptions
main.py: Contains the FastAPI server setup and the endpoints for making predictions.
utils.py: Includes utility functions for model predictions, data loading, and mapping movie IDs to titles.
generator_model.h5: The trained GAN generator model.
latent_vectors.npy: The saved latent vectors that represent user preferences.
all_pipeline.pkl: Preprocessing pipeline for handling the dataset transformations.
model_training.ipynb: Jupyter notebook that contains code for training GAN and autoencoder models.
