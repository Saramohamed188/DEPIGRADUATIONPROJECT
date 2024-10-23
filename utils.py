import numpy as np
import pandas as pd
import os
import joblib
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model  


movies_path = os.path.join(os.getcwd(),'dataset', 'Movie_Id_Titles.csv')
movies_data = pd.read_csv(movies_path)

movie_id_to_name = pd.Series(movies_data['title'].values, index=movies_data['item_id']).to_dict()

pipeline_path = os.path.join(os.getcwd(), 'all_pipeline.pkl')
model_path = os.path.join(os.getcwd(), 'generator_model.h5')

pipe = joblib.load(pipeline_path) 
model = load_model(model_path)

latent_vectors_path = os.path.join(os.getcwd(), 'latent_vectors.npy')
latent_vectors = np.load(latent_vectors_path)

class MovieData(BaseModel):
    user_id: int = Field(..., description="Unique ID of the user")

def get_latent_vector(user_id: int) -> np.ndarray:
    return latent_vectors[user_id - 1]  



def predict_new(data: MovieData) -> list:
    user_id = data.user_id  
    latent_vector = get_latent_vector(user_id)
    
    generated_ratings = model(np.expand_dims(latent_vector, axis=0)).numpy().flatten()
    
    top_movie_indices = np.argsort(generated_ratings)[::-1][:15]  
    
    recommended_movie_names = [movie_id_to_name[list(movie_id_to_name.keys())[i]] for i in top_movie_indices]
    
    return recommended_movie_names

