from fastapi import FastAPI
from pydantic import BaseModel
from utils import predict_new  

app = FastAPI(title='Movie Recommendation System', version='1.0.0')


@app.get('/', tags=['General'])
async def home():
    return {'status': 'Up & Running'}


class MovieData(BaseModel):
    user_id: int 

@app.post('/predict', tags=['Prediction'])
async def movie_recommendation(data: MovieData):
    pred = predict_new(data)

    return {'Prediction': pred}

