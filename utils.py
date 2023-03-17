"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np
import pickle
from itertools import islice


# movies = pd.read_csv('data/movies.csv') 

with open('nmf_model1.pkl','rb') as file:
    loaded_model = pickle.load(file)

Ratings = pd.read_csv("Ratings.csv")

def movie_to_id(string_titles):
    '''
    converts movie title to id for use in algorithms'''
    
    movieID = movies.set_index('title').loc[string_titles]['movieid']
    movieID = movieID.tolist()
    
    return movieID

def id_to_movie(movieID):
    '''
    converts movie Id to title
    '''
    rec_title = movies.set_index('movieid').loc[movieID]['title']
    
    return rec_title

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))