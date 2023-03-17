"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from faker import Faker
from utils import take


def recommend_random(Ratings, k=3):
    return np.random.choice(Ratings.columns, k)
                            

def recommend_with_NMF(new_user_query, model, Ratings, k=3):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of movieIds
    """
    # 1. candidate generation
    new_user_dataframe =  pd.DataFrame(new_user_query, columns=model.feature_names_in_, index=["new_user"])
    new_user_dataframe_imputed = new_user_dataframe.fillna(Ratings.mean()) # TO DO 

    # Get the Q matrix components
    Q_matrix = model.components_

    # 2. construct new_user-item dataframe given the query

    # Get the p matrix
    P_new_user_matrix = model.transform(new_user_dataframe_imputed)
    
    # Create the new R dataframe
    R_hat_new_user_matrix = np.dot(P_new_user_matrix,Q_matrix)

    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=model.feature_names_in_,
                         index = ['new_user'])
    # 3. Ranking
    sorted_list = R_hat_new_user.transpose().sort_values(by="new_user", ascending=False).index.to_list()

    # calculate the score with the NMF model
    
    
    # 4. ranking
    
    rated_movies = list(new_user_query.keys())

    # filter out movies already seen by the user
    

    # return the top-k highest rated movie ids or titles
    
    recommended = [movie for movie in sorted_list if movie not in rated_movies][:k]

    return recommended

def recommend_neighborhood(df, new_user, query, k =3):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   

    # Transpose the df
    # df = df.T

    # Add names to all rows
    faker = Faker("en_GB")
    names = [faker.name() for i in range(0, len(df))]
    df["users"] = names
    df = df.set_index('users')
    df = df.drop(columns = ["userId"])
    #print(df.columns)

    # get user name from the query
    new_user_dataframe =  pd.DataFrame(query, columns=df.columns, index=[new_user])
    new_user_dataframe_imputed = new_user_dataframe.fillna(df.mean()) # TO DO
    #print(new_user_dataframe_imputed)

    # Combine the df and new_user_dataframe
    df_final = df.append(new_user_dataframe_imputed)
    #print(df_final)

    # Transpose the df
    df_transpose = df_final.T
    #print(df_transpose)
    
    # Find the cosine similarity
    user_similarity = cosine_similarity(df_final)
    #print(len(user_similarity))
    user_similarity = pd.DataFrame(user_similarity, columns = df_transpose.columns, index = df_transpose.columns).round(2)
    #print(user_similarity)

    # movies unseen by the user
    new_user_t = new_user_dataframe.T
    unseen_movies = new_user_t[new_user_t[new_user].isna()].index
    print(unseen_movies)

    top_five_users = user_similarity[new_user].sort_values(ascending=False).index[1:6]

    movies = []
    ratios = []

    for movie in unseen_movies:
        print(movie)
        other_users = df_transpose.columns[~df_transpose.loc[movie].isna()]
        other_users = set(other_users)

        num = 0
        den = 0
        for other_user in other_users.intersection(set(top_five_users)):
            rating = df_transpose[other_user][movie]
            sim = user_similarity[new_user][other_user]
            num = num + (rating*sim)
            den = den + sim + 0.0001

        ratio = num/den
        movies.append(movie)
        ratios.append(ratio)
    
    movies_ratios = dict(zip(movies, ratios))
    sorted_dict = take(k, dict(sorted(movies_ratios.items(), key=lambda x:x[1], reverse=True)).items())
    keys = []
    for key, value in sorted_dict:
        keys.append(key)
    print(keys)
    # recommended_movie = max(movies_ratios, key=movies_ratios.get)

    #max(stats, key=stats.get)
    return keys




