from flask import Flask,render_template,request
from recommender import recommend_random,recommend_with_NMF,recommend_neighborhood
# from utils import movies
from utils import loaded_model as model
from utils import Ratings

app = Flask(__name__)

@app.route('/')
def hello():
    # print(movies.title.to_list())
    return render_template('index.html', name="Simantini's Movie Recommender Page", movies=model.feature_names_in_)

@app.route('/movies')
def recommendation():
    print(request.args)
    titles = request.args.getlist('title')
    ratings = request.args.getlist('rating')
    user_name = request.args.get('user_name')
    query = dict(zip(titles,ratings))

    for movie in query:
        query[movie] = float(query[movie])
    
    print(query)

    if request.args.get('option') =='Random':
        recommendation_list = recommend_random(Ratings)
        print(recommendation_list)
        return render_template('recommend.html', recommendation=recommendation_list)
    
    if request.args.get('option')=='NMF':
        recommendation_list = recommend_with_NMF(query, model, Ratings)
        
        print(query)
        return render_template('recommend.html', recommendation=recommendation_list)
    
    else:
        recommendation_list = recommend_neighborhood(Ratings,user_name, query)
        return render_template('recommend.html', recommendation=recommendation_list)

if __name__=='__main__':
    app.run(port=5000,debug=True)
    
