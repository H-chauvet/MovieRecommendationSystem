import pandas as pd
from surprise import Dataset, Reader, SVDpp, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

# Charger les données MovieLens
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Lecture des données en utilisant Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Division des données en ensemble d'entraînement et de test
trainset, testset = train_test_split(data, test_size=0.25)

# Définir les hyperparamètres à tester pour SVD++
param_grid = {
    'n_factors': [100, 150, 200],
    'n_epochs': [20, 30, 40],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

# Effectuer une recherche en grille avec validation croisée
gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

# Afficher les meilleurs paramètres trouvés
print(gs.best_params['rmse'])

# Utiliser les meilleurs paramètres pour entraîner le modèle
algo = gs.best_estimator['rmse']
algo.fit(trainset)

# Prédiction sur l'ensemble de test
predictions = algo.test(testset)

# Évaluation du modèle avec le RMSE
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# Fonction pour obtenir le nom du film à partir de l'ID
def get_movie_name(movie_id):
    return movies[movies['movieId'] == movie_id]['title'].values[0]

# Trouver un utilisateur présent dans l'ensemble d'entraînement
user_id = None
for uid in ratings['userId'].unique():
    if trainset.knows_user(uid):
        user_id = uid
        break

if user_id is not None:
    # Affichage de quelques recommandations pour cet utilisateur
    items = [iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]]
    predictions = [algo.predict(user_id, iid) for iid in items]
    predictions.sort(key=lambda x: x.est, reverse=True)

    print(f"Top 10 recommandations de films pour l'utilisateur {user_id}:")
    for pred in predictions[:10]:
        movie_name = get_movie_name(pred.iid)
        print(f'Film: {movie_name} - Score prédit: {pred.est}')
else:
    print("Aucun utilisateur trouvé dans l'ensemble d'entraînement.")
