#Modified from Campusewire#969
import pandas as pd
import requests
import numpy as np

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

R_all = pd.read_csv("https://github.com/shiz4-illinois/cs598_psl_proj4_app/raw/refs/heads/main/rating.csv", index_col=0)
S_df = pd.read_csv("https://github.com/shiz4-illinois/cs598_psl_proj4_app/raw/refs/heads/main/similarity_matrix_top30_100_movies.csv", index_col=0)
popularity_ranking = pd.read_csv("https://github.com/shiz4-illinois/cs598_psl_proj4_app/raw/refs/heads/main/popularity_ranking.csv")

# Create a DataFrame from the movie data
movies_all = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies_all['movie_id'] = movies_all['movie_id'].astype(int)
movies_all['movieId_with_m'] = 'm' + movies_all['movie_id'].astype(str)
movies = movies_all[movies_all['movieId_with_m'].isin(S_df.columns)]
R = R_all[S_df.columns]

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(new_user_ratings):
    recommended_movies = myIBCF(new_user_ratings)
    return recommended_movies



def myIBCF(newuser_input, R = R, S_df = S_df, popularity_ranking = popularity_ranking):

    predictions = {}

    # Filter the newuser dictionary to only include movies present in S_df
    common_movie_ids = {f"m{movie_id}" for movie_id in newuser_input.keys()}.intersection(set(S_df.index))

    # Initialize a new user vector with NaN values
    newuser = pd.Series(np.nan, index=R.columns)

    for movie_id in common_movie_ids:
        newuser[movie_id] = newuser_input[int(movie_id[1:])]  # Strip 'm' prefix and use the int key from newuser

    for movie in R.columns:

        if pd.isna(newuser[movie]):  # Only predict for unrated movies
            S_i = S_df.loc[movie]
            rated_movies = [m for m in R.columns if not pd.isna(newuser[m])]

            # Filter similarity values for rated movies
            S_i_filtered = S_i[rated_movies].dropna()

            if len(S_i_filtered) > 0:
                # Numerator and denominator of the prediction formula
                numerator = np.sum(S_i_filtered * newuser[S_i_filtered.index])
                denominator = np.sum(S_i_filtered)

                # Avoid division by zero
                if denominator != 0:
                    predictions[movie] = numerator / denominator

    # Sort predictions by rating in descending order
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    # Get top 10 predicted movies
    top10_predictions = [(f"m{movie}", rating) for movie, rating in sorted_predictions[:10]]

    # Fill remaining slots with popular movies if necessary
    if len(top10_predictions) < 10:
        already_rated_or_recommended = set([movie for movie, _ in top10_predictions] + [f"m{m}" for m in rated_movies])
        for _, row in popularity_ranking.iterrows():
            if row['movie'] not in already_rated_or_recommended:
                top10_predictions.append((f"mm{row['movie']}", None))
                if len(top10_predictions) == 10:
                    break


    # Create DataFrame for top 10 recommendations
    top10_df = pd.DataFrame(top10_predictions, columns=['movie_id', 'predicted_rating'])
    top10_df['movie_id'] = top10_df['movie_id'].str[2:].astype(int)  # Convert "m{movieId}" back to integer

    # Merge with movies metadata

    top10_df = top10_df.merge(movies_all, on='movie_id', how='left')

    # Rearrange columns for clarity
    top10_df = top10_df[['movie_id', 'title', 'genres', 'predicted_rating']]

    return top10_df