import numpy as np
import pandas as pd
#mood mapping
# Mood mapping using a dictionary
mood_map = {
    "0": "Angry",
    "1": "Disgusting",
    "2": "Fear",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprised",
    "6": "Neutral"
}


df_action = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/action.csv")
df_adventure = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/adventure.csv")
df_animation = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/animation.csv")
df_biography = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/biography.csv")
df_crime = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/crime.csv")
df_family = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/family.csv")
df_fantasy = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/fantasy.csv")
df_film_noir = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/film-noir.csv")
df_history = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/history.csv")
df_horror = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/horror.csv")
df_mystery = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/mystery.csv")
df_romance = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/romance.csv")
df_scifi = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/scifi.csv")
df_sports = pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/movie_dataset/sports.csv")
df_thriller = pd.read_csv("thriller.csv")
df_war = pd.read_csv("war.csv")

genre_to_dataframe = {
    "action": df_action,
    "adventure": df_adventure,
    "animation": df_animation,
    "biography": df_biography,
    "crime": df_crime,
    "family": df_family,
    "fantasy": df_fantasy,
    "film-noir": df_film_noir,
    "history": df_history,
    "horror": df_horror,
    "mystery": df_mystery,
    "romance": df_romance,
    "scifi": df_scifi,
    "sports": df_sports,
    "thriller": df_thriller,
    "war": df_war
}

mood_genre_mapping = {
    "Angry": ["action", "thriller", "crime", "war"],
    "Disgusting": ["horror", "thriller"],
    "Fear": ["horror", "mystery", "scifi"],
    "Happy": ["comedy", "family", "romance", "animation"],
    "Sad": ["drama", "romance", "biography"],
    "Surprised": ["thriller", "mystery", "fantasy"],
    "Neutral": ["biography", "documentary", "history"]
}
def select_mood(mood_map):
  print("""0": "Angry",
    "1": "Disgusting",
    "2": "Fear",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprised",
    "6": "Neutral"""
    )
  while True:
        try:
            user_input = int(input("Enter a mood code: "))
            if user_input < 0 or user_input > 6:
                raise ValueError("Mood code must be between 0 and 6.")
            return mood_map[str(user_input)]
        except ValueError as e:
            print(e)

mood = select_mood(mood_map)
print(f"Selected mood: {mood}")
def recommend_movies(mood, mood_genre_mapping, genre_to_dataframe):
    genres = mood_genre_mapping.get(mood, [])
    recommended_movies = pd.DataFrame()

    for genre in genres:
        if genre in genre_to_dataframe:
            df_genre = genre_to_dataframe[genre]
            # Keep all columns for sorting
            genre_movies = df_genre.copy()
            recommended_movies = pd.concat([recommended_movies, genre_movies])

    # Sort by rating and votes before selecting relevant columns
    recommended_movies = recommended_movies.sort_values(by=['rating', 'votes'], ascending=[False, False])

    # Now filter to keep only the relevant columns
    relevant_columns = ['movie_name', 'year', 'certificate', 'star']
    recommended_movies = recommended_movies[relevant_columns].drop_duplicates(subset=['movie_name']).reset_index(drop=True)

    return recommended_movies.head(5)

recommendations = recommend_movies(mood, mood_genre_mapping, genre_to_dataframe)
print(f"Top 5 movie recommendations for mood '{mood}':")
print(recommendations)
def recommend():
    mood = select_mood(mood_map)
    recommendations = recommend_movies(mood, mood_genre_mapping, genre_to_dataframe)
    print(f"Top 5 movie recommendations for mood '{mood}':")
    print(recommendations)

def recommend1(mood):
    
    recommendations = recommend_movies(mood, mood_genre_mapping, genre_to_dataframe)
    print(f"Top 5 movie recommendations for mood '{mood}':")
    print(recommendations)

def start():
    print("Welcome to Movie Recommendation System")
    print("Please select an option:")
    print("1: get recommendations ")
    print("2: exit")
    option = input("Enter your option:")
    if option == "1":
        recommend()
    elif option == "2":
        print("Thank you for using Movie Recommendation System")
    else:
        print("Invalid option")
        start()

start()
