import numpy as np
import pandas as pd
df=pd.read_csv('C:\Users\hp\Desktop\Projects\movie_recommendation\book_data\BooksDataSet.csv')
mood_map = {
    "0": "Angry",
    "1": "Disgusting",
    "2": "Fear",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprised",
    "6": "Neutral"
}
unique_genres =df['genre'].unique()
num_genres = len(unique_genres)
mood_genre_mapping={
    'Angry':'Thriller',
    'Disgusting':'Thriller',
    'Fear':'Crime Fiction',
    'Happy':'Fantasy',
    'Sad':'Fantasy',
    'Surprised':'Science Fiction',
    'Neutral':'Historic novel'
}
for genre in unique_genres:
        globals()[f'{genre.replace(" ", "_")}_df'] = df[df['genre'] == genre]
genre_to_dataframe={
'Fantasy':Fantasy_df,
'Science Fiction':Science_Fiction_df,
'Crime Fiction':Crime_Fiction_df,
'Historical novel':Historical_novel_df,
'Horror':Horror_df,
'Thriller':Thriller_df
}

def recommend_books_based_on_mood(mood):
    # Check if the mood exists in the mood_to_genre mapping
    if mood in mood_genre_mapping:
        genre = mood_genre_mapping[mood]  # Get the genre based on the mood
        genre_df_name = f'{genre.replace(" ", "_")}_df'
        
        # Get the corresponding dataframe for the genre
        if genre_df_name in globals():
            genre_df = globals()[genre_df_name]
            
            # Randomly select 5 books from the genre dataframe (ensure there are at least 5 books)
            num_books_to_select = min(5, len(genre_df))  # Select up to 5 books, or fewer if there aren't 5
            books = genre_df.sample(n=num_books_to_select)
            
            # Create a formatted string with the recommended books
            recommendation = f"Recommended books based on your mood '{mood}':\n"
            for index, book in books.iterrows():
                recommendation += f"\nBook Name: {book['book_name']}\n" \
                                  f"Genre: {book['genre']}\n" \
                                  f"Summary: {book['summary']}\n"
            return recommendation
        else:
            return f"No books found for the genre '{genre}'"
    else:
        return "Mood not found. Please enter a valid mood."


mood_input = 'Angry'  
print(recommend_books_based_on_mood(mood_input))



