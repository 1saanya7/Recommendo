import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')
df=pd.read_csv('C:/Users/hp/Desktop/Projects/movie_recommendation/music_data/data_moods.csv')
df.rename(columns={'mood': 'song_type'}, inplace=True)
mood_map = {
    "0": "Angry",
    "1": "Disgusting",
    "2": "Fear",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprised",
    "6": "Neutral"
}
mood_music_mapping={
    'Angry':'Calm',
    'Disgusting':'Calm',
    'Fear':'Energetic',
    'Happy':'Happy',
    'Sad':'Sad',
    'Surprised':'Energetic',
    'Neutral':'Calm'
}
unique_song_types = df['song_type'].unique()
for song_type in unique_song_types:
    globals()[f'{song_type.replace(" ", "_")}_df'] = df[df['song_type'] == song_type]

music_to_dataframe={
'Happy':Happy_df,
'Sad':Sad_df,
'Energetic':'Energetic_df',
'Calm':'Calm_df'   
}


def recommend_songs_based_on_mood(mood):
    # Check if the mood exists in the mood_to_song_type mapping
    if mood in mood_music_mapping:
        song_type = mood_music_mapping[mood]  # Get the song type based on the mood
        song_type_df_name = f'{song_type.replace(" ", "_")}_df'
        
        # Get the corresponding dataframe for the song type
        if song_type_df_name in globals():
            song_type_df = globals()[song_type_df_name]
            
            # Check if there are at least 5 songs in the dataframe
            if len(song_type_df) >= 5:
                # Randomly select 5 songs from the song type dataframe
                songs = song_type_df.sample(n=5)
            else:
                # If there are less than 5 songs, return all available songs
                songs = song_type_df
            
            # Create a formatted string with the recommended songs
            recommendation = f"Recommended songs based on your mood '{mood}':\n"
            for index, row in songs.iterrows():
                recommendation += f"\nSong Name: {row['name']}\n" \
                                  f"Album: {row['album']}\n" \
                                  f"Artist: {row['artist']}\n" \
                                  f"Song Type: {row['song_type']}\n"
            return recommendation
        else:
            return f"No songs found for the song type '{song_type}'"
    else:
        return "Mood not found. Please enter a valid mood."

# Example usage
mood_input = 'Happy'  # You can change this to any mood
print(recommend_songs_based_on_mood(mood_input))