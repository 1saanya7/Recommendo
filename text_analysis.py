import numpy as np
import pandas as pd
df=pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/text_data/training.csv")
df.head()
df['label'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
df['text_length'] = df['text'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
# Visualize the distribution of text lengths
sns.histplot(df['text_length'])
plt.show()

sns.countplot(x='label', data=df)
plt.show()
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
def normalize(text):
    if not isinstance(text, str):
       text = str(text)    
    text = text.lower()   
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)    
    stop_words = set(stopwords.words('english'))
    normalized_words = [word for word in words if word not in stop_words]
    return ' '.join(normalized_words)
df['text'] = df['text'].fillna('')  
df['text'] = df['text'].apply(lambda x: normalize(x))
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
# Assuming df['text'] contains your input data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

# Define num_words based on word_index after fitting the tokenizer
word_index = tokenizer.word_index
num_words = len(word_index) + 1  # Add 1 for the padding token

# Pad sequences
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Build the model
model = Sequential()
model.add(Embedding(num_words, 100, input_length=max_length))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, max_length))
model.summary()
history=model.fit(padded_sequences, pd.get_dummies(df['label']).values, epochs=10, batch_size=32, validation_split=0.2)

# Plotting epoch vs accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
test=pd.read_csv("C:/Users/hp/Desktop/Projects/movie_recommendation/text_data/test.csv")
test.head()
test_sequences = tokenizer.texts_to_sequences(test['text'])
X_test_padded = pad_sequences(test_sequences, maxlen=max_length)  
predictions = model.predict(X_test_padded)
predicted_classes = np.argmax(predictions, axis=1)
score=model.evaluate(X_test_padded,pd.get_dummies(test['label']).values,verbose=0)
print("Accuracy_Acheieved(Test) :",score[1]*100,"%")

model=model.save("text_based_emotion_classifier.h5")