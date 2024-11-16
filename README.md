
Recommendo is a recommendation system that suggests movies, songs, and books based on the user's mood. It combines computer vision and text analysis, using a deep learning model for facial recognition and natural language processing for mood detection. An added feature of Recommendo is an interactive chatbot that users can engage with for personalized recommendations, enhancing the overall experience. Once the mood is detected, users can choose a category, and Recommendo maps the mood to relevant genres for tailored suggestions. With curated datasets, it provides contextually accurate content, showcasing AI and machine learning for an engaging, user-friendly platform.
Text Emotion Detection
Dataset :
For this emotion detection task, I have used the famous Emotion Dataset, as this dataset is easily available dataset on Kaggle with each text labeled correctly, with multiple classes of emotion for the detection of the emotions. There are a total 6 classes of emotion as a target in this dataset.
Model:
I have used a basic LSTM model followed by an embedding layer as a model followed by spatial dropout as regularization.
Results:
Using the above model, I am able to get around 92% accuracy, with a minimum of 80% as the precision score for all of the classes except the surprise class, get a precision score of 0.72 here.
Face Emotion Detection:
For this Face Emotion Detection using open-cv we have used the dataset Fer 2013 on Kaggle.
Model:
I have trained a CNN for identifying the mood of the user using facial features.
Result:
The model is trained with the batch size 64 and epoch 50. We are able to achieve an accuracy of 90%.
