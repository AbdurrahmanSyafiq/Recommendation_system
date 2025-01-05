import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the trained model
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movie, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size, embeddings_initializer="he_normal", embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.movie_embedding = tf.keras.layers.Embedding(
            num_movie, embedding_size, embeddings_initializer="he_normal", embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.movie_bias = tf.keras.layers.Embedding(num_movie, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)

# Enable eager execution explicitly (if necessary)
tf.config.run_functions_eagerly(True)
with tf.keras.utils.custom_object_scope({'RecommenderNet': RecommenderNet}):
    loaded_model = tf.keras.models.load_model('recommender_model_savedd.h5', compile=False)

# Load dataset (merged_df and sample_df must be preprocessed beforehand)
sample_df = pd.read_csv("sample_df.csv")

# Streamlit App
st.title("Movie Recommender System")

st.sidebar.header("User Input")

# Dropdowns for movie selection
st.sidebar.subheader("Select 5 Movies You Like")
movies_selected = []
all_movies = sample_df['title'].unique()

for i in range(5):
    selected_movie = st.sidebar.selectbox(f"Movie {i+1}", all_movies, key=f"movie_{i+1}")
    movies_selected.append(selected_movie)

if st.sidebar.button("Get Recommendations"):
    # Get movie IDs for selected movies
    manual_movies_df = sample_df[sample_df['title'].isin(movies_selected)][['movieId', 'title']].drop_duplicates()

    # Movies not visited
    movie_not_visited = sample_df[~sample_df['movieId'].isin(manual_movies_df['movieId'].values)]['movieId']
    movie_not_visited = list(set(movie_not_visited))

    # Create user-movie array
    user_id = 1  # Dummy user ID for demonstration
    user_movie_array = np.array([[user_id, movie_id] for movie_id in movie_not_visited])

    # Predict ratings
    ratings = loaded_model.predict(user_movie_array).flatten()

    # Top 10 recommendations
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [movie_not_visited[x] for x in top_ratings_indices]
    recommended_movies = sample_df[sample_df['movieId'].isin(recommended_movie_ids)].drop_duplicates(subset=['movieId', 'title'])

    # Display recommendations
    st.write("### Movies You Selected")
    st.table(manual_movies_df)

    st.write("### Recommended Movies")
    st.table(recommended_movies[['movieId', 'title']])
else:
    st.write("### Select movies from the sidebar to get recommendations.")
