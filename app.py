# Required imports: flask, json, os, sklearn.metrics.pairwise, sklearn.feature_extraction.text
from flask import Flask, request, jsonify
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load all movies from JSON files in the 'api' folder
def load_movies_from_json(folder_path):
    movies = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                movies.extend(json.load(f))
    return movies

# Content-based filtering using cosine similarity on movie descriptions (extract field)
def content_based_filtering(movies, target_title):
    try:
        # Filter out movies without 'extract'
        movies = [movie for movie in movies if 'extract' in movie]


        
        # Create lists of descriptions and titles
        descriptions = [movie.get('extract', '') for movie in movies]
        titles = [movie.get('title', 'Unknown Title') for movie in movies]
        # Normalize to lowercase for case-insensitive matching
        titles_lower = [title.lower() for title in titles]
        target_title_lower = target_title.lower()

        # Check if target_title exists
        if target_title_lower not in titles_lower:
            print(f"Error: The title '{target_title}' is not in the dataset.")
            return []

        # Find the index of the target movie by title
        target_idx = titles_lower.index(target_title_lower)


        # Initialize the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(descriptions)

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()

        # print(f"Movies: {len(movies)}  \n Descriptions: {len(descriptions)}  \n Titles: {len(titles)} \n cosine_similarities: {len(cosine_similarities)}, vectorizer: {type(vectorizer)}, tfidf_matrix: {type(tfidf_matrix)}")

        # Get indices of the top 4 most similar movies
        similar_indices = cosine_similarities.argsort()[-8:-1][::-1]
        # print(f"similar_indices: {similar_indices}")
        # similar_indices = cosine_similarities

        return [titles[i] for i in similar_indices]
    except Exception as e:
        print(f"Error in content_based_filtering: {e}")
        return []
@app.route('/')
def home():
    return "Hello, World!"

# Flask route to handle POST requests for movie recommendations
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    data = request.get_json()
    movie_title = data.get('title')

    if not movie_title:
        return jsonify({"error": "No title provided"}), 400

    folder_path = './json'
    movies = load_movies_from_json(folder_path)

    recommended_titles = content_based_filtering(movies, movie_title)

    if not recommended_titles:
        return jsonify({"error": "Movie not found or no recommendations available"}), 404

    return jsonify({"recommended_movies": recommended_titles}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(port=8000, debug=True)