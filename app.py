# app.py - Simple Flask App for Book Recommendations

from flask import Flask, render_template, request, jsonify
import pandas as pd
import boto3
import io
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# S3 Configuration - Set these as environment variables in EC2
S3_BUCKET = "recommendation-bucket-mehreen"  # Recommendation bucket name

# Global variables to store data
books_df = None
books_rating = None
tfidf_matrix = None
tfidf_vectorizer = None

def load_data_from_s3():
    """Load CSV files from S3 bucket"""
    global books_df, books_rating, tfidf_matrix, tfidf_vectorizer
    
    try:
        s3 = boto3.client('s3')
        
        # Load books_data.csv
        print("Loading books_data.csv...")
        response = s3.get_object(Bucket=S3_BUCKET, Key='books_data.csv')
        books_df = pd.read_csv(io.BytesIO(response['Body'].read()), low_memory=False)
        
        # Load books_rating.csv
        print("Loading books_rating.csv...")
        response = s3.get_object(Bucket=S3_BUCKET, Key='books_rating.csv')
        books_rating = pd.read_csv(io.BytesIO(response['Body'].read()), low_memory=False)
        
        # Sample data for faster processing (use first 10,000 rows)
        books_rating_sample = books_rating.head(10000)
        
        # Build TF-IDF for content-based recommendations
        print("Building TF-IDF matrix...")
        books_df['content'] = books_df['Title'].fillna('') + ' ' + \
                             books_df['authors'].fillna('') + ' ' + \
                             books_df['description'].fillna('')
        
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['content'].fillna(''))
        
        print("Data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get book recommendations for a user"""
    try:
        user_id = request.json.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        # Get user's rated books
        user_ratings = books_rating[books_rating['User_id'] == user_id]
        
        if len(user_ratings) == 0:
            # Return popular books if user has no ratings
            popular_books = books_rating['Title'].value_counts().head(5).index.tolist()
            return jsonify({'recommendations': popular_books})
        
        # Get books user has rated
        rated_titles = user_ratings['Title'].tolist()
        rated_indices = books_df[books_df['Title'].isin(rated_titles)].index.tolist()
        
        if len(rated_indices) == 0:
            popular_books = books_rating['Title'].value_counts().head(5).index.tolist()
            return jsonify({'recommendations': popular_books})
        
        # Get content-based recommendations
        user_profile = tfidf_matrix[rated_indices].mean(axis=0)
        similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
        
        # Get top recommendations (excluding already rated)
        similarity_df = pd.DataFrame({
            'title': books_df['Title'],
            'score': similarity_scores
        })
        
        recommendations = similarity_df[
            ~similarity_df['title'].isin(rated_titles)
        ].nlargest(10, 'score')
        
        return jsonify({
            'recommendations': recommendations['title'].tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    """Get dataset statistics"""
    return jsonify({
        'total_books': len(books_df),
        'total_ratings': len(books_rating),
        'unique_users': books_rating['User_id'].nunique()
    })

if __name__ == '__main__':
    # Load data on startup
    load_data_from_s3()
    app.run(host='0.0.0.0', port=5000, debug=False)