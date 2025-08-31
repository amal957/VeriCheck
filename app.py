from flask import Flask, render_template, request, jsonify
import requests
import joblib
import re
import os
from datetime import datetime

app = Flask(__name__)


NEWS_API_KEY = 'd9df2d4f462a47ae987f6df492ca1e6e'

# Load ML models for fake news detection
def load_model(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"Model file not found: {file_path}")
            return None
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading model {file_path}: {str(e)}")
        return None

# Load models at startup
passive_model = load_model('models/pac_model.pkl')
tfidf_vectorizer = load_model('models/tfidf_vectorizer.pkl')

def preprocess_input(text):
    """Preprocess text for fake news detection"""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = text.lower()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/news')
def get_news():
    """API endpoint to fetch news"""
    query = request.args.get('q', '')
    country = request.args.get('country', 'us')
    category = request.args.get('category', '')

    try:
        if query:
            url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
        else:
            url = f'https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={NEWS_API_KEY}'

        response = requests.get(url)
        data = response.json()

        if data.get('status') != 'ok':
            return jsonify({'error': data.get('message', 'Unknown error')}), 400

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_article():
    """API endpoint for fake news detection"""
    try:
        data = request.get_json()
        article_text = data.get('text', '')

        if not article_text:
            return jsonify({'error': 'No text provided'}), 400

        # Check if models are loaded
        if not passive_model or not tfidf_vectorizer:
            return jsonify({'error': 'ML models not available'}), 500

        # Preprocess and analyze
        processed_text = preprocess_input(article_text)
        input_tfidf = tfidf_vectorizer.transform([processed_text])
        
        # Get prediction
        prediction = passive_model.predict(input_tfidf)[0]

        result = {
            'prediction': 'real' if prediction == 1 else 'fake',
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)