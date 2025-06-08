from functools import lru_cache
from flask import Flask, request, jsonify, redirect
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
import logging
from pymongo.collection import Collection
import traceback
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import pandas as pd
from fuzzywuzzy import fuzz
import requests
from unidecode import unidecode
import re
import ast
import math
from collections import defaultdict
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Initialize Flask app
app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {
        "origins": "http://localhost:3000",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"]
    }}
)
bcrypt = Bcrypt(app)

# MongoDB Atlas connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# Access the database and collections
db = client["movie"]
movies_collection = db["movies"]
users_collection = db["users"]
ratings_collection = db["ratings"]

# Create indexes
movies_collection.create_index(
    [
        ("genres", TEXT),
        ("rating", DESCENDING),
        ("year", DESCENDING)
    ],
    name="genre_rating_year_index"
)

# Create compound index for mood recommendations
movies_collection.create_index([
    ("genres", ASCENDING),
    ("rating", DESCENDING),
    ("year", DESCENDING)
], name="mood_recommendation_index")

users_collection.create_index([
    ("mood_genres.feel-good.learned", ASCENDING),
    ("mood_genres.feel-good.updated", DESCENDING)
])

users_collection.create_index([("email", ASCENDING)], unique=True)
users_collection.create_index([("last_login", DESCENDING)])
users_collection.create_index([("genres", ASCENDING)])
users_collection.create_index([("preference_history.timestamp", DESCENDING)])

# Load movie data
current_dir = os.path.dirname(os.path.abspath(__file__))
MOVIE_DATA_PATH = os.path.join(current_dir, "ratings_modified.csv")

@lru_cache(maxsize=1)
def load_movies_df():
    try:
        columns = ["title", "genres", "rating", "year"]
        df = pd.read_csv(
            MOVIE_DATA_PATH,
            usecols=columns,
            dtype={'year': 'float', 'genres': 'object'},
            engine='python'
        )

        # Filter only movies from 1995 onwards 🚀
        df = df[df["year"] >= 1995]

        # Clean up titles for ID
        df['id'] = df['title'].apply(
            lambda x: re.sub(r'\W+', '', str(x).lower()).strip()
        )
        df = df.dropna(subset=['id'])
        df = df[
            (df['id'].notnull()) &
            (df['id'] != '') &
            (df['genres'].apply(lambda x: isinstance(x, str) and len(x) > 0))
        ]

        df = df.dropna(subset=["genres", "rating", "year"])
        df = df[
            (df['year'] <= pd.Timestamp.now().year + 2)
        ]

        df['year'] = df['year'].astype('Int16')
        df["genres"] = df["genres"].astype(str).str.strip().str.lower().str.split('|')
        df["genres"] = df["genres"].apply(
            lambda x: [g.strip().lower() for g in x if g.strip()]
        )
        df = df[df["genres"].apply(len) > 0]
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.drop_duplicates(subset=["title"])
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"Error loading movie data: {str(e)}")
        return pd.DataFrame()


def normalize_genres(genres):
    if isinstance(genres, str):
        return [g.strip().lower() for g in genres.split('|') if g.strip()]
    elif isinstance(genres, list):
        return [g.strip().lower() for g in genres if isinstance(g, str)]
    return []


def get_fuzzy_genre_matches(user_genres, all_movies):
    scored_movies = []

    for movie in all_movies:
        movie_genres = movie.get('genres', [])
        if isinstance(movie_genres, str):
            movie_genres = normalize_genres(movie_genres)
        elif isinstance(movie_genres, list):
            movie_genres = [g for g in movie_genres if isinstance(g, str)]
        else:
            continue  # skip invalid genre data

        score = 0
        for user_genre in user_genres:
            for genre in movie_genres:
                if isinstance(user_genre, str) and isinstance(genre, str):
                    score += fuzz.partial_ratio(user_genre.lower(), genre.lower())

        if score > 0:  # Only consider movies with non-zero match
            scored_movies.append((score, movie))

    # Sort by descending score and return top matches
    scored_movies.sort(reverse=True, key=lambda x: x[0])
    return [m[1] for m in scored_movies]

@app.route("/api/genre-recommendations", methods=["GET"])
def genre_recommendations():
    try:
        email = request.args.get("email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        user = users_collection.find_one({"email": email})
        if not user or "genres" not in user or not user["genres"]:
            return jsonify({"error": "User genres not found"}), 404

        # Normalize user genres (handle strings or dicts like {"name": "Action"})
        user_genres = [
            g["name"].strip().lower() if isinstance(g, dict) else str(g).strip().lower()
            for g in user["genres"]
        ]

        def fuzzy_match(movie_genres):
            # Normalize movie genres as strings
            normalized_movie_genres = [
                str(genre).strip().lower() for genre in movie_genres if isinstance(genre, str)
            ]
            return any(
                fuzz.partial_ratio(user_genre, movie_genre) > 80
                for user_genre in user_genres
                for movie_genre in normalized_movie_genres
            )
        movies_df = load_movies_df()
        matched_df = movies_df[movies_df["genres"].apply(fuzzy_match)]
        matched_df = matched_df.sort_values(by="rating", ascending=False)

        top_movies = matched_df.head(10)[["title", "genres", "rating", "year"]].to_dict(orient="records")
        return jsonify({"recommendations": top_movies})

    except Exception as e:
        logger.error(f"Error in /api/genre-recommendations: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@lru_cache(maxsize=1000)
def get_movie_poster(title, year=None):
    """Fetch movie poster URL from TMDB API with enhanced matching"""
    try:
        clean_title = re.sub(r'\s*(?:\(\d{4}\)|\[\w+\]|-\s*\d+).*', '', title).strip()
        if not clean_title:
            return None

        api_key = os.getenv("TMDB_API_KEY")
        base_url = "https://api.themoviedb.org/3"
        
        # First try with search parameters
        params = {
            "api_key": api_key,
            "query": clean_title,
            "include_adult": False,
            "language": "en-US",
            "region": "US"
        }
        
        if year:
            params["year"] = int(year)

        search_url = f"{base_url}/search/movie"
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Try to find exact match
        for result in data.get("results", []):
            if not result.get("poster_path"):
                continue
                
            title_similarity = fuzz.ratio(
                clean_title.lower(),
                result["title"].lower()
            )
            
            if title_similarity >= 90:
                return f"https://image.tmdb.org/t/p/w500{result['poster_path']}"

        # Fallback to first result with poster
        for result in data.get("results", []):
            if result.get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{result['poster_path']}"

        # If no results, try with discover API
        discover_params = {
            "api_key": api_key,
            "sort_by": "popularity.desc",
            "include_adult": False,
            "with_original_language": "en"
        }
        if year:
            discover_params["primary_release_year"] = int(year)
            
        discover_url = f"{base_url}/discover/movie"
        response = requests.get(discover_url, params=discover_params, timeout=10)
        response.raise_for_status()
        discover_data = response.json()
        
        for result in discover_data.get("results", []):
            if result.get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{result['poster_path']}"

        return None

    except requests.exceptions.ConnectionError:
        logger.warning(f"TMDB connection failed for {title}, using fallback")
        return "/static/fallback-poster.jpg"
    except Exception as e:
        logger.error(f"TMDB API error: {str(e)}")
        return "/static/fallback-poster.jpg"


def get_movie_details(movie_id):
    """Get movie details from local dataset"""
    try:
        movies_df = load_movies_df()
        movie = movies_df[movies_df['id'] == movie_id].iloc[0]
        return {
            "id": movie_id,
            "title": f"{movie['title']} ({int(movie['year'])})",
            "genres": "|".join(movie['genres']),
            "year": int(movie['year']),
            "rating": round(float(movie['rating']), 1),
            "poster": get_movie_poster(movie['title'], movie['year'])
        }
    except Exception as e:
        logger.error(f"Error getting details for {movie_id}: {str(e)}")
        return None

@app.route("/")
def index():
    return redirect("https://movie-recommendation-system-zqq0.onrender.com/login")

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not email or not password:
            return jsonify({"success": False, "message": "Both email and password are required"}), 400

        user = users_collection.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})

        if not user or not bcrypt.check_password_hash(user.get("password", ""), password):
            return jsonify({"success": False, "message": "Invalid email or password"}), 401
        
        if "learned_genres" not in user:
            users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"learned_genres": {}}}
            )
            user["learned_genres"] = {}

        update_data = {
            "$inc": {"login_count": 1},
            "$set": {"last_login": datetime.now(timezone.utc)}
        }

        users_collection.update_one({"_id": user["_id"]}, update_data)

        user_data = {
            "id": str(user["_id"]),
            "name": user.get("name", ""),
            "email": user["email"],
            "has_genres": len(user.get("genres", [])) > 0,
            "has_genres": len(user.get("genres", [])) > 0,
            "login_count": user.get("login_count", 0) + 1,
            "created_at": user.get("created_at", datetime.now(timezone.utc)).isoformat(),
            "last_login": datetime.now(timezone.utc).isoformat(),
            "preferences": {
                "liked_movies": user.get("liked_movies", []),
                "disliked_movies": user.get("disliked_movies", [])
            },
            "learned_genres": user.get("learned_genres", {})
        }

        return jsonify({"success": True, "message": "Login successful", "user": user_data}), 200

    except Exception as e:
        logger.error(f"Login error for {email}: {str(e)}")
        return jsonify({"success": False, "message": "Internal server error"}), 500

@app.route('/api/save-genres', methods=['POST'])
def save_genres():
    data = request.json
    email = data.get("email")
    raw_genres = data.get("genres", [])
    genres = [g.strip().lower() for g in raw_genres if g.strip()]

    if not email or not genres:
        return jsonify({"message": "Email and genres are required"}), 400

    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"message": "User not found"}), 404

    users_collection.update_one(
        {"email": email},
        {
            "$set": {"genres": genres},
            "$push": {
                "preference_history": {
                    "type": "genre",
                    "value": ",".join(genres),
                    "timestamp": datetime.now(timezone.utc)
                }
            }
        }
    )

    return jsonify({"message": "Genres saved successfully"}), 200


@app.route('/api/search', methods=['GET'])
def dictionary_search():
    try:
        # Get parameters from URL query string
        search_query = request.args.get('query', '').strip().lower()
        email = request.args.get('email')

        # Validate search query - updated to allow single character searches
        if len(search_query) < 1:
            return jsonify({"movies": []}), 200

        # Update user search history if email exists
        if email:
            users_collection.update_one(
                {"email": email},
                {
                    "$push": {
                        "preference_history": {
                            "type": "search",
                            "value": search_query,
                            "timestamp": datetime.now(timezone.utc)
                        }
                    }
                }
            )

        # Clean and prepare search terms
        def clean_title(title):
            return re.sub(r'[^a-z0-9\s]', '', 
                re.sub(r'\s*\(\d{4}\)', '', str(title)).lower().strip()
            )
        movies_df = load_movies_df()
        search_df = movies_df.copy()
        search_df['search_title'] = search_df['title'].apply(clean_title)
        search_query_clean = clean_title(search_query)

        # Enhanced matching logic with fuzzy scoring
        search_df['fuzzy_score'] = search_df['search_title'].apply(
            lambda x: fuzz.partial_ratio(search_query_clean, x)
        )

        # Match scoring logic
        search_df['match_type'] = 'none'
        search_df['match_priority'] = 0
        
        # Priority 3: Exact match
        exact_mask = search_df['search_title'] == search_query_clean
        search_df.loc[exact_mask, 'match_type'] = 'exact'
        search_df.loc[exact_mask, 'match_priority'] = 3

        # Priority 2: Starts with
        start_mask = search_df['search_title'].str.startswith(search_query_clean)
        search_df.loc[start_mask & ~exact_mask, 'match_type'] = 'prefix'
        search_df.loc[start_mask & ~exact_mask, 'match_priority'] = 2

        # Priority 1: Contains or fuzzy match
        contains_mask = search_df['search_title'].str.contains(search_query_clean, na=False)
        fuzzy_mask = search_df['fuzzy_score'] >= 70
        combined_mask = (contains_mask | fuzzy_mask) & ~exact_mask & ~start_mask
        search_df.loc[combined_mask, 'match_type'] = 'fuzzy'
        search_df.loc[combined_mask, 'match_priority'] = 1

        # Process results - increased result count
        result_df = search_df[search_df['match_priority'] > 0].sort_values(
            ['match_priority', 'fuzzy_score', 'year'],
            ascending=[False, False, False]
        ).head(30)  # Increased from 20 to 30

        valid_movies = []
        for _, row in result_df.iterrows():
            try:
                clean_title = re.sub(r'\s*\(\d{4}\)', '', row['title']).strip()
                poster = get_movie_poster(row['title'], row['year'])
                
                valid_movies.append({
                    'title': f"{clean_title} ({int(row['year'])})",
                    'poster': poster or '/fallback-poster.jpg',
                    'year': int(row['year']),
                    'rating': round(float(row['rating']), 1),
                    'genres': list(row['genres']),
                    'match_type': row['match_type'],
                    'id': row['id']
                })
            except Exception as e:
                logger.error(f"Poster error for {row['title']}: {str(e)}")
            
            if len(valid_movies) >= 20:  # Increased from 10 to 20
                break

        return jsonify({
            "movies": valid_movies,
            "search_type": "enhanced",
            "status": "success"
        }), 200

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "Search failed",
            "message": str(e)
        }), 500

@app.route('/api/user-preferences', methods=['GET'])
def get_user_preferences():
    email = request.args.get('email')
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "liked": user.get("liked_movies", []),
        "disliked": user.get("disliked_movies", []),
        "genres": user.get("genres", []),
        "learned_genres": user.get("learned_genres", {}),
        "mood_genres": user.get("mood_genres", {})
    }), 200

@app.route('/api/recent-movies', methods=['GET'])
def get_recent_movies():
    try:
        current_year = datetime.now(timezone.utc).year
        movies_df = load_movies_df()
        recent_movies = movies_df[
            (movies_df['year'] >= current_year - 5)
        ].sort_values(
            ['rating', 'year'], 
            ascending=[False, False]
        ).head(3)
        
        recent_movies = recent_movies.to_dict('records')
        for movie in recent_movies:
            movie['poster'] = get_movie_poster(movie['title'], movie.get('year'))
            
        return jsonify({"movies": recent_movies}), 200
        
    except Exception as e:
        logger.error(f"Recent movies error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
def update_learned_mood_genres(user, mood_name, genres, liked):
    if "mood_genres" not in user or mood_name not in user["mood_genres"]:
        users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {f"mood_genres.{mood_name}.learned": []}}
    )

    field = f"mood_genres.{mood_name}.learned"

    if liked:
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$addToSet": {field: {"$each": genres}}}
        )
    else:
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$pull": {field: {"$in": genres}}}
        )

def update_learned_general_genres(user, genres, liked):
    if "learned_genres" not in user:
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"learned_genres": {}}}
        )
        user["learned_genres"] = {}

    for genre in genres:
        current_weight = user.get("learned_genres", {}).get(genre, 0)
        new_weight = max(0, current_weight + (1 if liked else -1))
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {f"learned_genres.{genre}": new_weight}}
        )

@app.route('/api/rate-movie', methods=['POST'])
def rate_movie():
    try:
        data = request.get_json()
        print("Incoming rate-movie data:", data)

        email = data.get('email')
        movie_id = data.get('movie_id')
        action = str(data.get('action')).lower()
        source = data.get('source', '')

        if not email or not movie_id or action not in ['like', 'dislike']:
            return jsonify({"error": "Invalid input"}), 400

        liked = action == 'like'

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Remove from both liked and disliked
        users_collection.update_one(
            {"email": email},
            {
                "$pull": {
                    "liked_movies": movie_id,
                    "disliked_movies": movie_id
                }
            }
        )

        # Add to correct list
        users_collection.update_one(
            {"email": email},
            {
                "$addToSet": {
                    "liked_movies" if liked else "disliked_movies": movie_id
                }
            }
        )

        # Get genres from movie
        movie = movies_collection.find_one({"id": movie_id})
        if not movie or "genres" not in movie:
            return jsonify({"message": "Preference updated (no genres found)"}), 200

        genres = movie["genres"]
        if isinstance(genres, str):
            genres = [g.strip().lower() for g in genres.split('|') if g.strip()]
        else:
            genres = [g.strip().lower() for g in genres if isinstance(g, str)]

        # ✅ General genre learning (only for selected sources)
        if source in ["personalized", "hybrid", "latest"]:
            update_learned_general_genres(user, genres, liked)

        # ✅ Mood-based genre learning
        if source.startswith("mood-"):
            mood_name = source.split("-", 1)[1]
            update_learned_genres(user, genres, liked)

        return jsonify({"message": "Preference updated"}), 200

    except Exception as e:
        logger.error(f"Rating error: {str(e)}")
        return jsonify({"error": "Rating update failed"}), 500


@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        email = data.get("email", "").lower().strip()
        password = data.get("password", "")
        name = data.get("name", "").strip()

        if not all([email, password, name]):
            return jsonify({"error": "All fields required"}), 400

        if users_collection.find_one({"email": email}):
            return jsonify({"error": "Email already exists"}), 409

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        current_time = datetime.now(timezone.utc)
        
        # Initialize mood profiles
        mood_genres = {
            mood: {
                "base": config["base"],
                "learned": [],
                "updated": current_time
            }
            for mood, config in MOOD_GENRES.items()
        }
        
        user_data = {
            "email": email,
            "password": hashed_pw,
            "name": name,
            "mood_genres": mood_genres,
            "liked_movies": [],
            "disliked_movies": [],
            "created_at": current_time,
            "login_count": 0,
            "learned_genres": {},
        }
        users_collection.insert_one(user_data)
        return jsonify({"message": "Account created successfully"}), 201
        
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return jsonify({"error": "Registration failed"}), 500      

# Mood configuration matching your exact requirements
MOOD_GENRES = {
    "feel-good": {
        "base": ["comedy", "romance", "musical", "children", "animation"],
        "min_match": 0.6,
        "learn_cap": 5
    },
    "mind-bending": {
        "base": ["sci-fi", "mystery", "thriller", "drama"],
        "min_match": 0.6,
        "learn_cap": 5
    },
    "action-packed": {
        "base": ["action", "adventure", "war", "thriller", "crime"],
        "min_match": 0.6,
        "learn_cap": 5
    },
    "chill-relax": {
        "base": ["documentary", "drama", "romance", "animation"],
        "min_match": 0.6,
        "learn_cap": 5
    }
}

VALID_GENRES = {
    g.lower() for g in [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
}

# ====== CORE FUNCTIONS ======
def normalize_genres(genres):
    """Sanitize and validate genres against CSV list"""
    return list({
        g.strip().lower()
        for g in genres
        if g.strip().lower() in VALID_GENRES
    })

# In update_learned_genres function
def update_learned_genres(user, movie_genres, liked):
    """Update learned genres per mood with preferred base genres"""
    updates = {}
    current_time = datetime.now(timezone.utc)
    movie_genres = set(g.strip().lower() for g in movie_genres)
    
    for mood, config in MOOD_GENRES.items():
        base_genres = set(config["base"])
        min_match = config["min_match"]
        learn_cap = config["learn_cap"]
        
        # Calculate match ratio for this mood's base
        matched_base = movie_genres & base_genres
        total_genres = len(movie_genres)
        match_ratio = len(matched_base) / total_genres if total_genres > 0 else 0
        
        if match_ratio >= min_match:
            current_learned = set(user["mood_genres"][mood].get("learned", []))
            update_needed = False
            
            if liked:
                # Add matched base genres up to learn cap
                new_learned = current_learned.union(matched_base)
                if len(new_learned) > learn_cap:
                    # Prioritize most recent preferences
                    existing = list(current_learned)
                    new_learned = set(list(matched_base) + existing)[:learn_cap]
                update_needed = new_learned != current_learned
            else:
                # Remove disliked base genres
                new_learned = current_learned - matched_base
                update_needed = new_learned != current_learned

            if update_needed:
                updates[f"mood_genres.{mood}.learned"] = list(new_learned)
                updates[f"mood_genres.{mood}.updated"] = current_time

    if updates:
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": updates}
        )
    return bool(updates)
   
# ===== WEIGHT DECAY FUNCTION =====
def apply_weight_decay(user):
    """Safely apply genre learning weight decay with robust datetime parsing."""
    try:
        current_time = datetime.now(timezone.utc)
        updates = {}
        
        for mood, mood_data in user.get("mood_genres", {}).items():
            last_updated_raw = mood_data.get("updated")
            
            # Skip if no update timestamp
            if not last_updated_raw:
                continue

            # Normalize datetime
            try:
                if isinstance(last_updated_raw, str):
                    last_updated = datetime.fromisoformat(last_updated_raw)
                elif isinstance(last_updated_raw, datetime):
                    last_updated = last_updated_raw
                else:
                    continue
                if not last_updated.tzinfo:
                    last_updated = last_updated.replace(tzinfo=timezone.utc)
            except Exception as e:
                logger.warning(f"Invalid datetime for mood '{mood}': {last_updated_raw}")
                continue

            # Decay logic
            hours_diff = (current_time - last_updated).total_seconds() / 3600
            decay_factor = 0.97 ** (hours_diff / 24)

            learned = [
                [g[0], g[1] * decay_factor]
                for g in mood_data.get("learned", [])
                if g[1] * decay_factor > 0.1
            ]
            if learned:
                updates[f"mood_genres.{mood}.learned"] = learned
                updates[f"mood_genres.{mood}.updated"] = current_time.isoformat()
        
        if updates:
            users_collection.update_one({"_id": user["_id"]}, {"$set": updates})
        return True

    except Exception as e:
        logger.error(f"Decay error: {str(e)}")
        return False
    
@app.route('/api/liked-movies', methods=['GET'])
def get_liked_movies():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email required"}), 400

    user = users_collection.find_one({"email": email})
    if not user or "liked_movies" not in user:
        return jsonify([])

    liked_ids = user["liked_movies"]
    liked_movies = list(movies_collection.find({"id": {"$in": liked_ids}}))
    for m in liked_movies:
        m["_id"] = str(m["_id"])  # Convert ObjectId to string if needed

    return jsonify(liked_movies)

def get_personalized_recommendations(user, limit=15):
    selected_genres = [g.lower().strip() for g in user.get("genres", [])]
    learned_genres = user.get("learned_genres", {})

    def normalize(genres):
        if isinstance(genres, str):
            try:
                genres = ast.literal_eval(genres)
            except:
                genres = []
        return [g.strip().lower() for g in genres if isinstance(g, str)]

    def score_row(row):
        clean_genres = normalize(row['genres'])
        selected_match = len(set(clean_genres) & set(selected_genres))
        learned_score = sum([learned_genres.get(g, 0) for g in clean_genres])
        return pd.Series([clean_genres, selected_match, learned_score])
    
    movies_df = load_movies_df()
    movies_df[['clean_genres', 'selected_match', 'learned_score']] = movies_df.apply(score_row, axis=1)
    filtered = movies_df[(movies_df['selected_match'] > 0) | (movies_df['learned_score'] > 0)]

    sorted_df = filtered.sort_values(
        by=['learned_score', 'selected_match', 'rating', 'year'],
        ascending=[False, False, False, False]
    )

    return sorted_df.head(limit)[['id', 'title', 'clean_genres', 'year', 'rating']] \
           .rename(columns={"clean_genres": "genres"}) \
           .to_dict(orient='records')


def get_mood_recommendations(user, mood, limit=15):
    config = MOOD_GENRES[mood]

    # ✅ Normalize base and learned genres (strip + lowercase)
    base_genres = [g.strip().lower() for g in config["base"]]
    learned_genres = [g.strip().lower() for g in user["mood_genres"][mood].get("learned", [])]

    # 🔍 Optional debug logging to verify genres
    print(f"[DEBUG] Mood: {mood}")
    print(f"[DEBUG] Base genres: {base_genres}")
    print(f"[DEBUG] Learned genres: {learned_genres}")

    def normalize(genres):
        if isinstance(genres, str):
            try:
                genres = ast.literal_eval(genres)
            except:
                genres = []
        return [g.strip().lower() for g in genres if isinstance(g, str)]

    def compute_matches(row):
        clean_genres = normalize(row['genres'])
        base_match = len(set(clean_genres) & set(base_genres))
        preferred_match = len(set(clean_genres) & set(learned_genres))
        base_ratio = base_match / len(base_genres) if base_genres else 0
        return pd.Series([clean_genres, base_match, preferred_match, base_ratio])

    # Apply matching logic to the entire DataFrame
    movies_df = load_movies_df()
    movies_df[['clean_genres', 'base_match', 'preferred_match', 'base_ratio']] = movies_df.apply(compute_matches, axis=1)

    # Filter by base genre match ratio
    filtered = movies_df[movies_df['base_ratio'] >= config["min_match"]]

    # Sort based on relevance
    filtered = filtered.sort_values(
        by=['preferred_match', 'base_match', 'rating', 'year'],
        ascending=[False, False, False, False]
    )

    # Sample top recommendations
    top_n = filtered.head(limit * 2)
    if len(top_n) > limit:
        top_n = top_n.sample(limit)

    # Return cleaned-up DataFrame
    return top_n[[
        'id', 'title', 'clean_genres', 'year', 'rating', 'base_match', 'preferred_match'
    ]].rename(columns={"clean_genres": "genres"})

# ====== UPDATED ROUTES ======
  
@app.route('/api/mood-recommendations', methods=['GET'])
def mood_recommendations():
    try:
        email = request.args.get('email')
        mood = request.args.get('mood', '').lower()
        print(f"[MOOD API] Email: {email}, Mood: {mood}")

        if not email or mood not in MOOD_GENRES:
            print(f"[MOOD API] Invalid input: email={email}, mood={mood}")
            return jsonify({"error": "Invalid input"}), 400

        user = users_collection.find_one({"email": email})
        if not user:
            print(f"[MOOD API] User not found: {email}")
            return jsonify({"error": "User not found"}), 404

        recommendations = get_mood_recommendations(user, mood)
        print(f"[MOOD API] Raw recommendations: {len(recommendations)} rows")

        results = []
        for _, row in recommendations.iterrows():
            genres = normalize_genres(row.get('genres', []))
            base_matches = set(genres) & set(MOOD_GENRES[mood]["base"])
            learned_matches = set(genres) & set(user["mood_genres"][mood].get("learned", []))

            if not base_matches and not learned_matches:
                continue

            results.append({
                "id": row['id'],
                "title": f"{row['title']} ({row.get('year', 'N/A')})",
                "poster": get_movie_poster(row['title'], row.get('year')),
                "genres": genres,
                "base_match": len(base_matches),
                "learned_match": len(learned_matches),
                "rating": round(row.get('rating', 0), 1)
            })

        print(f"[MOOD API] Final mood-filtered movies: {len(results)}")

        # Only log sample genres if there are any results
        if results:
            logger.debug(f"[DEBUG] Sample genres in recommendations: {results[0]['genres']}")
        else:
            logger.debug(f"[DEBUG] No genres to sample from.")

        logger.debug(f"[DEBUG] Mood base genres: {MOOD_GENRES[mood]['base']}")

        return jsonify({
            "mood": mood,
            "base_genres": MOOD_GENRES[mood]["base"],
            "learned_genres": user["mood_genres"][mood].get("learned", []),
            "movies": sorted(
                results,
                key=lambda x: (-x['base_match'], -x['learned_match'], -x['rating'])
            )[:15]
        }), 200

    except Exception as e:
        logger.error(f"[MOOD API] Exception: {str(e)}")
        return jsonify({"error": "Recommendation service unavailable"}), 500

@app.route('/api/recommendations', methods=['POST'])
def handle_recommendations():
    try:
        if not request.is_json:
            return jsonify({"error": "Missing JSON in request"}), 400

        data = request.get_json()
        email = data.get('email')
        rec_type_raw = data.get('type') or data.get('recommendation_type') or 'personalized'

        if not isinstance(rec_type_raw, str):
            return jsonify({"error": "Invalid recommendation type"}), 400

        rec_type = rec_type_raw.lower()
        mood = data.get('mood')
        print("Incoming recommendation data:", data)

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "User not found"}), 404

        apply_weight_decay(user)

        recommendations = []
        rec_label = rec_type

        if rec_type == 'personalized':
            recommendations = get_personalized_recommendations(user)
            rec_label = "personalized"

        elif rec_type == 'mood' and mood:
            if mood not in MOOD_GENRES:
                return jsonify({"error": "Invalid mood"}), 400
            recommendations = get_mood_recommendations(user, mood)
            rec_label = f"mood-{mood}"

        elif rec_type == 'latest':
            current_year = datetime.now().year
            movies_df = load_movies_df()
            recent_movies_df = movies_df[
                (movies_df['year'] >= current_year - 2)
            ].sort_values(by='rating', ascending=False).head(40)

            recent_movies = []
            for movie in recent_movies_df.to_dict(orient='records'):
                poster = get_movie_poster(movie['title'], movie.get('year'))
                if poster:
                    movie['poster'] = poster
                    recent_movies.append(movie)
                if len(recent_movies) >= 20:
                    break

            recommendations = recent_movies
            rec_label = "latest"

        elif rec_type == 'hybrid':
            user_genres = set(user.get("genres", []))
            for mood_data in user.get("mood_genres", {}).values():
                user_genres.update(mood_data.get("learned", []))

            if not user_genres:
                return jsonify({"error": "No genres available for hybrid recommendation"}), 400

            movies_df["genres"] = movies_df["genres"].astype(str).str.strip().str.lower().str.split('|')
            movies_df["genres"] = movies_df["genres"].apply(lambda x: [g.strip() for g in x if g.strip()])
            all_movies = movies_df[['id', 'title', 'genres', 'rating', 'year']].dropna(subset=['genres']).to_dict(orient='records')

            genre_based_all = get_fuzzy_genre_matches(list(user_genres), all_movies)[:40]
            genre_based = []
            for movie in genre_based_all:
                poster = get_movie_poster(movie['title'], movie.get('year'))
                if poster:
                    movie['poster'] = poster
                    genre_based.append(movie)
                if len(genre_based) >= 10:
                    break

            print("🎯 Filtered genre-based with posters:", [m['title'] for m in genre_based])

            current_year = datetime.now().year
            recent_popular_df = movies_df[
                (movies_df['year'] >= current_year - 5)
            ].sort_values(['rating', 'year'], ascending=[False, False]).head(40)
            recent_popular_all = recent_popular_df[['id', 'title', 'genres', 'rating', 'year']].to_dict('records')

            recent_popular = []
            for movie in recent_popular_all:
                poster = get_movie_poster(movie['title'], movie.get('year'))
                if poster:
                    movie['poster'] = poster
                    recent_popular.append(movie)
                if len(recent_popular) >= 10:
                    break

            print("🎬 Filtered recent popular with posters:", [m['title'] for m in recent_popular])

            combined = genre_based + recent_popular
            unique_recommendations = []
            seen_ids = set()

            for movie in combined:
                if movie['id'] not in seen_ids:
                    unique_recommendations.append(movie)
                    seen_ids.add(movie['id'])

            recommendations = unique_recommendations
            rec_label = "hybrid"

        elif rec_type == 'favorites':
           liked_titles = set(unidecode(title).lower().replace(" ", "").strip() for title in user.get("liked_movies", []))

           def normalize_title(title):
             return unidecode(str(title)).lower().replace(" ", "").strip()

    # Match liked titles from the user with titles in the DataFrame
           matched_movies = movies_df[movies_df['title'].apply(lambda t: normalize_title(t) in liked_titles)]

           recommendations = matched_movies.to_dict(orient="records")
           rec_label = "favorites"



        else:
            return jsonify({"error": "Invalid recommendation type"}), 400

        results = []
        seen_ids = set()

        for movie in recommendations:
            movie_id = movie.get('id')
            if not movie_id or movie_id in seen_ids:
                continue

            genres = normalize_genres(movie.get('genres', []))
            details = {
                "id": movie_id,
                "title": f"{movie['title']} ({movie.get('year', 'N/A')})",
                "poster": movie.get('poster') or get_movie_poster(movie['title'], movie.get('year')),
                "genres": genres,
                "year": movie.get('year', 2023),
                "rating": round(movie.get('rating', 0), 1),
                "type": rec_label
            }

            if 'mood' in rec_label:
                mood_name = rec_label.split('-')[-1]
                details.update({
                    "base_match": len(set(genres) & set(MOOD_GENRES[mood_name]["base"])),
                    "learned_match": len(set(genres) & set(user['mood_genres'][mood_name].get("learned", [])))
                })

            results.append(details)
            seen_ids.add(movie_id)

            if len(results) >= 20:
                break

        if rec_type == 'hybrid':
            print("🚀 Final hybrid recommendations sent:", [m['title'] for m in results])

        return jsonify({
            "movies": results,
            "recommendation_type": rec_label,
            "message": f"{rec_label.replace('-', ' ')} recommendations"
        }), 200

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": "Recommendation service unavailable",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
