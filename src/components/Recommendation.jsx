import React, { useState, useEffect, useMemo, useRef } from "react";
import axios from "axios";
import confetti from 'canvas-confetti';
import { useNavigate } from "react-router-dom";
import "../styles/Recommendation.css";
import logo from "../assets/logo.jpg";

export const MOOD_GENRES = {
  "feel-good": {
    base: ["comedy", "romance", "musical", "children", "animation"],
    min_match: 0.6
  },
  "mind-bending": {
    base: ["sci-fi", "mystery", "thriller", "drama"],
    min_match: 0.6
  },
  "action-packed": {
    base: ["action", "adventure", "war", "thriller", "crime"],
    min_match: 0.6
  },
  "chill-relax": {
    base: ["documentary", "drama", "romance", "animation"],
    min_match: 0.6
  }
};

const Recommendation = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationType, setRecommendationType] = useState("personalized");
  const [recommendationMessage, setRecommendationMessage] = useState("");
  const [error, setError] = useState("");
  const [loadingStates, setLoadingStates] = useState({
    main: false,
    mood: false,
    search: false
  });
  const [searchQuery, setSearchQuery] = useState("");
  const [topMovies, setTopMovies] = useState([]);
  const [currentMovieIndex, setCurrentMovieIndex] = useState(0);
  const [selectedMood, setSelectedMood] = useState(null);
  const [showTypeDropdown, setShowTypeDropdown] = useState(false);
  const [showMoodDropdown, setShowMoodDropdown] = useState(false);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [showFavouritePrompt, setShowFavouritePrompt] = useState(false);
  const navigate = useNavigate();
  const [currentSource, setCurrentSource] = useState("personalized"); 
  const [movies, setMovies] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [userPreferences, setUserPreferences] = useState({
    liked: new Set(),
    disliked: new Set(),
    genres: []
  });
  const [moodData, setMoodData] = useState({
    learnedGenres: [],
    baseGenres: []
  });

  const filteredRecommendations = useMemo(
    () => recommendations.filter(m => 
      m?.id && !userPreferences.disliked.has(m.id)
    ),
    [recommendations, userPreferences.disliked]
  );

  useEffect(() => {
    const fetchPreferences = async () => {
      try {
        const email = localStorage.getItem("email");
        const response = await axios.get(
          `/api/user-preferences?email=${encodeURIComponent(email)}`
        );
        setUserPreferences({
          liked: new Set(response.data.liked),
          disliked: new Set(response.data.disliked),
          genres: response.data.genres
        });
      } catch (error) {
        console.error("Error fetching preferences:", error);
      }
    };
    fetchPreferences();
  }, []);

  useEffect(() => {
    const fetchTopMovies = async () => {
      try {
        const response = await axios.get('/api/recent-movies');
        setTopMovies(response.data.movies.slice(0, 5));
      } catch (error) {
        console.error('Error fetching top movies:', error);
      }
    };
    fetchTopMovies();
  }, []);

  useEffect(() => {
    let interval;
    if (topMovies.length > 0 && !selectedMovie) {
      interval = setInterval(() => {
        setCurrentMovieIndex((prev) => (prev + 1) % topMovies.length);
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [topMovies, selectedMovie]);

  const typeDropdownRef = useRef(null);
const moodDropdownRef = useRef(null);

useEffect(() => {
  const handleClickOutside = (event) => {
    if (
      typeDropdownRef.current && 
      !typeDropdownRef.current.contains(event.target)
    ) {
      setShowTypeDropdown(false);
    }
    if (
      moodDropdownRef.current && 
      !moodDropdownRef.current.contains(event.target)
    ) {
      setShowMoodDropdown(false);
    }
  };

  document.addEventListener("click", handleClickOutside);
  return () => {
    document.removeEventListener("click", handleClickOutside);
  };
}, []);


  const handleSearch = async (query) => {
    setSearchQuery(query);
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    setLoadingStates(prev => ({...prev, search: true}));
    try {
      const response = await axios.get('/api/search', {
        params: { query },
        headers: {'Content-Type': 'application/json'}
      });

      if (response.data?.movies) {
        setSearchResults(response.data.movies);
        setError('');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message);
      setSearchResults([]);
    } finally {
      setLoadingStates(prev => ({...prev, search: false}));
    }
  };

  const typeBadges = {
    'personalized': '⭐ Personalized',
    'hybrid': '✨ Hybrid',
    'latest': '🆕 New',
    'feel-good': '😊 Feel-Good',
    'mind-bending': '🤯 Mind-Bending',
    'action-packed': '🔥 Action',
    'chill-relax': '🍷 Chill'
  };

  const handleRateMovie = async (movieId, action) => {
    try {
      const email = localStorage.getItem("email");
      if (!email) return;
  
      setUserPreferences((prev) => {
        const updatedPrefs = {
          ...prev,
          liked: new Set(prev.liked),
          disliked: new Set(prev.disliked),
        };
  
        if (action === "like") {
          updatedPrefs.disliked.delete(movieId);
          updatedPrefs.liked.add(movieId);
  
  
          confetti({
            particleCount: 100,
            spread: 70,
            origin: { y: 0.6 }
          });
  
          setShowFavouritePrompt(true);
          setTimeout(() => setShowFavouritePrompt(false), 1000);
        } else if (action === "dislike") {
          updatedPrefs.liked.delete(movieId);
          updatedPrefs.disliked.add(movieId);
        }
  
        return updatedPrefs;
      });

      await axios.post("/api/rate-movie", {
        email,
        movie_id: movieId,
        action,
        source: currentSource || "",
      });
    } catch (error) {
      console.error("Rating failed:", error);
    }
  };
    
  const fetchRecommendations = async (source) => {
    const type = source || currentSource;
  
    if (type === "mood") return; 
  
    try {
      setLoadingStates(prev => ({ ...prev, main: true }));
      const email = localStorage.getItem("email");
  
      if (!email) {
        setError("Missing user email");
        return;
      }
  
      console.log(`[${type.toUpperCase()}] Requesting recommendations...`);
  
      const response = await axios.post("/api/recommendations", {
        email,
        type: type
      });
  
      const allMovies = response.data.movies || [];
      console.log(`[${type.toUpperCase()}] Raw response movies:`, allMovies.length);
  
      const validMovies = type === "personalized"
        ? allMovies.filter(movie => {
            const movieGenres = new Set(
              Array.isArray(movie.genres)
                ? movie.genres
                : movie.genres?.split('|') || []
            );
            const matches = [...movieGenres].filter(g => userPreferences.genres.includes(g));
            return matches.length / movieGenres.size >= 0.6;
          })
        : allMovies;
  
      console.log(`[${type.toUpperCase()}] Filtered/valid movies:`, validMovies.length);
  
      setRecommendations(validMovies);
      setRecommendationType(response.data.recommendation_type);
      setError("");
    } catch (err) {
      console.error("Recommendation fetch failed:", err.response?.data || err.message);
      setError("Recommendation service unavailable");
    } finally {
      setLoadingStates(prev => ({ ...prev, main: false }));
    }
  };
  
  
  
  const fetchMoodRecommendations = async (mood) => {
        try {
      setLoadingStates(prev => ({...prev, mood: true}));
      const email = localStorage.getItem("email"); 
      const response = await axios.get(
        `/api/mood-recommendations`,
        {
          params: { email, mood },
          headers: {
            'Accept': 'application/json'
          }
        }
      );      

      const processedMovies = response.data.movies.map(movie => ({
        ...movie,
        confidence: `${Math.round((movie.base_match + movie.learned_match) * 25)}%`,
        genres: Array.isArray(movie.genres) ? movie.genres : movie.genres.split('|')
      }));

      setRecommendations(processedMovies);
      setMoodData({
        learnedGenres: response.data.learned_genres || [],
        baseGenres: MOOD_GENRES[mood].base
      });
      setRecommendationType('mood');
      setSelectedMood(mood);
      setError("");
    } catch (error) {
      handleMoodError(error, mood);
    } finally {
      setLoadingStates(prev => ({...prev, mood: false}));
    }
  };

  const handleMoodError = (error, mood) => {
    let message = 'Failed to load recommendations';
    if (error.response?.data?.error?.includes('genre match')) {
      message = `No movies found matching ${mood} requirements`;
    } else if (error.message.includes('network')) {
      message = 'Network error - check your connection';
    }
    setError(message);
    
    if (!error.response || error.response.status >= 500) {
      fetchRecommendations('hybrid');
    }
  };
  const MovieCard = ({ movie }) => {
    const parseGenres = () => {
      try {
        if (!movie.genres) return [];
        return Array.isArray(movie.genres) 
          ? movie.genres 
          : movie.genres.split('|');
      } catch (e) {
        return ['Unknown Genre'];
      }
    };

    const safeGenres = parseGenres().slice(0, 4);

    return (
      <div 
        className={`movie-card ${userPreferences.disliked.has(movie.id) ? 'disliked' : ''}`}
        onClick={() => setSelectedMovie(movie)}
      >

        <div className={`type-badge ${recommendationType}`}>
          {typeBadges[recommendationType] || '🎬 Movie'}
        </div>

        <img 
          src={movie.poster || '/fallback-poster.jpg'}
          alt={movie.title}
          className="movie-poster"
          onError={(e) => e.target.src = '/fallback-poster.jpg'}
          loading="lazy"
        />

        <div className="movie-content">
          <h3 className="movie-title">
            {movie.title.replace(/\(\d{4}\)/, '').trim()}
          </h3>
          
          <div className="genre-container">
            {safeGenres.map((genre, index) => {
              const isBase = moodData.baseGenres.includes(genre);
              const isLearned = moodData.learnedGenres.includes(genre);
              
              return (
                <span
  key={`${movie.id}-${genre}-${index}`}
  className="genre-tag neutral-genre"
>
  {genre}
</span>

              );
            })}
          </div>

          <div className="rating-buttons">
            <button
  className={`like-btn ${userPreferences.liked.has(movie.id) ? 'active' : ''}`}
  onClick={(e) => {
    e.stopPropagation();
    handleRateMovie(movie.id, "like");
  }}
  disabled={userPreferences.disliked.has(movie.id)}
>
  {userPreferences.liked.has(movie.id) ? '✓ Liked' : '👍 Like'}
</button>

<button
  className={`dislike-btn ${userPreferences.disliked.has(movie.id) ? 'active' : ''}`}
  onClick={(e) => {
    e.stopPropagation();
    handleRateMovie(movie.id, "dislike");
  }}
  disabled={userPreferences.liked.has(movie.id)}
>
  {userPreferences.disliked.has(movie.id) ? '✗ Disliked' : '👎 Dislike'}
</button>

          </div>

          <div className="movie-metadata">
            <p className="movie-rating">
              ⭐{Number(movie.rating).toFixed(1)}/5
            </p>
            {movie.year && (
              <p className="movie-year">
                {new Date(movie.year, 0).getFullYear()}
              </p>
            )}
          </div>
        </div>
      </div>
    );
  };

  const getRecommendationSubtitle = () => {
    const subtitles = {
      personalized: 'Based on your preferences and viewing history',
      hybrid: 'Mix of personalized picks and popular choices',
      latest: 'Recent releases you might enjoy',
      'feel-good': 'Uplifting themes and cheerful stories to boost your mood',
      'mind-bending': 'Thought-provoking narratives that challenge perception',
      'action-packed': 'High-octane adventures and thrilling sequences',
      'chill-relax': 'Soothing content for relaxation and unwinding'
    };
    return subtitles[recommendationType] || 'Curated recommendations just for you';
  };

  return (
    <div className="recommendation-container">
      <div className="background-overlay"></div>
      
      <nav className="navbar">
        <div className="nav-left">
          <img src={logo} className="nav-logo" alt="CinePodium Logo" />
          
          <div className="dropdown" ref={typeDropdownRef}>
            <button 
              className="dropdown-toggle"
              onClick={(e) => {
                e.stopPropagation();
                setShowTypeDropdown(!showTypeDropdown);
                setShowMoodDropdown(false);
              }}
            >
              🎯 Recommendation Type ▼
            </button>
            {showTypeDropdown && (
              <div className="dropdown-menu">
                <button onClick={() => {
                  fetchRecommendations('personalized')
                  setCurrentSource("personalized")
                }}>
                  Personal Preferences
                </button>
                <button onClick={() => {
                  fetchRecommendations('hybrid')
                  setCurrentSource("hybrid")
                }}>
                  Hybrid Suggestions
                </button>
                <button onClick={() => {
                  fetchRecommendations('latest')
                  setCurrentSource("latest")
                }}>
                  New Releases
                </button>
              </div>
            )}
          </div>
  
          <div className="dropdown" ref={moodDropdownRef}>
            <button 
              className="dropdown-toggle"
              onClick={(e) => {
                e.stopPropagation();
                setShowMoodDropdown(!showMoodDropdown);
                setShowTypeDropdown(false);
              }}
            >
              🎭 Mood-Based ▼
            </button>
            {showMoodDropdown && (
              <div className="dropdown-menu mood-menu">
                <button onClick={() =>{
                  fetchMoodRecommendations('feel-good')
                  setCurrentSource("feel-good")
                }}>
                  😊 Feel-Good
                </button>
                <button onClick={() => {
                  fetchMoodRecommendations('mind-bending')
                  setCurrentSource("mind-bending")
                }}>
                  🤯 Mind-Bending
                </button>
                <button onClick={() => {
                  fetchMoodRecommendations('action-packed')
                  setCurrentSource("action-packed")
                }}>
                  🔥 Action-Packed
                </button>
                <button onClick={() => {
                  fetchMoodRecommendations('chill-relax')
                  setCurrentSource("chill-relax")
                }}>
                  🍷 Chill & Relax
                </button>
              </div>
            )}
          </div>
        </div>
        
        <button className="favorite-button" onClick={() => fetchRecommendations("favorites")}>
  💖 Favorites
</button>

        <div className="nav-right">
          <input
            type="text"
            placeholder="Search movies..."
            className="search-bar"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
          />
<button 
  className="logout-btn" 
  onClick={() => {
    localStorage.removeItem("email");
    navigate("/", { replace: true });
    window.location.reload();
  }}
>
  Logout
</button>
        </div>
      </nav>

      {selectedMovie ? (
        <div className="selected-movie-container">
          <button 
            className="close-button"
            onClick={() => setSelectedMovie(null)}
          >
            &times;
          </button>
          {selectedMovie.mood && (
            <div className="mood-badge selected-mood-badge">
              {selectedMovie.mood.replace(/-/g, ' ').toUpperCase()}
              <span className="confidence-badge large">
                {selectedMovie.confidence || '60%+ Match'}
              </span>
            </div>
          )}
          <img
            src={selectedMovie.poster || '/fallback-poster.jpg'}
            alt={selectedMovie.title}
            className="selected-movie-poster"
            onError={(e) => e.target.src = '/fallback-poster.jpg'}
          />
          <div className="selected-movie-details">
            <h2 className="selected-movie-title">
              {selectedMovie.title.replace(/\(\d{4}\)/, '').trim()}
            </h2>
            <div className="selected-movie-info">
              <span className="selected-movie-year">
                {selectedMovie.year || 'Year not available'}
              </span>
              <span className="selected-movie-rating">
                ⭐{Number(selectedMovie.rating || 0).toFixed(1)}/5
              </span>
            </div>
            <div className="genre-container">
              {selectedMovie.genres?.split('|').map(genre => {
                const isBase = moodData.baseGenres.includes(genre);
                const isLearned = moodData.learnedGenres.includes(genre);
                return (
                  <span
  key={genre}
  className="genre-tag neutral-genre"
>
  {genre}
</span>

                );
              })}
            </div>
            <div className="selected-movie-actions">
            <button
  className={`like-btn ${userPreferences.liked.has(selectedMovie.id) ? 'active' : ''}`}
  onClick={() => handleRateMovie(selectedMovie.id, "like")}
  disabled={userPreferences.disliked.has(selectedMovie.id)}
  title="Like this movie"
>
  {userPreferences.liked.has(selectedMovie.id) ? '✓ Liked' : '👍 Like'}
</button>

<button
  className={`dislike-btn ${userPreferences.disliked.has(selectedMovie.id) ? 'active' : ''}`}
  onClick={() => handleRateMovie(selectedMovie.id, "dislike")}
  disabled={userPreferences.liked.has(selectedMovie.id)}
  title="Dislike this movie"
>
  {userPreferences.disliked.has(selectedMovie.id) ? '✗ Disliked' : '👎 Dislike'}
</button>

</div>

          </div>
        </div>
      ) : (
        <div className="carousel-container">
          {topMovies.length > 0 && (
            <>
              <div className="carousel-slide">
                <img 
                  src={topMovies[currentMovieIndex]?.poster || '/fallback-poster.jpg'}
                  alt={topMovies[currentMovieIndex]?.title}
                  className="carousel-image"
                />
                <div className="carousel-overlay">
                  <h3 className="carousel-title">
                    {topMovies[currentMovieIndex]?.title}
                  </h3>
                </div>
              </div>
              <div className="carousel-indicators">
                {topMovies.map((_, index) => (
                  <div 
                    key={index}
                    className={`indicator ${index === currentMovieIndex ? 'active' : ''}`}
                    onClick={() => setCurrentMovieIndex(index)}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      )}

      <h1 className="recommendation-header">
        {searchQuery ? 
          `Search Results for "${searchQuery}"` : 
          recommendationMessage || 'Your Recommendations'
        }
      </h1>
      
      {!searchQuery && (
        <p className="recommendation-subtitle">
          {getRecommendationSubtitle()}
          {recommendationType === 'mood' && (
            <span className="match-disclaimer">
              (Minimum 60% genre match)
            </span>
          )}
        </p>
      )}

      {error && <p className="error-message">{error}</p>}
      
      {showFavouritePrompt && (
  <div className="favourite-popup-center">
    🎉 Added to Favourites!
  </div>
)}

      <div className="movie-grid">
        {(loadingStates.main || loadingStates.mood || loadingStates.search) ? (
          <div className="loading-message">
            <div className="spinner"></div>
            {loadingStates.main && "Loading recommendations..."}
            {loadingStates.mood && "Curating mood picks..."}
            {loadingStates.search && "Searching movies..."}
          </div>
        ) : searchQuery ? (
          searchResults.length > 0 ? (
            searchResults.map((movie) => (
              <div className="movie-card-wrapper" key={movie.id}>
                <MovieCard movie={movie} />
              </div>
            ))
          ) : (
            <div className="empty-message">
              {`No movies found for "${searchQuery}"`}
            </div>
          )
        ) : filteredRecommendations.length > 0 ? (
          filteredRecommendations.map((movie) => (
            <div className="movie-card-wrapper" key={movie.id}>
              <MovieCard movie={movie} />
            </div>
          ))
        ) : (
          <div className="empty-message">
            <p>No recommendations found. Try selecting different genres!</p>
            <button 
              className="fallback-button"
              onClick={() => fetchRecommendations('hybrid')}
            >
              Try Hybrid Recommendations
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Recommendation;