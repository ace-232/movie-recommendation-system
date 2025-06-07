import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "../styles/Genre.css";
import logo from "../assets/logo.jpg";
import { MOOD_GENRES } from './Recommendation';

const genresList = [
  "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
  "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery",
  "Romance", "Sci-Fi", "Thriller", "War", "Western",
];

function Genre() {
  const [selectedGenres, setSelectedGenres] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const email = localStorage.getItem("email");
    if (!email) {
      alert("Please log in to select genres.");
      navigate("/login");
    }
  }, [navigate]);

  const handleGenreSelect = (genre) => {
    setSelectedGenres((prev) =>
      prev.includes(genre)
        ? prev.filter((g) => g !== genre)
        : prev.length < 5
        ? [...prev, genre]
        : prev
    );
  };

  const handleSaveGenres = async () => {
    try {
      const email = localStorage.getItem("email"); // Get the email from localStorage
      if (!email) {
        alert("Please log in.");
        navigate("/login");
        return;
      }

      const response = await axios.post("http://localhost:5000/api/save-genres", {
        email: email,
        genres: selectedGenres,
      });

      if (response.status === 200) {
        alert("Genres saved successfully!");
        navigate("/recommendations");
      } else {
        alert("Failed to save genres.");
      }
    } catch (error) {
      console.error("Failed to save genres:", error);
      alert("Error saving genres. Please try again.");
    }
  };

  return (
    <div className="genre-selection-container">
      <img src={logo} alt="Cineopia Logo" className="logo" />
      <h2 className="title">Select Your Favorite Genres</h2>
      <p className="subtitle">(Choose up to 5 genres)</p>

      <div className="genre-grid">
        {genresList.map((genre) => (
          <button
            key={genre}
            className={`genre-button ${selectedGenres.includes(genre) ? "selected" : ""}`}
            onClick={() => handleGenreSelect(genre)}
          >
            {genre}
          </button>
        ))}
      </div>

      <button
        className="proceed-button"
        onClick={handleSaveGenres}
        disabled={selectedGenres.length === 0}
      >
        Proceed
      </button>
    </div>
  );
};


export default Genre;
