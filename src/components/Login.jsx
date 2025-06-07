import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Login.css";
import logo from "../assets/logo.jpg";
import { MOOD_GENRES } from './Recommendation';

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!email || !password) {
      alert("Please fill in all fields.");
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store user session information
        localStorage.setItem("email", email);
        if (data.token) {
          localStorage.setItem("token", data.token);
        }

        // Redirect based on genre selection status
        if (data.user.has_genres) {
          navigate("/recommendations");
      } else {
          navigate("/genre");
      }
      } else {
        alert(data.message || "Login failed.");
      }
    } catch (error) {
      console.error("Error logging in:", error);
      alert("An error occurred. Please try again later.");
    }
  };

  return (
    <div className="login-container">
      <img src={logo} alt="Cineopia Logo" className="logo" />
      <button 
        className="signup-button signup-btn" 
        onClick={() => navigate("/signup")}
      >
        Sign up
      </button>
      <div className="login-box">
        <h2>Log in to continue watching</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="email"
            placeholder="Email Address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="current-password"
          />
          <button type="submit" className="signin-btn">
            Log In
          </button>
        </form>
        <p className="forgot-password">Forgot Password?</p>
      </div>
    </div>
  );
};

export default Login;