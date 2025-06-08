import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./components/Login";
import SignUp from "./components/Signup";
import Genre from "./components/Genre";
import Recommendation from "./components/Recommendation";
import ErrorBoundary from "./components/ErrorBoundary";

const App = () => {
  const [recommendations, setRecommendations] = useState([]);
  
  return (
    <Router>
      {/* Wrap entire app with ErrorBoundary */}
      <ErrorBoundary>
        <Routes>
          {/* Login Page */}
          <Route path="/" element={<Navigate to="/login" />} />
          <Route path="/login" element={<Login />} />
          {/* Sign-Up Page */}
          <Route path="/signup" element={<SignUp />} />

          {/* Genre Selection Page */}
          <Route
            path="/genre"
            element={
              <ErrorBoundary>
                <Genre setRecommendations={setRecommendations} />
              </ErrorBoundary>
            }
          />

          {/* Recommendation Page */}
          <Route
            path="/recommendations"
            element={
              <ErrorBoundary>
                <Recommendation recommendations={recommendations} />
              </ErrorBoundary>
            }
          />
        </Routes>
      </ErrorBoundary>
    </Router>
  );
};

export default App;
