import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    // Fetch model information on mount
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model-info`);
      setModelInfo(response.data);
    } catch (err) {
      console.error('Error fetching model info:', err);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to predict. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1 className="title">🥗 Food Quality Inspection</h1>
          <p className="subtitle">AI-Powered Fruit Quality Detection System</p>
          {modelInfo && (
            <div className="model-info">
              <span className="model-badge">Model: {modelInfo.model_name}</span>
            </div>
          )}
        </header>

        <main className="main-content">
          <div className="upload-section">
            <div className="upload-area">
              {preview ? (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview-image" />
                  <button className="change-button" onClick={handleReset}>
                    Change Image
                  </button>
                </div>
              ) : (
                <label htmlFor="file-upload" className="upload-label">
                  <div className="upload-icon">📷</div>
                  <p className="upload-text">Click to upload an image</p>
                  <p className="upload-hint">or drag and drop</p>
                  <p className="upload-formats">PNG, JPG, JPEG up to 10MB</p>
                  <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="file-input"
                  />
                </label>
              )}
            </div>

            <div className="action-buttons">
              <button
                onClick={handlePredict}
                disabled={!selectedFile || loading}
                className="predict-button"
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    🔍 Analyze Quality
                  </>
                )}
              </button>
            </div>
          </div>

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {prediction && (
            <div className="prediction-result">
              <div className={`result-card ${prediction.prediction.toLowerCase()}`}>
                <div className="result-header">
                  <h2>Prediction Result</h2>
                  <span className={`status-badge ${prediction.prediction.toLowerCase()}`}>
                    {prediction.prediction === 'Fresh' ? '✓ Fresh' : '✗ Spoiled'}
                  </span>
                </div>
                
                <div className="confidence-meter">
                  <div className="confidence-label">Confidence</div>
                  <div className="confidence-bar-container">
                    <div
                      className={`confidence-bar ${prediction.prediction.toLowerCase()}`}
                      style={{ width: `${prediction.confidence}%` }}
                    >
                      <span className="confidence-value">{prediction.confidence}%</span>
                    </div>
                  </div>
                </div>

                <div className="probability-breakdown">
                  <div className="probability-item">
                    <span className="probability-label">Fresh</span>
                    <span className="probability-value">{prediction.probability.Fresh}%</span>
                  </div>
                  <div className="probability-item">
                    <span className="probability-label">Spoiled</span>
                    <span className="probability-value">{prediction.probability.Spoiled}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>

        <footer className="footer">
          <p>Powered by Deep Learning | Hybrid RGB-Pseudo Spectral Model</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
