# Food Quality Inspection Model

AI-Powered Food Quality Inspection System with React Frontend

## 🎯 Features

- **Automated Model Selection**: Automatically selects the model with the highest validation accuracy
- **Real-time Prediction**: Upload fruit images and get instant quality predictions (Fresh/Spoiled)
- **Beautiful UI**: Modern, responsive React frontend with gradient design
- **Confidence Metrics**: Detailed probability breakdown and confidence scores

## 📁 Project Structure

```
├── Food_Quality_Inspection_Model.h5      # Model file 1
├── best_hybrid_hsi_model.h5              # Model file 2
├── evaluate_models.py                    # Script to determine best model
├── app.py                                # Flask backend API
├── requirements.txt                      # Python dependencies
├── frontend/                             # React frontend
│   ├── src/
│   │   ├── App.js                       # Main React component
│   │   ├── App.css                      # Styling
│   │   └── index.js                     # Entry point
│   ├── package.json                     # Node dependencies
│   └── public/
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Evaluate models (optional - will run automatically on first start):**
   ```bash
   python evaluate_models.py
   ```
   This script will:
   - Load both model files
   - Evaluate them on the validation set
   - Determine which has the highest validation accuracy
   - Save the best model name to `best_model.txt`

3. **Start the Flask backend:**
   ```bash
   python app.py
   ```
   The API will run on `http://localhost:5000`

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```
   The frontend will run on `http://localhost:3000`

## 📖 API Endpoints

- `GET /api/health` - Health check endpoint
- `GET /api/model-info` - Get information about the loaded model
- `POST /api/predict` - Predict food quality from uploaded image

### Example Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@path/to/your/image.jpg"
```

### Example Response

```json
{
  "success": true,
  "prediction": "Fresh",
  "confidence": 95.5,
  "probability": {
    "Fresh": 95.5,
    "Spoiled": 4.5
  }
}
```

## 🎨 Usage

1. Open the React app in your browser (`http://localhost:3000`)
2. Click the upload area or drag and drop an image
3. Click "🔍 Analyze Quality" button
4. View the prediction results with confidence scores

## 📊 Model Information

The system automatically loads the model with the highest validation accuracy:
- `Food_Quality_Inspection_Model.h5`: Final model after training
- `best_hybrid_hsi_model.h5`: Best model saved during training (based on validation AUC)

## 🔧 Technology Stack

### Backend
- Flask - Web framework
- TensorFlow/Keras - Deep learning model
- Flask-CORS - Cross-origin resource sharing

### Frontend
- React - UI framework
- Axios - HTTP client
- CSS3 - Styling with gradients and animations

## 📝 Dataset

- Dataset: https://drive.google.com/drive/folders/1Yogurdf2ZwvksChvmREILEhepfqt0zFj?usp=sharing

## 🎯 Model Architecture

The model uses a hybrid approach combining:
- RGB image features (MobileNetV2 backbone)
- Pseudo-spectral features
- Optional HSI (Hyperspectral Imaging) features

## 📄 License

This project is part of a Computer Vision course project.
