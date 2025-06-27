# ML REST API

A machine learning REST API built with FastAPI that serves Random Forest predictions with a Streamlit frontend.

## Features

- 🤖 **Random Forest Model** for regression predictions
- 🚀 **FastAPI Backend** with automatic documentation
- 🎨 **Streamlit Frontend** for CSV upload and predictions
- 🐳 **Docker Support** for easy deployment
- 📊 **Interactive API Documentation** (Swagger UI)

## Project Structure

```
ML_REST_API/
├── rf_api.py              # FastAPI application
├── model.py               # Model training script
├── synth_data.py          # Synthetic data generation
├── streamlit_api_app.py   # Streamlit frontend
├── request.py             # API testing script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── README.md             # Project documentation
```

## Quick Start


1. **Clone the repository:**
   ```bash
   git clone https://github.com/marcohernandezusal/rf_docker_api
   cd ML_REST_API
   ```
### Option 1: Local Development

2. **Create virtual environment:**
   ```bash
   python -m venv apienv
   apienv\Scripts\activate  # Windows
   # source apienv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model (if needed):**
   ```bash
   python model.py
   ```

5. **Run the API:**
   ```bash
   uvicorn rf_api:app --reload --port 8000
   ```

6. **Run Streamlit frontend:**
   ```bash
   streamlit run streamlit_api_app.py
   ```

### Option 2: Docker

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the services:**
   - API Documentation: http://localhost:8000/docs
   - Streamlit App: http://localhost:8501

## API Usage

### Predict Endpoint

**POST** `/predict`

```json
{
  "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}
```

**Response:**
```json
[42.5]
```

### Example with curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}'
```

## Development

### Model Training

The model uses Bayesian optimization to find the best hyperparameters:

```bash
python model.py
```

This will:
- Generate synthetic data (if not exists)
- Perform Bayesian hyperparameter optimization
- Train the Random Forest model
- Save the model as `random_forest_model.pkl`

### Testing

```bash
# Test the API
python request.py

# Or use the interactive docs
# Visit http://localhost:8000/docs
```

## Requirements

- Python 3.10+
- FastAPI
- Streamlit
- scikit-learn
- Docker (optional)

## License

This project is licensed under the MIT License.
