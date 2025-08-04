# Battery Monitoring System with ML/LLM & MLOps

A comprehensive battery monitoring system that provides advanced analytics, machine learning capabilities, and real-time monitoring for battery management systems (BMS).

## 🚀 Features

### Core ML Features
- **Anomaly Detection**: Detect anomalies in cell voltage, temperature, and specific gravity
- **Cell Health Prediction**: Predict if cells are likely to be dead or alive with confidence scores
- **Future Value Forecasting**: Forecast battery parameters for the next 25 time steps

### MLOps & Monitoring
- **Continuous Delivery for ML (CD4ML)**: Full ML lifecycle management
- **Operational Monitoring**: Real-time model performance tracking
- **Drift Detection**: Monitor data and model drift
- **Proxy Metrics**: Track confidence scores, prediction distributions, and output ranges
- **Ground Truth Feedback**: Compare predictions with actual outcomes
- **Alerting System**: Automated alerts for model degradation

### LLM Integration
- **AI Chatbot**: Powered by Ollama (Llama2 7B) for data analysis
- **Natural Language Queries**: Ask questions about battery data in plain English
- **Intelligent Insights**: Get automated analysis and recommendations

### Web Application
- **Real-time Dashboard**: Live monitoring with WebSocket support
- **Interactive Visualizations**: 3D plots for voltage, temperature, and specific gravity
- **MLOps Dashboard**: Comprehensive monitoring and management interface
- **LLMOps Dashboard**: LLM performance evaluation and management

## 📋 Prerequisites

- Python 3.9+
- Ollama (for LLM functionality)
- Git

## 🛠️ Local Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd battery-monitoring-systems-flux
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Install Ollama (for LLM features)
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Download and install from https://ollama.ai/download for other platforms
```

### 5. Pull LLM Model
```bash
ollama pull llama2:7b
```

### 6. Setup Configuration
The system uses `config_local.yaml` by default. You can modify it to suit your needs:

```yaml
# Key configuration sections:
app:
  debug: true
  port: 8000
  websocket_port: 8001

database:
  type: "sqlite"  # For local development
  url: "sqlite:///./battery_monitoring.db"

llm:
  provider: "ollama"
  model: "llama2:7b"
  base_url: "http://localhost:11434"
```

## 🚀 Quick Start

### 1. Setup the System
```bash
# Setup with sample data
python main.py setup

# Setup with synthetic data (recommended for testing)
python main.py setup --generate-data
```

### 2. Start the Application
```bash
python main.py start
```

### 3. Access the Application
- **Web Application**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8001

## 📊 Data Structure

The system expects battery monitoring data with the following key columns:

- `CellNumber`: Battery cell identifier
- `CellVoltage`: Cell voltage in volts
- `CellTemperature`: Cell temperature in Celsius
- `CellSpecificGravity`: Cell specific gravity
- `DeviceID`: BMS device identifier
- `SerialNumber`: Device serial number
- `PacketDateTime`: Timestamp of the data packet

## 🔧 Usage

### Command Line Interface

```bash
# Setup system with synthetic data
python main.py setup --generate-data

# Start the application
python main.py start

# Analyze specific device/cell
python main.py analyze --device-id 1 --cell-number 1

# Check system status
python main.py status

# Clean up old data
python main.py cleanup --days 30
```

### API Usage

The system provides a comprehensive REST API:

```python
import requests

# Get battery data
response = requests.get("http://localhost:8000/api/v1/battery-data")
data = response.json()

# Get anomaly detection results
response = requests.get("http://localhost:8000/api/v1/anomalies")
anomalies = response.json()

# Get cell predictions
response = requests.get("http://localhost:8000/api/v1/predictions")
predictions = response.json()

# Get forecasts
response = requests.get("http://localhost:8000/api/v1/forecasts")
forecasts = response.json()
```

### WebSocket for Real-time Data

```javascript
const ws = new WebSocket('ws://localhost:8001');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time data:', data);
};

ws.onopen = function() {
    console.log('WebSocket connected');
};
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=battery_monitoring
```

### Test Data
The system includes synthetic data generation for testing:

```python
from battery_monitoring.data.loader import DataLoader

loader = DataLoader()
synthetic_data = loader.generate_synthetic_data(
    num_devices=5,
    num_cells_per_device=10,
    num_samples=1000
)
```

## 📈 ML Models

### Anomaly Detection
- Uses Isolation Forest algorithm
- Detects anomalies in voltage, temperature, and specific gravity
- Configurable contamination factor

### Cell Health Prediction
- Binary classification (alive/dead)
- Confidence scores for predictions
- Feature engineering from historical patterns

### Forecasting
- Time series forecasting for 25 steps ahead
- Multiple algorithms (ARIMA, Prophet, LSTM)
- Confidence intervals for predictions

## 🔍 Monitoring & MLOps

### Model Monitoring
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Data Drift**: Statistical tests for data distribution changes
- **Model Drift**: Performance degradation detection
- **Proxy Metrics**: Confidence score distributions, prediction rates

### Alerting
- Email notifications for model degradation
- Slack integration for team alerts
- Webhook support for custom integrations

### Model Registry
- MLflow integration for model versioning
- Model metadata tracking
- Performance history

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama service
   ollama serve
   ```

2. **Database Connection Error**
   ```bash
   # Check database file
   ls -la battery_monitoring.db
   
   # Reset database (will lose data)
   rm battery_monitoring.db
   python main.py setup
   ```

3. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   
   # Kill the process or change port in config
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size in config
   performance:
     batch_size: 500  # Reduce from 1000
   ```

### Logs
Check logs for detailed error information:

```bash
# Master log
tail -f logs/project_master.log

# Module-specific logs
tail -f logs/ml_training.log
tail -f logs/anomaly_detection.log
tail -f logs/web_app.log
```

## 🔧 Development

### Project Structure
```
battery-monitoring-systems-flux/
├── battery_monitoring/
│   ├── core/           # Core functionality (config, logging, database)
│   ├── data/           # Data processing and loading
│   ├── ml/             # Machine learning models
│   ├── llm/            # LLM integration
│   ├── web_app/        # Web application (FastAPI + WebSocket)
│   ├── mlops/          # MLOps and monitoring
│   └── utils/          # Utility functions
├── web-app/
│   ├── frontend/       # Next.js frontend
│   └── backend/        # Backend services
├── tests/              # Test files
├── logs/               # Log files
├── models/             # Trained models
├── data/               # Data files
├── config_local.yaml   # Local configuration
├── config_prod.yaml    # Production configuration
├── pyproject.toml      # Project dependencies
└── main.py            # Main entry point
```

### Adding New Features

1. **New ML Model**
   ```python
   # Add to battery_monitoring/ml/
   class NewModel:
       def __init__(self, config):
           self.config = config
       
       def train(self, data):
           # Training logic
           pass
       
       def predict(self, data):
           # Prediction logic
           pass
   ```

2. **New API Endpoint**
   ```python
   # Add to battery_monitoring/web_app/api/
   @router.get("/new-endpoint")
   async def new_endpoint():
       return {"message": "New endpoint"}
   ```

3. **New WebSocket Event**
   ```python
   # Add to battery_monitoring/web_app/websocket/
   async def handle_new_event(websocket, data):
       # Handle new event
       pass
   ```

### Code Style
- Follow PEP 8 for Python code
- Use type hints for all functions
- Add docstrings for all classes and methods
- Use logging for debugging and monitoring

## 📝 Configuration

### Environment Variables
```bash
export BATTERY_CONFIG_PATH=config_local.yaml
export OLLAMA_HOST=http://localhost:11434
export DEBUG=true
```

### Configuration Files
- `config_local.yaml`: Local development settings
- `config_prod.yaml`: Production settings (commented cloud integrations)

## 🔒 Security

### Local Development
- Uses SQLite database (no authentication required)
- Debug mode enabled for development
- CORS configured for localhost

### Production Considerations
- Use PostgreSQL with proper authentication
- Enable HTTPS
- Configure proper CORS origins
- Use environment variables for secrets
- Enable rate limiting

## 📊 Performance

### Optimization Tips
1. **Data Processing**: Use batch processing for large datasets
2. **Model Inference**: Cache model predictions
3. **Database**: Use connection pooling
4. **WebSocket**: Implement proper connection management

### Monitoring
- CPU and memory usage tracking
- Database query performance
- Model inference latency
- WebSocket connection count

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the logs for error details
- Review the API documentation at http://localhost:8000/docs
- Open an issue on GitHub

## 🔄 Changelog

### Version 1.0.0
- Initial release
- Anomaly detection for battery parameters
- Cell health prediction with confidence scores
- Time series forecasting
- MLOps with monitoring and alerting
- LLM-powered chatbot
- Real-time web application with WebSocket
- Comprehensive API and documentation 