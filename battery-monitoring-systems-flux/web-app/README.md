# Battery Monitoring System - Web Application

This directory contains the web application for the Battery Monitoring System, including both frontend and backend components.

## 📁 Directory Structure

```
web-app/
├── backend/           # FastAPI backend server
│   ├── api.py        # REST API endpoints
│   ├── websocket.py  # WebSocket server for real-time updates
│   └── main.py       # Backend server entry point
├── frontend/         # Static frontend files
│   ├── index.html    # Main HTML page
│   ├── styles.css    # CSS styles
│   ├── app.js        # JavaScript functionality
│   └── package.json  # Frontend dependencies
└── README.md         # This file
```

## 🚀 Quick Start

### Option 1: Start from the main project (Recommended)

From the project root directory:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the complete system (includes web app)
python main.py start
```

### Option 2: Start web application separately

```bash
# Navigate to web-app directory
cd web-app

# Start backend server
cd backend
python main.py
```

## 🌐 Access Points

Once started, you can access:

- **Frontend Dashboard**: http://localhost:8000/static
- **API Documentation**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8001

## 📊 Features

### Frontend Dashboard
- **Real-time Monitoring**: Live battery data visualization
- **Interactive Charts**: Voltage, temperature, and specific gravity trends
- **Data Filtering**: Filter by device ID and cell number
- **System Statistics**: Quick overview of system health
- **AI Chatbot**: Interactive AI assistant for data queries
- **Responsive Design**: Works on desktop and mobile devices

### Backend API
- **RESTful Endpoints**: Complete CRUD operations for battery data
- **Real-time Updates**: WebSocket support for live data streaming
- **AI Integration**: LLM-powered chatbot for data analysis
- **Data Analytics**: Statistical endpoints for system insights
- **Authentication**: Secure API access (production-ready)

## 🔧 Configuration

The web application uses the main project configuration:

- **API Port**: 8000 (configurable in `config_local.yaml`)
- **WebSocket Port**: 8001 (configurable in `config_local.yaml`)
- **Host**: localhost (configurable for production)

## 🛠️ Development

### Frontend Development

The frontend is built with vanilla HTML, CSS, and JavaScript:

```bash
cd web-app/frontend

# Start a local development server
python -m http.server 3000

# Or use npm (if you have Node.js installed)
npm start
```

### Backend Development

```bash
cd web-app/backend

# Run with auto-reload
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### Core Endpoints
- `GET /api/status` - System status and statistics
- `GET /api/battery-data` - Battery data with filtering
- `GET /api/devices` - List of all devices
- `GET /api/cells` - List of all cells
- `POST /api/chat` - AI chatbot interface

### WebSocket Events
- `battery_update` - Real-time battery data updates
- `anomaly_detected` - Anomaly detection alerts
- `cell_status_change` - Cell status changes
- `system_alert` - System-wide alerts

## 🎨 Frontend Features

### Real-time Dashboard
- Live voltage, temperature, and specific gravity monitoring
- Interactive line charts with Chart.js
- Trend indicators (up/down arrows)
- Connection status indicator

### Data Management
- Filterable data table
- Device and cell selection dropdowns
- Real-time data refresh
- Export capabilities

### AI Assistant
- Floating chat button
- Modal-based chat interface
- Context-aware responses
- Data analysis queries

### Responsive Design
- Bootstrap 5 framework
- Mobile-friendly layout
- Custom CSS animations
- Professional UI/UX

## 🔒 Security Features

- Input validation and sanitization
- CORS configuration
- Rate limiting (production-ready)
- Authentication endpoints (ready for implementation)
- Secure WebSocket connections

## 🚀 Production Deployment

### Docker Support (Commented in main project)
```dockerfile
# Production Dockerfile available in main project
# Includes nginx, gunicorn, and security configurations
```

### Cloud Deployment
- **AWS**: ECS/Fargate with ALB
- **Azure**: App Service with Application Gateway
- **GCP**: Cloud Run with Load Balancer
- **Kubernetes**: Helm charts available

### Environment Variables
```bash
# Production environment variables
export DATABASE_URL="postgresql://..."
export API_KEY="your-secure-api-key"
export WEB_HOST="0.0.0.0"
export WEB_PORT="8000"
export WEBSOCKET_PORT="8001"
```

## 🧪 Testing

### Frontend Testing
```bash
cd web-app/frontend
# Manual testing via browser
# Automated testing with Jest (future enhancement)
```

### Backend Testing
```bash
cd web-app/backend
# Run with test data
python -m pytest tests/
```

## 📈 Monitoring & Logging

- **Application Logs**: Structured logging with timestamps
- **Performance Metrics**: Response times and throughput
- **Error Tracking**: Comprehensive error handling
- **Health Checks**: `/health` endpoint for monitoring

## 🔄 Updates & Maintenance

### Frontend Updates
- Static files served by FastAPI
- Cache busting via version parameters
- CDN support for production

### Backend Updates
- Hot reload during development
- Graceful shutdown handling
- Database migration support

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Project Documentation**: See main project README files
- **Configuration**: See `config_local.yaml` and `config_prod.yaml`
- **Logs**: Check `logs/` directory for detailed logs

## 🤝 Contributing

1. Follow the main project coding standards
2. Test frontend changes in multiple browsers
3. Ensure responsive design works on mobile
4. Update API documentation for new endpoints
5. Add appropriate error handling

## 📄 License

This web application is part of the Battery Monitoring System project and follows the same license terms. 