#!/bin/bash
# Battery Monitor Project Bootstrap Script (Backend + Frontend)
# This script creates the full directory structure and populates key files.
# Run: bash setup_battery_monitor.sh

set -e

# --- Directory structure ---
mkdir -p battery-monitor/backend/src/{models,controllers,middleware,routes,services,utils}
mkdir -p battery-monitor/frontend/src/{components,context,pages,services}
mkdir -p battery-monitor/frontend/public
mkdir -p battery-monitor/.devcontainer

# --- README.md ---
cat > battery-monitor/README.md <<'EOF'
# Battery Monitor Platform

A full-stack real-time battery telemetry, alert, and device management platform.

See `docker-compose.yml` and README for setup instructions.
EOF

# --- docker-compose.yml ---
cat > battery-monitor/docker-compose.yml <<'EOF'
version: '3.8'
services:
  mongo:
    image: mongo:6
    ports:
      - 27017:27017
    volumes:
      - mongo_data:/data/db
  backend:
    build: ./backend
    environment:
      - MONGO_URI=mongodb://mongo:27017/battery_monitor
      - JWT_SECRET=your_jwt_secret
      - EMAIL_HOST=smtp.example.com
      - EMAIL_USER=your_email@example.com
      - EMAIL_PASS=yourpassword
      - EMAIL_FROM=noreply@yourdomain.com
      - FRONTEND_URL=http://localhost:3000
    ports:
      - "4000:4000"
    depends_on:
      - mongo
  frontend:
    build: ./frontend
    environment:
      - REACT_APP_API_URL=http://localhost:4000/api
    ports:
      - "3000:80"
    depends_on:
      - backend
volumes:
  mongo_data:
EOF

# --- .devcontainer/devcontainer.json ---
mkdir -p battery-monitor/.devcontainer
cat > battery-monitor/.devcontainer/devcontainer.json <<'EOF'
{
  "name": "Battery Monitor Dev",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "backend",
  "workspaceFolder": "/workspaces/battery-monitor",
  "forwardPorts": [3000, 4000],
  "postCreateCommand": "cd backend && npm install && cd ../frontend && npm install",
  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode"
      ]
    }
  }
}
EOF

# --- BACKEND FILES ---

cat > battery-monitor/backend/.env.example <<'EOF'
MONGO_URI=mongodb://mongo:27017/battery_monitor
JWT_SECRET=your_jwt_secret
EMAIL_HOST=smtp.example.com
EMAIL_USER=your_email@example.com
EMAIL_PASS=yourpassword
EMAIL_FROM=noreply@yourdomain.com
FRONTEND_URL=http://localhost:3000
EOF

cat > battery-monitor/backend/Dockerfile <<'EOF'
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
CMD ["npm", "start"]
EOF

cat > battery-monitor/backend/package.json <<'EOF'
{
  "name": "battery-monitor-backend",
  "version": "1.0.0",
  "main": "src/server.js",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js"
  },
  "dependencies": {
    "bcrypt": "^5.1.0",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1",
    "express": "^4.19.2",
    "jsonwebtoken": "^9.0.2",
    "mongoose": "^8.2.0",
    "morgan": "^1.10.0",
    "nodemailer": "^6.9.12",
    "socket.io": "^4.7.5"
  },
  "devDependencies": {
    "nodemon": "^3.1.0"
  }
}
EOF

cat > battery-monitor/backend/src/app.js <<'EOF'
const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const morgan = require('morgan');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

app.use('/api/batteries', require('./routes/battery'));
app.use('/api/device-telemetry', require('./routes/deviceTelemetry'));
app.use('/api/alerts', require('./routes/alert'));
app.use('/api/audit-logs', require('./routes/auditLog'));
// Add other routes (user, password, device key) as above

app.get('/api/health', (req, res) => res.json({ status: 'ok' }));

module.exports = app;
EOF

cat > battery-monitor/backend/src/server.js <<'EOF'
const http = require('http');
const mongoose = require('mongoose');
const app = require('./app');
const { Server } = require('socket.io');

const PORT = process.env.PORT || 4000;
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });
global.io = io; // Make socket.io available globally

mongoose.connect(process.env.MONGO_URI)
  .then(() => {
    server.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
  })
  .catch(err => console.error('MongoDB connection error:', err));
EOF

# ... (rest of backend/src files, see previous script for model/controller/route/service/middleware files)
# For brevity, use the previous messages for backend/src/models, controllers, middleware, etc.
# You can copy the backend/src/models/Battery.js and the rest as shown above.

# --- FRONTEND FILES ---

cat > battery-monitor/frontend/.env.example <<'EOF'
REACT_APP_API_URL=http://localhost:4000/api
EOF

cat > battery-monitor/frontend/Dockerfile <<'EOF'
FROM node:18 as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

cat > battery-monitor/frontend/package.json <<'EOF'
{
  "name": "battery-monitor-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@testing-library/react": "^14.2.0",
    "chart.js": "^4.4.1",
    "react": "^18.3.0",
    "react-chartjs-2": "^5.2.0",
    "react-dom": "^18.3.0",
    "react-leaflet": "^4.3.1",
    "leaflet": "^1.9.4",
    "react-router-dom": "^6.23.0",
    "socket.io-client": "^4.7.5",
    "leaflet.markercluster": "^1.5.3",
    "leaflet.heat": "^0.2.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  }
}
EOF

cat > battery-monitor/frontend/public/index.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Battery Monitor</title>
</head>
<body>
  <div id="root"></div>
</body>
</html>
EOF

cat > battery-monitor/frontend/src/index.js <<'EOF'
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
EOF

cat > battery-monitor/frontend/src/App.js <<'EOF'
import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import DashboardPage from "./pages/DashboardPage";
import AlertManagementPage from "./pages/AlertManagementPage";
import AdminDeviceKeysPage from "./pages/AdminDeviceKeysPage";
import AuditLogPage from "./pages/AuditLogPage";
import BatteryMapPage from "./pages/BatteryMapPage";
import AdvancedTelemetryDashboard from "./pages/AdvancedTelemetryDashboard";
// ...other imports

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/alerts" element={<AlertManagementPage />} />
        <Route path="/admin/device-keys" element={<AdminDeviceKeysPage />} />
        <Route path="/admin/audit-logs" element={<AuditLogPage />} />
        <Route path="/batterymap" element={<BatteryMapPage />} />
        <Route path="/advanced-telemetry" element={<AdvancedTelemetryDashboard />} />
        {/* ...other routes per your previous features */}
      </Routes>
    </BrowserRouter>
  );
}
export default App;
EOF

# -- Minimal example Navbar --
cat > battery-monitor/frontend/src/components/Navbar.js <<'EOF'
import React from "react";
import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav style={{ padding: 8, background: "#eee", marginBottom: 16 }}>
      <Link to="/">Dashboard</Link> |{" "}
      <Link to="/alerts">Alerts</Link> |{" "}
      <Link to="/admin/device-keys">Device Keys</Link> |{" "}
      <Link to="/admin/audit-logs">Audit Logs</Link> |{" "}
      <Link to="/batterymap">Map</Link> |{" "}
      <Link to="/advanced-telemetry">Advanced Telemetry</Link>
    </nav>
  );
}
EOF

cat > battery-monitor/frontend/src/pages/BatteryMapPage.js <<'EOF'
import React, { useEffect, useState } from "react";
import MapClusterHeat from "../components/MapClusterHeat";

export default function BatteryMapPage() {
  const [batteries, setBatteries] = useState([]);
  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    fetch(`${process.env.REACT_APP_API_URL}/batteries`)
      .then(res => res.json())
      .then(setBatteries);
  }, []);

  return (
    <div>
      <h2>Battery Map</h2>
      <label>
        <input type="checkbox" checked={showHeatmap} onChange={e => setShowHeatmap(e.target.checked)} />
        Show Heatmap
      </label>
      <MapClusterHeat batteries={batteries} showHeatmap={showHeatmap} />
    </div>
  );
}
EOF

cat > battery-monitor/frontend/src/components/MapClusterHeat.js <<'EOF'
import React, { useEffect, useRef } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.markercluster";
import "leaflet.heat";

function MarkersCluster({ batteries }) {
  const map = useMap();
  const markerClusterGroup = useRef();
  useEffect(() => {
    if (!map) return;
    if (markerClusterGroup.current) {
      markerClusterGroup.current.clearLayers();
      map.removeLayer(markerClusterGroup.current);
    }
    const clusterGroup = L.markerClusterGroup();
    batteries.forEach(b => {
      if (b.location && b.location.lat && b.location.lng) {
        const marker = L.marker([b.location.lat, b.location.lng]);
        marker.bindPopup(
          `<b>${b.name}</b><br>Voltage: ${b.voltage}<br>Temp: ${b.temperature}<br>Charge: ${b.charge}`
        );
        clusterGroup.addLayer(marker);
      }
    });
    map.addLayer(clusterGroup);
    markerClusterGroup.current = clusterGroup;
    return () => {
      if (markerClusterGroup.current) {
        markerClusterGroup.current.clearLayers();
        map.removeLayer(markerClusterGroup.current);
      }
    };
  }, [batteries, map]);
  return null;
}

function HeatLayer({ batteries }) {
  const map = useMap();
  useEffect(() => {
    const heatData = batteries
      .filter(b => b.location && b.location.lat && b.location.lng)
      .map(b => [b.location.lat, b.location.lng, Math.max(0.1, b.charge / 100)]);
    const heat = L.heatLayer(heatData, { radius: 25, blur: 15 });
    map.addLayer(heat);
    return () => map.removeLayer(heat);
  }, [batteries, map]);
  return null;
}

export default function MapClusterHeat({ batteries, showHeatmap }) {
  const validLocs = batteries.filter(b => b.location && b.location.lat && b.location.lng);
  const center = validLocs.length
    ? [
        validLocs.reduce((sum, b) => sum + b.location.lat, 0) / validLocs.length,
        validLocs.reduce((sum, b) => sum + b.location.lng, 0) / validLocs.length
      ]
    : [0, 0];

  return (
    <MapContainer center={center} zoom={3} style={{ height: 500, width: "100%" }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {showHeatmap ? <HeatLayer batteries={batteries} /> : <MarkersCluster batteries={batteries} />}
    </MapContainer>
  );
}
EOF

# -- Add similar placeholder files for other pages/components/context/services as needed --

echo "Battery Monitor project structure and main files created."
echo "Next: cd battery-monitor, copy .env.example to .env in backend and frontend, then docker-compose up --build"