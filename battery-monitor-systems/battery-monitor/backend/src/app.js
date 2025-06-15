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
