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
