# Battery Monitoring System - Next.js Frontend

A modern, beautiful, and responsive Next.js frontend for the Battery Monitoring System with ML/LLM & MLOps capabilities.

## 🚀 Features

- **Modern UI/UX**: Built with Next.js 15, TypeScript, and Tailwind CSS
- **Real-time Dashboard**: Live battery monitoring with WebSocket connections
- **Interactive Charts**: Beautiful charts using Recharts for voltage, temperature, and specific gravity
- **AI Chatbot**: Integrated LLM-powered chatbot for data analysis
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Dark Mode Support**: Automatic dark/light theme switching
- **Real-time Updates**: Live data updates with connection status indicators
- **Advanced Filtering**: Filter by device and cell number
- **Professional Animations**: Smooth animations using Framer Motion

## 🛠️ Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **Utilities**: clsx, tailwind-merge

## 📦 Installation

1. **Navigate to the frontend directory**:
   ```bash
   cd web-app/frontend-next
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Open your browser** and visit `http://localhost:3000`

## 🔧 Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

### Project Structure

```
src/
├── app/
│   ├── globals.css          # Global styles
│   ├── layout.tsx           # Root layout
│   └── page.tsx             # Main dashboard page
├── lib/
│   └── utils.ts             # Utility functions
└── components/              # Reusable components (future)
```

## 🌐 API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000` and includes:

- **REST API Endpoints**: Battery data, system status, devices, cells
- **WebSocket Connection**: Real-time updates at `ws://localhost:8000/ws`
- **AI Chatbot**: LLM integration for data analysis

## 🎨 UI Components

### Dashboard Features

1. **Header**: System title, connection status, and chat button
2. **Stats Grid**: Key metrics with animated cards
3. **Filters**: Device and cell number filtering
4. **Charts**: Interactive line charts for voltage and temperature
5. **Data Table**: Sortable battery data table
6. **Chat Modal**: AI assistant for data queries

### Design System

- **Colors**: Professional blue/gray color scheme
- **Typography**: Inter font for excellent readability
- **Spacing**: Consistent spacing using Tailwind's spacing scale
- **Animations**: Smooth transitions and micro-interactions
- **Responsive**: Mobile-first design approach

## 🔌 Backend Requirements

Ensure the FastAPI backend is running with:

- **API Server**: `http://localhost:8000`
- **WebSocket**: `ws://localhost:8000/ws`
- **CORS**: Enabled for frontend domain
- **Endpoints**: All battery monitoring API endpoints

## 🚀 Production Deployment

### Build for Production

```bash
npm run build
npm run start
```

### Environment Variables

Create a `.env.local` file for production settings:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

### Docker Deployment

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🎯 Key Improvements Over Vanilla JS

1. **Component Architecture**: Reusable, maintainable components
2. **Type Safety**: Full TypeScript support
3. **Modern Styling**: Tailwind CSS with utility classes
4. **Performance**: Next.js optimizations and code splitting
5. **Developer Experience**: Hot reload, TypeScript, ESLint
6. **Accessibility**: Built-in accessibility features
7. **SEO**: Server-side rendering capabilities
8. **Scalability**: Easy to add new features and pages

## 🔮 Future Enhancements

- [ ] Add more chart types (bar charts, pie charts)
- [ ] Implement user authentication
- [ ] Add data export functionality
- [ ] Create mobile app with React Native
- [ ] Add more AI features and insights
- [ ] Implement real-time alerts and notifications

## 📝 License

This project is part of the Battery Monitoring System and follows the same license terms.
