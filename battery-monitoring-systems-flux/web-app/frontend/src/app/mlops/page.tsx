'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  LineChart,
  Monitor,
  Network,
  Play,
  Server,
  Shield,
  Square,
  TrendingDown,
  TrendingUp,
  Zap
} from 'lucide-react'
import Link from 'next/link'

// Sample data for charts

const driftData = [
  { time: '00:00', voltage: 0.05, temperature: 0.03, current: 0.08 },
  { time: '04:00', voltage: 0.07, temperature: 0.04, current: 0.12 },
  { time: '08:00', voltage: 0.09, temperature: 0.06, current: 0.15 },
  { time: '12:00', voltage: 0.11, temperature: 0.08, current: 0.18 },
  { time: '16:00', voltage: 0.12, temperature: 0.09, current: 0.21 },
  { time: '20:00', voltage: 0.13, temperature: 0.10, current: 0.23 },
  { time: '24:00', voltage: 0.12, temperature: 0.08, current: 0.22 }
]

const performanceData = [
  { time: '00:00', accuracy: 94.2, latency: 45, throughput: 1200 },
  { time: '04:00', accuracy: 93.8, latency: 48, throughput: 1180 },
  { time: '08:00', accuracy: 94.5, latency: 42, throughput: 1250 },
  { time: '12:00', accuracy: 93.9, latency: 50, throughput: 1150 },
  { time: '16:00', accuracy: 94.1, latency: 46, throughput: 1220 },
  { time: '20:00', accuracy: 94.3, latency: 44, throughput: 1240 },
  { time: '24:00', accuracy: 94.0, latency: 47, throughput: 1210 }
]

export default function MLOpsDashboard() {
  const [isClient, setIsClient] = useState(false)
  const [mlopsData, setMlopsData] = useState({
    monitoring_active: false,
    systemHealth: {
      status: "healthy",
      uptime: "99.8%",
      latency: "45ms",
      throughput: "1250 req/s",
      errorRate: "0.02%",
      cpuUsage: "23%",
      memoryUsage: "67%",
      gpuUsage: "12%"
    },
    modelPerformance: {
      anomalyDetector: { accuracy: 94.2, f1Score: 0.91, latency: 12 },
      cellPredictor: { accuracy: 89.7, f1Score: 0.87, latency: 8 },
      forecaster: { mse: 0.023, mae: 0.045, latency: 15 }
    },
    dataDrift: {
      voltage: { driftScore: 0.12, status: "normal", trend: "stable" },
      temperature: { driftScore: 0.08, status: "normal", trend: "stable" },
      current: { driftScore: 0.23, status: "warning", trend: "increasing" }
    },
    alerts: [
      { id: 1, type: "drift", severity: "warning", message: "Current data drift detected", time: "2 min ago" },
      { id: 2, type: "performance", severity: "info", message: "Model retraining completed", time: "15 min ago" },
      { id: 3, type: "system", severity: "info", message: "Backup completed successfully", time: "1 hour ago" }
    ],
    metrics: {
      totalPredictions: 15420,
      successfulPredictions: 15380,
      failedPredictions: 40,
      avgResponseTime: 45,
      dataQualityScore: 98.5,
      modelAccuracy: 92.3
    }
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    const fetchMlopsData = async () => {
      try {
        const response = await fetch('/api/mlops/status');
        if (response.ok) {
          const data = await response.json();
          setMlopsData(data);
        }
      } catch (error) {
        console.error('Error fetching MLOps data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMlopsData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchMlopsData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-400'
      case 'warning': return 'text-yellow-400'
      case 'critical': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'warning': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'info': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="header">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <Link href="/">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg animate-glow cursor-pointer">
                  <Monitor className="h-7 w-7 text-white" />
                </div>
              </Link>
              <div>
                <h1 className="text-2xl font-bold gradient-text tracking-tight">MLOps Dashboard</h1>
                <p className="text-sm text-white/70 font-medium">Monitoring & Observability</p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${mlopsData.monitoring_active ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
                <span className="text-sm font-medium text-white/80">
                  {mlopsData.monitoring_active ? 'Monitoring Active' : 'Monitoring Inactive'}
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <button 
                  onClick={async () => {
                    try {
                      const response = await fetch('/api/mlops/start-monitoring', { method: 'POST' });
                      if (response.ok) {
                        window.location.reload();
                      }
                    } catch (error) {
                      console.error('Error starting monitoring:', error);
                    }
                  }}
                  className="btn-primary flex items-center space-x-2 group"
                  disabled={mlopsData.monitoring_active}
                >
                  <Play className="h-4 w-4 group-hover:scale-110 transition-transform" />
                  <span>Start Monitoring</span>
                </button>
                <button 
                  onClick={async () => {
                    try {
                      const response = await fetch('/api/mlops/stop-monitoring', { method: 'POST' });
                      if (response.ok) {
                        window.location.reload();
                      }
                    } catch (error) {
                      console.error('Error stopping monitoring:', error);
                    }
                  }}
                  className="btn-secondary flex items-center space-x-2 group"
                  disabled={!mlopsData.monitoring_active}
                >
                  <Square className="h-4 w-4 group-hover:scale-110 transition-transform" />
                  <span>Stop Monitoring</span>
                </button>
              </div>
              <Link href="/">
                <button className="btn-secondary flex items-center space-x-2 group">
                  <BarChart3 className="h-5 w-5 group-hover:scale-110 transition-transform" />
                  <span>Back to Dashboard</span>
                </button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {/* System Health Overview - Compact Horizontal Layout */}
        <div className="flex flex-wrap items-center gap-4 mb-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <Server className="h-4 w-4 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">System Status</p>
                <p className={`text-lg font-bold ${getStatusColor(mlopsData.systemHealth.status)}`}>
                  {mlopsData.systemHealth.status.charAt(0).toUpperCase() + mlopsData.systemHealth.status.slice(1)}
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <Gauge className="h-4 w-4 text-blue-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Avg Response</p>
                <p className="text-lg font-bold text-white">{mlopsData.systemHealth.latency}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <Zap className="h-4 w-4 text-purple-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Throughput</p>
                <p className="text-lg font-bold text-white">{mlopsData.systemHealth.throughput}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-red-500/20 rounded-lg flex items-center justify-center">
                <AlertTriangle className="h-4 w-4 text-red-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Error Rate</p>
                <p className="text-lg font-bold text-white">{mlopsData.systemHealth.errorRate}</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Resource Usage & Model Performance - Compact Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Resource Usage</h3>
              <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <Cpu className="h-4 w-4 text-blue-400" />
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-white/70 text-sm">CPU Usage</span>
                <span className="text-white font-semibold text-sm">{mlopsData.systemHealth.cpuUsage}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-1.5">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-purple-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: mlopsData.systemHealth.cpuUsage }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-white/70 text-sm">Memory Usage</span>
                <span className="text-white font-semibold text-sm">{mlopsData.systemHealth.memoryUsage}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-1.5">
                <div 
                  className="bg-gradient-to-r from-green-500 to-blue-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: mlopsData.systemHealth.memoryUsage }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-white/70 text-sm">GPU Usage</span>
                <span className="text-white font-semibold text-sm">{mlopsData.systemHealth.gpuUsage}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-1.5">
                <div 
                  className="bg-gradient-to-r from-purple-500 to-pink-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: mlopsData.systemHealth.gpuUsage }}
                ></div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Model Performance</h3>
              <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <BarChart3 className="h-4 w-4 text-green-400" />
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-white/70 text-sm">Anomaly Detector</span>
                <span className="text-white font-semibold text-sm">{mlopsData.modelPerformance.anomalyDetector.accuracy}%</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-1.5">
                <div 
                  className="bg-gradient-to-r from-green-500 to-emerald-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${mlopsData.modelPerformance.anomalyDetector.accuracy}%` }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-white/70 text-sm">Cell Predictor</span>
                <span className="text-white font-semibold text-sm">{mlopsData.modelPerformance.cellPredictor.accuracy}%</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-1.5">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-cyan-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${mlopsData.modelPerformance.cellPredictor.accuracy}%` }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-white/70 text-sm">Forecaster (MSE)</span>
                <span className="text-white font-semibold text-sm">{mlopsData.modelPerformance.forecaster.mse}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-1.5">
                <div 
                  className="bg-gradient-to-r from-purple-500 to-violet-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${(1 - mlopsData.modelPerformance.forecaster.mse) * 100}%` }}
                ></div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Data Drift</h3>
              <div className="w-10 h-10 bg-yellow-500/20 rounded-xl flex items-center justify-center">
                <TrendingDown className="h-5 w-5 text-yellow-400" />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4">
              {/* Voltage */}
              <div className="flex flex-col space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-white/70 text-sm">Voltage</span>
                  <span className={`font-semibold text-sm ${getStatusColor(mlopsData.dataDrift.voltage.status)}`}>
                    {mlopsData.dataDrift.voltage.driftScore}
                  </span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      mlopsData.dataDrift.voltage.status === 'warning' 
                        ? 'bg-gradient-to-r from-yellow-500 to-orange-600'
                        : 'bg-gradient-to-r from-green-500 to-emerald-600'
                    }`}
                    style={{ width: `${mlopsData.dataDrift.voltage.driftScore * 400}%` }}
                  ></div>
                </div>
              </div>
              
              {/* Temperature */}
              <div className="flex flex-col space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-white/70 text-sm">Temperature</span>
                  <span className={`font-semibold text-sm ${getStatusColor(mlopsData.dataDrift.temperature.status)}`}>
                    {mlopsData.dataDrift.temperature.driftScore}
                  </span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      mlopsData.dataDrift.temperature.status === 'warning' 
                        ? 'bg-gradient-to-r from-yellow-500 to-orange-600'
                        : 'bg-gradient-to-r from-green-500 to-emerald-600'
                    }`}
                    style={{ width: `${mlopsData.dataDrift.temperature.driftScore * 400}%` }}
                  ></div>
                </div>
              </div>
              
              {/* Current */}
              <div className="flex flex-col space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-white/70 text-sm">Current</span>
                  <span className={`font-semibold text-sm ${getStatusColor(mlopsData.dataDrift.current.status)}`}>
                    {mlopsData.dataDrift.current.driftScore}
                  </span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      mlopsData.dataDrift.current.status === 'warning' 
                        ? 'bg-gradient-to-r from-yellow-500 to-orange-600'
                        : 'bg-gradient-to-r from-green-500 to-emerald-600'
                    }`}
                    style={{ width: `${mlopsData.dataDrift.current.driftScore * 400}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Data Drift Trend</h3>
              <div className="w-8 h-8 bg-red-500/20 rounded-lg flex items-center justify-center">
                <TrendingDown className="h-4 w-4 text-red-400" />
              </div>
            </div>
            <div className="h-48 bg-white/5 rounded-xl flex items-center justify-center p-4">
              {isClient ? (
                <div className="w-full h-full flex items-end justify-between space-x-2">
                  {driftData.map((point, index) => (
                    <div key={index} className="flex flex-col items-center space-y-2">
                      <div className="flex flex-col space-y-1">
                        <div 
                          className="w-8 bg-gradient-to-t from-blue-500 to-purple-600 rounded-t"
                          style={{ height: `${point.voltage * 150}px` }}
                        ></div>
                        <div 
                          className="w-8 bg-gradient-to-t from-green-500 to-emerald-600 rounded-t"
                          style={{ height: `${point.temperature * 150}px` }}
                        ></div>
                        <div 
                          className="w-8 bg-gradient-to-t from-yellow-500 to-orange-600 rounded-t"
                          style={{ height: `${point.current * 150}px` }}
                        ></div>
                      </div>
                      <span className="text-xs text-white/60">{point.time}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
                </div>
              )}
            </div>
            <div className="flex justify-center space-x-6 mt-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-xs text-white/70">Voltage</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-xs text-white/70">Temperature</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span className="text-xs text-white/70">Current</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Performance Metrics</h3>
              <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <TrendingUp className="h-4 w-4 text-green-400" />
              </div>
            </div>
            <div className="h-48 bg-white/5 rounded-xl flex items-center justify-center p-4">
              {isClient ? (
                <div className="w-full h-full flex items-end justify-between space-x-2">
                  {performanceData.map((point, index) => (
                    <div key={index} className="flex flex-col items-center space-y-2">
                      <div className="flex flex-col space-y-1">
                        <div 
                          className="w-8 bg-gradient-to-t from-green-500 to-emerald-600 rounded-t"
                          style={{ height: `${point.accuracy - 90}px` }}
                        ></div>
                        <div 
                          className="w-8 bg-gradient-to-t from-blue-500 to-cyan-600 rounded-t"
                          style={{ height: `${point.latency / 2}px` }}
                        ></div>
                      </div>
                      <span className="text-xs text-white/60">{point.time}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-400"></div>
                </div>
              )}
            </div>
            <div className="flex justify-center space-x-6 mt-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-xs text-white/70">Accuracy (%)</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-xs text-white/70">Latency (ms)</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Alerts and Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
            className="lg:col-span-2 data-table"
          >
            <div className="px-6 py-4 border-b border-white/10">
              <h3 className="text-xl font-semibold text-white">Recent Alerts</h3>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {mlopsData.alerts.map((alert) => (
                  <div 
                    key={alert.id}
                    className={`p-4 rounded-lg border ${getSeverityColor(alert.severity)}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <AlertTriangle className="h-5 w-5" />
                        <span className="font-medium">{alert.message}</span>
                      </div>
                      <span className="text-sm opacity-70">{alert.time}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.1 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Key Metrics</h3>
              <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
                <Activity className="h-5 w-5 text-purple-400" />
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Total Predictions</span>
                <span className="text-white font-semibold">{mlopsData.metrics.totalPredictions.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Success Rate</span>
                <span className="text-green-400 font-semibold">
                  {((mlopsData.metrics.successfulPredictions / mlopsData.metrics.totalPredictions) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Data Quality</span>
                <span className="text-blue-400 font-semibold">{mlopsData.metrics.dataQualityScore}%</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Model Accuracy</span>
                <span className="text-purple-400 font-semibold">{mlopsData.metrics.modelAccuracy}%</span>
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
} 