'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Database,
  GitBranch,
  Zap,
  BarChart3,
  Settings,
  Monitor,
  ArrowLeft,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react'
import Link from 'next/link'

// Sample MLOps data
const pipelineMetrics = {
  totalRuns: 47,
  successRate: 94.2,
  avgExecutionTime: 127,
  modelsDeployed: 12,
  activePipelines: 3,
  lastRun: '2 minutes ago'
}

const modelPerformance = [
  {
    name: 'Anomaly Detection',
    version: 'v2.1.0',
    accuracy: 0.947,
    precision: 0.923,
    recall: 0.981,
    f1Score: 0.951,
    drift: 'none',
    status: 'production',
    lastTrained: '2024-01-15 09:30:00'
  },
  {
    name: 'Cell Health Prediction',
    version: 'v1.8.2',
    accuracy: 0.912,
    precision: 0.889,
    recall: 0.934,
    f1Score: 0.911,
    drift: 'low',
    status: 'production',
    lastTrained: '2024-01-14 16:45:00'
  },
  {
    name: 'Forecasting Model',
    version: 'v3.0.1',
    accuracy: 0.876,
    precision: 0.901,
    recall: 0.845,
    f1Score: 0.872,
    drift: 'medium',
    status: 'staging',
    lastTrained: '2024-01-15 11:20:00'
  }
]

const pipelineHistory = [
  {
    id: 'cd4ml_1705314600_batch_training',
    status: 'completed',
    startTime: '2024-01-15 10:30:00',
    duration: '8m 42s',
    modelsTrained: 3,
    dataQuality: 0.95,
    efficiency: 0.88
  },
  {
    id: 'cd4ml_1705310400_model_validation',
    status: 'completed',
    startTime: '2024-01-15 09:20:00',
    duration: '5m 18s',
    modelsTrained: 1,
    dataQuality: 0.92,
    efficiency: 0.91
  },
  {
    id: 'cd4ml_1705306800_drift_detection',
    status: 'warning',
    startTime: '2024-01-15 08:20:00',
    duration: '12m 05s',
    modelsTrained: 0,
    dataQuality: 0.89,
    efficiency: 0.75
  },
  {
    id: 'cd4ml_1705303200_retrain_forecast',
    status: 'failed',
    startTime: '2024-01-15 07:20:00',
    duration: '3m 22s',
    modelsTrained: 0,
    dataQuality: 0.87,
    efficiency: 0.42
  }
]

const alerts = [
  {
    id: 1,
    type: 'warning',
    message: 'Model drift detected in Forecasting Model',
    timestamp: '5 minutes ago',
    severity: 'medium'
  },
  {
    id: 2,
    type: 'info',
    message: 'CD4ML pipeline completed successfully',
    timestamp: '12 minutes ago',
    severity: 'low'
  },
  {
    id: 3,
    type: 'error',
    message: 'Data quality check failed for device 104',
    timestamp: '1 hour ago',
    severity: 'high'
  }
]

export default function MLOpsDashboard() {
  const [isMonitoring, setIsMonitoring] = useState(true)
  const [selectedModel, setSelectedModel] = useState(modelPerformance[0])
  const [pipelineStatus, setPipelineStatus] = useState('running')

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'production': return 'text-green-400 bg-green-500/20 border-green-500/30'
      case 'staging': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      case 'training': return 'text-blue-400 bg-blue-500/20 border-blue-500/30'
      case 'deprecated': return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
    }
  }

  const getDriftColor = (drift: string) => {
    switch (drift) {
      case 'none': return 'text-green-400'
      case 'low': return 'text-yellow-400'
      case 'medium': return 'text-orange-400'
      case 'high': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getPipelineStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400 bg-green-500/20'
      case 'running': return 'text-blue-400 bg-blue-500/20'
      case 'warning': return 'text-yellow-400 bg-yellow-500/20'
      case 'failed': return 'text-red-400 bg-red-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="header">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <Link href="/" className="btn-secondary flex items-center space-x-2">
                <ArrowLeft className="h-4 w-4" />
                <span>Back</span>
              </Link>
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg animate-glow">
                  <Monitor className="h-7 w-7 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold gradient-text tracking-tight">MLOps Dashboard</h1>
                  <p className="text-sm text-white/70 font-medium">CD4ML Pipeline Monitoring & Model Management</p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${isMonitoring ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-sm font-medium text-white/80">
                  {isMonitoring ? 'Monitoring Active' : 'Monitoring Paused'}
                </span>
              </div>
              <button 
                onClick={() => setIsMonitoring(!isMonitoring)}
                className="btn-secondary flex items-center space-x-2"
              >
                {isMonitoring ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                <span>{isMonitoring ? 'Pause' : 'Resume'}</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Overview Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="stat-card"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <GitBranch className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Pipeline Runs</p>
                <p className="text-xl font-bold text-white">{pipelineMetrics.totalRuns}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="stat-card"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                <CheckCircle className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Success Rate</p>
                <p className="text-xl font-bold gradient-text-success">{pipelineMetrics.successRate}%</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="stat-card"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <Clock className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Avg Execution</p>
                <p className="text-xl font-bold text-white">{pipelineMetrics.avgExecutionTime}s</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="stat-card"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-indigo-500/20 rounded-lg flex items-center justify-center">
                <Database className="h-5 w-5 text-indigo-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Models Deployed</p>
                <p className="text-xl font-bold text-white">{pipelineMetrics.modelsDeployed}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="stat-card"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-yellow-500/20 rounded-lg flex items-center justify-center">
                <Activity className="h-5 w-5 text-yellow-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Active Pipelines</p>
                <p className="text-xl font-bold text-white">{pipelineMetrics.activePipelines}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="stat-card"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center">
                <Zap className="h-5 w-5 text-red-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Last Run</p>
                <p className="text-sm font-bold text-white">{pipelineMetrics.lastRun}</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Model Performance & Pipeline Status */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Model Performance */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Model Performance</h3>
              <div className="w-10 h-10 bg-green-500/20 rounded-xl flex items-center justify-center">
                <BarChart3 className="h-5 w-5 text-green-400" />
              </div>
            </div>
            <div className="space-y-4">
              {modelPerformance.map((model, index) => (
                <div 
                  key={index}
                  className={`p-4 rounded-lg border transition-all duration-300 cursor-pointer ${
                    selectedModel?.name === model.name 
                      ? 'bg-white/10 border-blue-500/50' 
                      : 'bg-white/5 border-white/10 hover:bg-white/8'
                  }`}
                  onClick={() => setSelectedModel(model)}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-semibold text-white">{model.name}</h4>
                      <p className="text-xs text-white/60">Version {model.version}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded-full text-xs border ${getStatusColor(model.status)}`}>
                        {model.status}
                      </span>
                      <span className={`text-xs font-medium ${getDriftColor(model.drift)}`}>
                        {model.drift === 'none' ? 'No Drift' : `${model.drift} Drift`}
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-white/60">Accuracy</p>
                      <p className="font-semibold text-white">{(model.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-white/60">Precision</p>
                      <p className="font-semibold text-white">{(model.precision * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-white/60">Recall</p>
                      <p className="font-semibold text-white">{(model.recall * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-white/60">F1 Score</p>
                      <p className="font-semibold text-white">{(model.f1Score * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Pipeline Status */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Pipeline History</h3>
              <button className="btn-secondary flex items-center space-x-2">
                <RotateCcw className="h-4 w-4" />
                <span>Refresh</span>
              </button>
            </div>
            <div className="space-y-3">
              {pipelineHistory.map((run, index) => (
                <div key={run.id} className="p-4 bg-white/5 rounded-lg border border-white/10">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <span className={`w-3 h-3 rounded-full ${getPipelineStatusColor(run.status)}`}></span>
                      <span className="text-sm font-medium text-white">{run.id}</span>
                    </div>
                    <span className="text-xs text-white/60">{run.startTime}</span>
                  </div>
                  <div className="grid grid-cols-4 gap-4 text-xs">
                    <div>
                      <p className="text-white/60">Duration</p>
                      <p className="font-semibold text-white">{run.duration}</p>
                    </div>
                    <div>
                      <p className="text-white/60">Models</p>
                      <p className="font-semibold text-white">{run.modelsTrained}</p>
                    </div>
                    <div>
                      <p className="text-white/60">Data Quality</p>
                      <p className="font-semibold text-white">{(run.dataQuality * 100).toFixed(0)}%</p>
                    </div>
                    <div>
                      <p className="text-white/60">Efficiency</p>
                      <p className="font-semibold text-white">{(run.efficiency * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Alerts & Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="data-table"
        >
          <div className="px-6 py-4 border-b border-white/10">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold text-white">Recent Alerts & Actions</h3>
              <div className="flex items-center space-x-4">
                <button className="btn-primary">
                  Trigger Pipeline
                </button>
                <Link href="/llm" className="btn-secondary">
                  LLM Dashboard
                </Link>
              </div>
            </div>
          </div>
          <div className="p-6">
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div key={alert.id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/10">
                  <div className="flex items-center space-x-3">
                    {alert.type === 'warning' && <AlertTriangle className="h-5 w-5 text-yellow-400" />}
                    {alert.type === 'error' && <AlertTriangle className="h-5 w-5 text-red-400" />}
                    {alert.type === 'info' && <CheckCircle className="h-5 w-5 text-blue-400" />}
                    <div>
                      <p className="text-white font-medium">{alert.message}</p>
                      <p className="text-xs text-white/60">{alert.timestamp}</p>
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    alert.severity === 'high' ? 'bg-red-500/20 text-red-300' :
                    alert.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-300' :
                    'bg-blue-500/20 text-blue-300'
                  }`}>
                    {alert.severity}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  )
} 