'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Brain,
  MessageSquare,
  TrendingUp,
  Clock,
  DollarSign,
  Target,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Zap,
  Shield,
  ArrowLeft,
  Play,
  Pause,
  RotateCcw,
  Settings,
  Eye,
  TestTube
} from 'lucide-react'
import Link from 'next/link'

// Sample LLMOps data
const llmMetrics = {
  totalRequests: 2847,
  avgQualityScore: 0.89,
  avgResponseTime: 1250,
  totalCost: 12.47,
  activeBTests: 2,
  lastRequest: '30 seconds ago'
}

const modelComparison = [
  {
    name: 'Llama2-7B',
    version: 'v2.1.0',
    qualityScore: 0.892,
    responseTime: 1180,
    costPerRequest: 0.0045,
    requestCount: 1523,
    biasScore: 0.12,
    safetyScore: 0.97,
    status: 'production'
  },
  {
    name: 'ChatGPT-3.5',
    version: 'gpt-3.5-turbo',
    qualityScore: 0.945,
    responseTime: 850,
    costPerRequest: 0.0021,
    requestCount: 987,
    biasScore: 0.08,
    safetyScore: 0.99,
    status: 'testing'
  },
  {
    name: 'Claude-2',
    version: 'claude-2.1',
    qualityScore: 0.923,
    responseTime: 1520,
    costPerRequest: 0.0063,
    requestCount: 337,
    biasScore: 0.05,
    safetyScore: 0.98,
    status: 'staging'
  }
]

const abTests = [
  {
    id: 'test_001',
    name: 'Battery Analysis Prompts',
    modelA: 'Llama2-7B',
    modelB: 'ChatGPT-3.5',
    status: 'running',
    trafficSplit: '50/50',
    duration: '2d 4h',
    performance: {
      modelA: { quality: 0.87, speed: 1180 },
      modelB: { quality: 0.94, speed: 850 }
    },
    confidenceLevel: 0.85
  },
  {
    id: 'test_002',
    name: 'Anomaly Explanation',
    modelA: 'Claude-2',
    modelB: 'Llama2-7B',
    status: 'completed',
    trafficSplit: '30/70',
    duration: '5d 12h',
    performance: {
      modelA: { quality: 0.91, speed: 1520 },
      modelB: { quality: 0.89, speed: 1180 }
    },
    confidenceLevel: 0.92
  }
]

const promptTemplates = [
  {
    id: 'battery_analysis',
    name: 'Battery Data Analysis',
    template: 'Analyze the following battery cell data for {device_id}: Voltage={voltage}V, Temperature={temperature}°C, Specific Gravity={specific_gravity}. Provide insights and recommendations.',
    performance: {
      qualityScore: 0.91,
      relevanceScore: 0.94,
      coherenceScore: 0.88,
      avgResponseTime: 1450
    },
    usageCount: 847,
    lastUsed: '5 minutes ago'
  },
  {
    id: 'anomaly_explanation',
    name: 'Anomaly Detection Explanation',
    template: 'Explain why the following battery readings are considered anomalous: {anomaly_data}. Include potential causes and recommended actions for {device_type}.',
    performance: {
      qualityScore: 0.87,
      relevanceScore: 0.89,
      coherenceScore: 0.92,
      avgResponseTime: 1680
    },
    usageCount: 623,
    lastUsed: '12 minutes ago'
  },
  {
    id: 'forecast_insights',
    name: 'Forecast Explanation',
    template: 'Based on the predicted battery values for {forecast_period}, explain the expected performance trends and maintenance recommendations for device {device_id}.',
    performance: {
      qualityScore: 0.84,
      relevanceScore: 0.86,
      coherenceScore: 0.90,
      avgResponseTime: 1820
    },
    usageCount: 412,
    lastUsed: '8 minutes ago'
  }
]

const qualityMetrics = [
  { time: '00:00', quality: 0.88, relevance: 0.91, coherence: 0.85, safety: 0.97 },
  { time: '04:00', quality: 0.89, relevance: 0.93, coherence: 0.87, safety: 0.98 },
  { time: '08:00', quality: 0.91, relevance: 0.89, coherence: 0.92, safety: 0.96 },
  { time: '12:00', quality: 0.87, relevance: 0.88, coherence: 0.89, safety: 0.98 },
  { time: '16:00', quality: 0.92, relevance: 0.94, coherence: 0.91, safety: 0.97 },
  { time: '20:00', quality: 0.90, relevance: 0.92, coherence: 0.88, safety: 0.99 }
]

const alerts = [
  {
    id: 1,
    type: 'warning',
    message: 'Response quality below threshold for battery analysis prompts',
    timestamp: '3 minutes ago',
    severity: 'medium'
  },
  {
    id: 2,
    type: 'info',
    message: 'A/B test "Battery Analysis Prompts" showing significant improvement',
    timestamp: '15 minutes ago',
    severity: 'low'
  },
  {
    id: 3,
    type: 'error',
    message: 'High bias score detected in anomaly explanations',
    timestamp: '45 minutes ago',
    severity: 'high'
  }
]

export default function LLMDashboard() {
  const [isMonitoring, setIsMonitoring] = useState(true)
  const [selectedModel, setSelectedModel] = useState(modelComparison[0])
  const [selectedPrompt, setSelectedPrompt] = useState(promptTemplates[0])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'production': return 'text-green-400 bg-green-500/20 border-green-500/30'
      case 'testing': return 'text-blue-400 bg-blue-500/20 border-blue-500/30'
      case 'staging': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
    }
  }

  const getTestStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-blue-400 bg-blue-500/20'
      case 'completed': return 'text-green-400 bg-green-500/20'
      case 'paused': return 'text-yellow-400 bg-yellow-500/20'
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
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg animate-glow">
                  <Brain className="h-7 w-7 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold gradient-text tracking-tight">LLMOps Dashboard</h1>
                  <p className="text-sm text-white/70 font-medium">LLM Performance Monitoring & Optimization</p>
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
              <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <MessageSquare className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Total Requests</p>
                <p className="text-xl font-bold text-white">{llmMetrics.totalRequests.toLocaleString()}</p>
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
                <Target className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Avg Quality</p>
                <p className="text-xl font-bold gradient-text-success">{(llmMetrics.avgQualityScore * 100).toFixed(1)}%</p>
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
              <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <Clock className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Avg Response</p>
                <p className="text-xl font-bold text-white">{llmMetrics.avgResponseTime}ms</p>
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
              <div className="w-10 h-10 bg-yellow-500/20 rounded-lg flex items-center justify-center">
                <DollarSign className="h-5 w-5 text-yellow-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Total Cost</p>
                <p className="text-xl font-bold text-white">${llmMetrics.totalCost.toFixed(2)}</p>
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
              <div className="w-10 h-10 bg-indigo-500/20 rounded-lg flex items-center justify-center">
                <TestTube className="h-5 w-5 text-indigo-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">A/B Tests</p>
                <p className="text-xl font-bold text-white">{llmMetrics.activeBTests}</p>
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
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Last Request</p>
                <p className="text-sm font-bold text-white">{llmMetrics.lastRequest}</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Model Performance & A/B Tests */}
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
              <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
                <Brain className="h-5 w-5 text-purple-400" />
              </div>
            </div>
            <div className="space-y-4">
              {modelComparison.map((model, index) => (
                <div 
                  key={index}
                  className={`p-4 rounded-lg border transition-all duration-300 cursor-pointer ${
                    selectedModel?.name === model.name 
                      ? 'bg-white/10 border-purple-500/50' 
                      : 'bg-white/5 border-white/10 hover:bg-white/8'
                  }`}
                  onClick={() => setSelectedModel(model)}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-semibold text-white">{model.name}</h4>
                      <p className="text-xs text-white/60">{model.version}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded-full text-xs border ${getStatusColor(model.status)}`}>
                        {model.status}
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm mb-3">
                    <div>
                      <p className="text-white/60">Quality</p>
                      <p className="font-semibold text-white">{(model.qualityScore * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-white/60">Speed</p>
                      <p className="font-semibold text-white">{model.responseTime}ms</p>
                    </div>
                    <div>
                      <p className="text-white/60">Cost</p>
                      <p className="font-semibold text-white">${model.costPerRequest.toFixed(4)}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-white/60">Requests</p>
                      <p className="font-semibold text-white">{model.requestCount.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-white/60">Bias</p>
                      <p className={`font-semibold ${model.biasScore < 0.1 ? 'text-green-400' : model.biasScore < 0.2 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {(model.biasScore * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-white/60">Safety</p>
                      <p className="font-semibold text-green-400">{(model.safetyScore * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* A/B Tests */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">A/B Tests</h3>
              <button className="btn-secondary flex items-center space-x-2">
                <TestTube className="h-4 w-4" />
                <span>New Test</span>
              </button>
            </div>
            <div className="space-y-4">
              {abTests.map((test, index) => (
                <div key={test.id} className="p-4 bg-white/5 rounded-lg border border-white/10">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-semibold text-white">{test.name}</h4>
                      <p className="text-xs text-white/60">{test.id}</p>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs ${getTestStatusColor(test.status)}`}>
                      {test.status}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mb-3">
                    <div>
                      <p className="text-xs text-white/60">Model A: {test.modelA}</p>
                      <div className="flex justify-between text-sm">
                        <span>Quality: {(test.performance.modelA.quality * 100).toFixed(1)}%</span>
                        <span>Speed: {test.performance.modelA.speed}ms</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-white/60">Model B: {test.modelB}</p>
                      <div className="flex justify-between text-sm">
                        <span>Quality: {(test.performance.modelB.quality * 100).toFixed(1)}%</span>
                        <span>Speed: {test.performance.modelB.speed}ms</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-white/60">
                    <span>Split: {test.trafficSplit}</span>
                    <span>Duration: {test.duration}</span>
                    <span>Confidence: {(test.confidenceLevel * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Prompt Templates */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="chart-container mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">Prompt Templates</h3>
            <div className="flex items-center space-x-2">
              <button className="btn-secondary flex items-center space-x-2">
                <Settings className="h-4 w-4" />
                <span>Optimize</span>
              </button>
              <button className="btn-secondary flex items-center space-x-2">
                <Eye className="h-4 w-4" />
                <span>Analyze</span>
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {promptTemplates.map((prompt, index) => (
              <div 
                key={prompt.id}
                className={`p-4 rounded-lg border transition-all duration-300 cursor-pointer ${
                  selectedPrompt?.id === prompt.id 
                    ? 'bg-white/10 border-blue-500/50' 
                    : 'bg-white/5 border-white/10 hover:bg-white/8'
                }`}
                onClick={() => setSelectedPrompt(prompt)}
              >
                <div className="mb-3">
                  <h4 className="font-semibold text-white mb-1">{prompt.name}</h4>
                  <p className="text-xs text-white/60 line-clamp-2">{prompt.template}</p>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm mb-3">
                  <div>
                    <p className="text-white/60">Quality</p>
                    <p className="font-semibold text-white">{(prompt.performance.qualityScore * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-white/60">Relevance</p>
                    <p className="font-semibold text-white">{(prompt.performance.relevanceScore * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-white/60">Coherence</p>
                    <p className="font-semibold text-white">{(prompt.performance.coherenceScore * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-white/60">Avg Time</p>
                    <p className="font-semibold text-white">{prompt.performance.avgResponseTime}ms</p>
                  </div>
                </div>
                <div className="flex justify-between text-xs text-white/60">
                  <span>Used: {prompt.usageCount} times</span>
                  <span>Last: {prompt.lastUsed}</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Quality Trends & Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Quality Trends */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Quality Trends</h3>
              <div className="w-10 h-10 bg-green-500/20 rounded-xl flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-green-400" />
              </div>
            </div>
            <div className="h-64 bg-white/5 rounded-xl flex items-center justify-center p-4">
              <div className="w-full h-full flex items-end justify-between space-x-2">
                {qualityMetrics.map((point, index) => (
                  <div key={index} className="flex flex-col items-center space-y-2">
                    <div className="flex flex-col space-y-1">
                      <div 
                        className="w-6 bg-gradient-to-t from-green-500 to-emerald-600 rounded-t"
                        style={{ height: `${point.quality * 120}px` }}
                      ></div>
                      <div 
                        className="w-6 bg-gradient-to-t from-blue-500 to-cyan-600 rounded-t"
                        style={{ height: `${point.relevance * 120}px` }}
                      ></div>
                      <div 
                        className="w-6 bg-gradient-to-t from-purple-500 to-violet-600 rounded-t"
                        style={{ height: `${point.coherence * 120}px` }}
                      ></div>
                      <div 
                        className="w-6 bg-gradient-to-t from-red-500 to-pink-600 rounded-t"
                        style={{ height: `${point.safety * 120}px` }}
                      ></div>
                    </div>
                    <span className="text-xs text-white/60">{point.time}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="flex justify-center space-x-4 mt-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-xs text-white/70">Quality</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-xs text-white/70">Relevance</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                <span className="text-xs text-white/70">Coherence</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-xs text-white/70">Safety</span>
              </div>
            </div>
          </motion.div>

          {/* Alerts & Actions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.1 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Recent Alerts</h3>
              <div className="flex items-center space-x-2">
                <button className="btn-primary">
                  Start A/B Test
                </button>
                <Link href="/mlops" className="btn-secondary">
                  MLOps Dashboard
                </Link>
              </div>
            </div>
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
              
              {/* Additional metrics */}
              <div className="mt-6 p-4 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-lg border border-purple-500/20">
                <h4 className="font-semibold text-white mb-3">Quick Actions</h4>
                <div className="grid grid-cols-2 gap-3">
                  <button className="btn-secondary text-sm">
                    <Shield className="h-4 w-4 mr-2" />
                    Bias Analysis
                  </button>
                  <button className="btn-secondary text-sm">
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Cost Analysis
                  </button>
                  <button className="btn-secondary text-sm">
                    <Settings className="h-4 w-4 mr-2" />
                    Optimize Prompts
                  </button>
                  <button className="btn-secondary text-sm">
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Retrain Model
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
} 