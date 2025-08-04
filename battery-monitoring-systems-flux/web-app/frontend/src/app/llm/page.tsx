'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Brain,
  Bot,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  FileText,
  GitBranch,
  Globe,
  HardDrive,
  MessageCircle,
  Play,
  Settings,
  Shield,
  TestTube,
  TrendingUp,
  Zap
} from 'lucide-react'
import Link from 'next/link'

// Sample LLM data
const llmData = {
  ollama: {
    status: 'connected',
    model: 'llama2:7b',
    version: '2.0.0',
    responseTime: '1.2s',
    tokensPerSecond: 45,
    memoryUsage: '8.2GB',
    gpuUsage: '78%',
    temperature: 0.7,
    maxTokens: 2048
  },
  evaluation: {
    accuracy: 94.2,
    relevance: 91.8,
    coherence: 93.5,
    fluency: 96.1,
    safety: 98.7,
    bias: 2.3,
    toxicity: 0.8
  },
  performance: {
    totalRequests: 15420,
    successfulRequests: 15380,
    failedRequests: 40,
    avgResponseTime: 1200,
    throughput: 1250,
    errorRate: 0.26
  },
  tests: [
    { id: 1, name: 'Battery Analysis', status: 'passed', score: 94.2, time: '2 min ago' },
    { id: 2, name: 'Anomaly Detection', status: 'passed', score: 91.8, time: '5 min ago' },
    { id: 3, name: 'Safety Check', status: 'passed', score: 98.7, time: '8 min ago' },
    { id: 4, name: 'Bias Detection', status: 'warning', score: 85.2, time: '12 min ago' }
  ],
  conversations: [
    { id: 1, query: 'Analyze battery voltage trends', response: 'Based on the data, I can see...', timestamp: '2 min ago' },
    { id: 2, query: 'Detect anomalies in temperature', response: 'I found several temperature spikes...', timestamp: '5 min ago' },
    { id: 3, query: 'Predict battery life', response: 'Based on current usage patterns...', timestamp: '8 min ago' }
  ]
}

const evaluationData = [
  { metric: 'Accuracy', score: 94.2, threshold: 90, status: 'passed' },
  { metric: 'Relevance', score: 91.8, threshold: 85, status: 'passed' },
  { metric: 'Coherence', score: 93.5, threshold: 88, status: 'passed' },
  { metric: 'Fluency', score: 96.1, threshold: 90, status: 'passed' },
  { metric: 'Safety', score: 98.7, threshold: 95, status: 'passed' },
  { metric: 'Bias', score: 2.3, threshold: 5, status: 'passed' },
  { metric: 'Toxicity', score: 0.8, threshold: 2, status: 'passed' }
]

const performanceData = [
  { time: '00:00', requests: 1200, responseTime: 1100, accuracy: 94.2 },
  { time: '04:00', requests: 1180, responseTime: 1150, accuracy: 93.8 },
  { time: '08:00', requests: 1250, responseTime: 1050, accuracy: 94.5 },
  { time: '12:00', requests: 1150, responseTime: 1200, accuracy: 93.9 },
  { time: '16:00', requests: 1220, responseTime: 1120, accuracy: 94.1 },
  { time: '20:00', requests: 1240, responseTime: 1080, accuracy: 94.3 },
  { time: '24:00', requests: 1210, responseTime: 1140, accuracy: 94.0 }
]

export default function LLMDashboard() {
  const [isClient, setIsClient] = useState(false)
  const [testQuery, setTestQuery] = useState('')
  const [testResponse, setTestResponse] = useState('')
  const [isTesting, setIsTesting] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-green-400'
      case 'disconnected': return 'text-red-400'
      case 'warning': return 'text-yellow-400'
      case 'passed': return 'text-green-400'
      case 'failed': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusBg = (status: string) => {
    switch (status) {
      case 'connected': return 'bg-green-500/20'
      case 'disconnected': return 'bg-red-500/20'
      case 'warning': return 'bg-yellow-500/20'
      case 'passed': return 'bg-green-500/20'
      case 'failed': return 'bg-red-500/20'
      default: return 'bg-gray-500/20'
    }
  }

  const handleTestQuery = async () => {
    if (!testQuery.trim()) return
    
    setIsTesting(true)
    setTestResponse('')
    
    // Simulate API call
    setTimeout(() => {
      setTestResponse(`Based on your query "${testQuery}", I can provide the following analysis:

1. **Battery Health Assessment**: The current voltage readings indicate normal operation within expected parameters.

2. **Temperature Analysis**: Temperature trends show stable conditions with no significant anomalies detected.

3. **Performance Metrics**: Overall system performance is optimal with efficiency scores above 90%.

4. **Recommendations**: Continue monitoring and schedule routine maintenance as per standard protocols.

This analysis is based on real-time data from your battery monitoring system.`)
      setIsTesting(false)
    }, 2000)
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
                  <Brain className="h-7 w-7 text-white" />
                </div>
              </Link>
              <div>
                <h1 className="text-2xl font-bold gradient-text tracking-tight">LLM & DeepEval</h1>
                <p className="text-sm text-white/70 font-medium">Ollama 2 7B Integration & Model Evaluation</p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${llmData.ollama.status === 'connected' ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
                <span className="text-sm font-medium text-white/80">
                  {llmData.ollama.status === 'connected' ? 'Ollama Connected' : 'Ollama Disconnected'}
                </span>
              </div>
              <Link href="/">
                <button className="btn-secondary flex items-center space-x-2 group">
                  <Bot className="h-5 w-5 group-hover:scale-110 transition-transform" />
                  <span>Back to Dashboard</span>
                </button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Ollama Status Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="stat-card"
          >
            <div className="flex items-center space-x-3">
              <div className={`w-10 h-10 ${getStatusBg(llmData.ollama.status)} rounded-lg flex items-center justify-center`}>
                <Bot className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Ollama Status</p>
                <p className={`text-xl font-bold ${getStatusColor(llmData.ollama.status)}`}>
                  {llmData.ollama.status.charAt(0).toUpperCase() + llmData.ollama.status.slice(1)}
                </p>
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
              <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <Cpu className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Model</p>
                <p className="text-xl font-bold text-white">{llmData.ollama.model}</p>
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
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Response Time</p>
                <p className="text-xl font-bold text-white">{llmData.ollama.responseTime}</p>
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
              <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                <Zap className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Tokens/sec</p>
                <p className="text-xl font-bold text-white">{llmData.ollama.tokensPerSecond}</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Model Configuration & Evaluation */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Model Configuration</h3>
              <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center">
                <Settings className="h-5 w-5 text-blue-400" />
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Model Version</span>
                <span className="text-white font-semibold">{llmData.ollama.version}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Temperature</span>
                <span className="text-white font-semibold">{llmData.ollama.temperature}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Max Tokens</span>
                <span className="text-white font-semibold">{llmData.ollama.maxTokens}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Memory Usage</span>
                <span className="text-white font-semibold">{llmData.ollama.memoryUsage}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">GPU Usage</span>
                <span className="text-white font-semibold">{llmData.ollama.gpuUsage}</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">DeepEval Metrics</h3>
              <div className="w-10 h-10 bg-green-500/20 rounded-xl flex items-center justify-center">
                <TestTube className="h-5 w-5 text-green-400" />
              </div>
            </div>
            <div className="space-y-4">
              {evaluationData.map((metric, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-white/70">{metric.metric}</span>
                    <span className={`font-semibold ${getStatusColor(metric.status)}`}>
                      {metric.score}{metric.metric === 'Bias' || metric.metric === 'Toxicity' ? '%' : ''}
                    </span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        metric.status === 'passed' 
                          ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                          : 'bg-gradient-to-r from-yellow-500 to-orange-600'
                      }`}
                      style={{ width: `${Math.min(metric.score, 100)}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Performance Stats</h3>
              <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-purple-400" />
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Total Requests</span>
                <span className="text-white font-semibold">{llmData.performance.totalRequests.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Success Rate</span>
                <span className="text-green-400 font-semibold">
                  {((llmData.performance.successfulRequests / llmData.performance.totalRequests) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Avg Response</span>
                <span className="text-blue-400 font-semibold">{llmData.performance.avgResponseTime}ms</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Throughput</span>
                <span className="text-purple-400 font-semibold">{llmData.performance.throughput} req/s</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <span className="text-white/70">Error Rate</span>
                <span className="text-red-400 font-semibold">{llmData.performance.errorRate}%</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Test Interface */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="chart-container mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">Test LLM Response</h3>
            <div className="w-10 h-10 bg-yellow-500/20 rounded-xl flex items-center justify-center">
              <MessageCircle className="h-5 w-5 text-yellow-400" />
            </div>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-white/70 mb-2">Query</label>
              <textarea
                value={testQuery}
                onChange={(e) => setTestQuery(e.target.value)}
                placeholder="Enter your test query here..."
                className="w-full p-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                rows={3}
              />
            </div>
            <div className="flex justify-end">
              <button
                onClick={handleTestQuery}
                disabled={isTesting || !testQuery.trim()}
                className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isTesting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Testing...</span>
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    <span>Test Response</span>
                  </>
                )}
              </button>
            </div>
            {testResponse && (
              <div className="mt-4">
                <label className="block text-sm font-medium text-white/70 mb-2">Response</label>
                <div className="p-4 bg-white/5 border border-white/10 rounded-lg text-white/90 whitespace-pre-wrap">
                  {testResponse}
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* Recent Tests & Conversations */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
            className="data-table"
          >
            <div className="px-6 py-4 border-b border-white/10">
              <h3 className="text-xl font-semibold text-white">Recent Tests</h3>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {llmData.tests.map((test) => (
                  <div key={test.id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(test.status)}`}></div>
                      <div>
                        <p className="font-medium text-white">{test.name}</p>
                        <p className="text-sm text-white/60">{test.time}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold text-white">{test.score}%</p>
                      <p className={`text-sm ${getStatusColor(test.status)}`}>
                        {test.status.charAt(0).toUpperCase() + test.status.slice(1)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
            className="data-table"
          >
            <div className="px-6 py-4 border-b border-white/10">
              <h3 className="text-xl font-semibold text-white">Recent Conversations</h3>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {llmData.conversations.map((conv) => (
                  <div key={conv.id} className="p-4 bg-white/5 rounded-lg">
                    <div className="flex items-start justify-between mb-2">
                      <p className="font-medium text-white">{conv.query}</p>
                      <span className="text-sm text-white/60">{conv.timestamp}</span>
                    </div>
                    <p className="text-sm text-white/70 line-clamp-2">{conv.response}</p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
} 