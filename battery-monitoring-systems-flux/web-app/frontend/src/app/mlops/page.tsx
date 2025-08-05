'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { Line, Bar } from 'react-chartjs-2'
import {
  Activity,
  TrendingUp,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  BarChart3,
  Brain,
  CheckCircle,
  Clock,
  Database,
  GitBranch,
  GitCompare,
  Monitor,
  Pause,
  Play,
  RotateCcw,
  Server,
  Settings,
  User,
  Zap
} from 'lucide-react'
import Link from 'next/link'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

// Comprehensive ML Monitoring Data
const operationalMetrics = {
  // Infrastructure & System Health
  cpuUtilization: 67.3,
  memoryUsage: 82.1,
  gpuUtilization: 45.8,
  diskUsage: 73.4,
  networkLatency: 12.5,
  
  // Pipeline Operations
  totalRuns: 47,
  successRate: 94.2,
  avgExecutionTime: 127,
  failedRuns: 3,
  pendingRuns: 2,
  
  // Model Deployment
  modelsDeployed: 12,
  activePipelines: 3,
  deploymentHealth: 98.5,
  lastDeployment: '2024-01-15 10:30:00',
  
  // API & Service Health
  apiRequests: 2847,
  avgResponseTime: 1250,
  errorRate: 0.8,
  uptime: 99.97,
  lastRequest: '30 seconds ago'
}

const driftMetrics = {
  // Data Drift Detection
  featureDrift: {
    voltage: { score: 0.23, status: 'low', trend: 'stable' },
    temperature: { score: 0.45, status: 'medium', trend: 'increasing' },
    current: { score: 0.12, status: 'low', trend: 'stable' },
    capacity: { score: 0.67, status: 'high', trend: 'increasing' }
  },
  
  // Model Drift
  modelDrift: {
    anomalyDetection: { score: 0.18, status: 'low', lastCheck: '2 hours ago' },
    cellHealthPrediction: { score: 0.34, status: 'medium', lastCheck: '1 hour ago' },
    forecastingModel: { score: 0.52, status: 'high', lastCheck: '30 minutes ago' }
  },
  
  // Concept Drift
  conceptDrift: {
    overall: { score: 0.29, status: 'low', trend: 'stable' },
    seasonal: { score: 0.41, status: 'medium', trend: 'increasing' },
    gradual: { score: 0.15, status: 'low', trend: 'stable' }
  },
  
  // Data Quality Metrics
  dataQuality: {
    completeness: 98.7,
    accuracy: 96.3,
    consistency: 97.8,
    timeliness: 99.1,
    validity: 95.9
  }
}

const performanceMetrics = {
  // Model Performance Over Time
  accuracyTrend: [
    { date: '2024-01-10', accuracy: 0.947, precision: 0.923, recall: 0.981 },
    { date: '2024-01-11', accuracy: 0.945, precision: 0.921, recall: 0.979 },
    { date: '2024-01-12', accuracy: 0.943, precision: 0.919, recall: 0.977 },
    { date: '2024-01-13', accuracy: 0.941, precision: 0.917, recall: 0.975 },
    { date: '2024-01-14', accuracy: 0.939, precision: 0.915, recall: 0.973 },
    { date: '2024-01-15', accuracy: 0.937, precision: 0.913, recall: 0.971 }
  ],
  
  // Real-time Performance
  realTimeMetrics: {
    predictionsPerSecond: 156,
    avgPredictionLatency: 45,
    throughput: 89.2,
    errorRate: 0.3,
    confidenceScore: 0.87
  },
  
  // Business Metrics
  businessMetrics: {
    costPerPrediction: 0.0023,
    revenueImpact: 12.5,
    userSatisfaction: 4.7,
    modelROI: 156.8,
    timeToValue: 2.3
  }
}

// Resource Monitoring Data (24-hour timeline)
const resourceMonitoringData = {
  // Generate 24 hours of data with realistic patterns
  hours: Array.from({ length: 24 }, (_, i) => i),
  
  // GPU Usage - similar to the chart pattern
  gpuUsage: [
    35, 45, 75, 68, 62, 58, 52, 48, 55, 60, 42, 45, 63, 58, 52, 48, 65, 72, 78, 82, 88, 88, 85, 82
  ],
  
  // CPU Usage - similar to the chart pattern
  cpuUsage: [
    50, 55, 62, 68, 65, 68, 58, 45, 48, 52, 55, 58, 62, 65, 68, 72, 75, 72, 68, 65, 58, 52, 48, 45
  ],
  
  // Memory Usage - similar to the chart pattern
  memoryUsage: [
    45, 48, 52, 58, 60, 62, 58, 55, 58, 62, 65, 68, 72, 75, 78, 75, 72, 68, 78, 75, 72, 68, 45, 40
  ],
  
  // System Uptime (cumulative)
  systemUptime: [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
  ]
}

// Feature Distribution Data for Drift Monitoring
const featureDistributionData = {
  // Feature values (0-10 range)
  featureValues: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  
  // Train data distribution (centered around 4.0, bell-shaped)
  trainData: [50, 120, 280, 450, 620, 580, 420, 280, 150, 80, 30],
  
  // New data distribution (centered around 6.5, bell-shaped, shifted right)
  newData: [20, 60, 150, 280, 420, 580, 720, 680, 450, 280, 120],
  
  // Battery-specific features for drift monitoring
  batteryFeatures: {
    voltage: {
      trainData: [30, 80, 180, 350, 520, 480, 320, 180, 90, 40, 15],
      newData: [15, 50, 120, 250, 380, 520, 650, 580, 350, 200, 80]
    },
    temperature: {
      trainData: [40, 100, 220, 380, 550, 520, 380, 220, 110, 50, 20],
      newData: [25, 70, 160, 300, 420, 580, 700, 620, 400, 220, 100]
    },
    current: {
      trainData: [60, 140, 300, 450, 600, 550, 400, 250, 130, 60, 25],
      newData: [35, 90, 200, 320, 450, 600, 680, 600, 380, 200, 90]
    }
  }
}

// Proxy Metrics and Ground Truth Monitoring Data
const proxyMetricsData = {
  // 30 days of data
  days: Array.from({ length: 31 }, (_, i) => i),
  
  // Model Confidence Over Time (0.0 to 1.0)
  modelConfidence: [
    0.89, 0.87, 0.88, 0.91, 0.89, 0.87, 0.99, 0.88, 0.87, 1.0, 0.88, 0.91, 0.89, 0.93, 0.87, 0.89, 1.0, 0.88, 0.87, 0.95, 0.73, 0.71, 0.73, 0.75, 0.77, 0.74, 0.65, 0.67, 0.68, 0.66, 0.58
  ],
  
  // Risk Threshold (constant at 0.5)
  riskThreshold: Array.from({ length: 31 }, () => 0.5),
  
  // Positive Prediction Rate (0.0 to 0.07)
  positiveRate: [
    0.022, 0.026, 0.025, 0.024, 0.006, 0.008, 0.015, 0.018, 0.032, 0.011, 0.014, 0.019, 0.025, 0.021, 0.028, 0.024, 0.031, 0.036, 0.029, 0.025, 0.022, 0.018, 0.013, 0.016, 0.019, 0.058, 0.062, 0.065, 0.068, 0.071, 0.073
  ],
  
  // Expected Max threshold (constant at 0.05)
  expectedMax: Array.from({ length: 31 }, () => 0.05),
  
  // Regression Output Monitoring (400 to 1000)
  predictedValue: [
    520, 480, 510, 490, 530, 470, 450, 350, 420, 480, 520, 490, 510, 480, 520, 490, 510, 480, 600, 580, 560, 540, 520, 500, 420, 500, 700, 740, 870, 920, 1000
  ],
  
  // Suspicious Threshold (constant at 1000)
  suspiciousThreshold: Array.from({ length: 31 }, () => 1000),
  
  // Ground Truth Accuracy (Backtesting) (0.76 to 0.90)
  backtestedAccuracy: [
    0.875, 0.882, 0.878, 0.885, 0.879, 0.883, 0.881, 0.887, 0.884, 0.905, 0.898, 0.892, 0.889, 0.895, 0.891, 0.84, 0.905, 0.896, 0.838, 0.887, 0.892, 0.885, 0.879, 0.882, 0.878, 0.875, 0.798, 0.802, 0.805, 0.795, 0.79
  ],
  
  // Min Acceptable threshold (constant at 0.756)
  minAcceptable: Array.from({ length: 31 }, () => 0.756)
}

// Alerting and Response Monitoring Data
const alertingData = {
  // 30 days of data
  days: Array.from({ length: 31 }, (_, i) => i),
  
  // Model Accuracy with Thresholds (0.78 to 0.88)
  modelAccuracy: [
    0.865, 0.862, 0.868, 0.88, 0.875, 0.85, 0.87, 0.865, 0.872, 0.88, 0.875, 0.868, 0.872, 0.875, 0.87, 0.865, 0.868, 0.872, 0.875, 0.87, 0.865, 0.868, 0.872, 0.875, 0.87, 0.865, 0.868, 0.872, 0.875, 0.77, 0.775
  ],
  
  // Min Acceptable threshold (constant at 0.80)
  minAcceptable: Array.from({ length: 31 }, () => 0.80),
  
  // Alert Count (0 to 11)
  alertCount: [
    1, 3, 2, 2, 2, 3, 1, 5, 1, 2, 1, 8, 9, 11, 4, 4, 2, 2, 2, 1, 1, 2, 2, 2, 1, 5, 6, 4, 4, 4, 4
  ],
  
  // Alert Threshold (constant at 5)
  alertThreshold: Array.from({ length: 31 }, () => 5),
  
  // Performance Metric with Distribution Shift (0.70 to 1.00)
  modelAccuracyWithShift: [
    0.91, 0.915, 0.92, 0.918, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.82, 0.825, 0.83, 0.835, 0.84, 0.845, 0.85, 0.855, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0
  ],
  
  // Agreement with Ground Truth (Doctor) - 60 days
  agreementWithGroundTruth: [
    0.91, 0.915, 0.92, 0.918, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.92, 0.918, 0.922, 0.925, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79, 0.795, 0.8, 0.805, 0.81, 0.815, 0.82, 0.825, 0.83, 0.835, 0.84, 0.845, 0.85, 0.855, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0
  ],
  
  // Distribution Shift Day (Day 30)
  distributionShiftDay: 30,
  
  // Alert Response Policies
  responsePolicies: {
    dataDriftThreshold: 0.5,
    accuracyThreshold: 0.80,
    alertThreshold: 5,
    autoRetrainEnabled: true,
    autoRollbackEnabled: true,
    rollbackThreshold: 0.05, // 5% worse performance
    manualInterventionRequired: false
  },
  
  // Current Alerts Status
  currentAlerts: {
    activeAlerts: 4,
    criticalAlerts: 1,
    dataDriftAlerts: 2,
    performanceAlerts: 1,
    lastAlertTime: '2 minutes ago',
    escalationLevel: 'medium'
  },
  
  // Automated Response Actions
  automatedActions: {
    modelRollbacks: 2,
    autoRetrains: 3,
    engineerNotifications: 8,
    systemScaling: 1,
    lastAction: 'Model rollback initiated - 15 minutes ago'
  }
}

// Shadow Model Monitoring Data
const shadowModelData = {
  // 30 days of data
  days: Array.from({ length: 31 }, (_, i) => i),
  
  // Live Model Accuracy (0.83 to 0.87)
  liveModelAccuracy: [
    0.855, 0.852, 0.848, 0.853, 0.849, 0.845, 0.851, 0.847, 0.854, 0.850, 0.846, 0.852, 0.83, 0.83, 0.835, 0.87, 0.865, 0.86, 0.855, 0.85, 0.87, 0.865, 0.86, 0.855, 0.85, 0.845, 0.84, 0.835, 0.83, 0.825, 0.83
  ],
  
  // Shadow Model Accuracy (0.85 to 0.91)
  shadowModelAccuracy: [
    0.86, 0.863, 0.867, 0.864, 0.868, 0.865, 0.869, 0.866, 0.87, 0.867, 0.871, 0.868, 0.85, 0.85, 0.855, 0.89, 0.885, 0.88, 0.875, 0.87, 0.89, 0.885, 0.88, 0.875, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.88
  ],
  
  // Model Comparison Metrics
  comparisonMetrics: {
    avgLiveAccuracy: 0.847,
    avgShadowAccuracy: 0.873,
    accuracyGap: 0.026,
    shadowAdvantage: '+2.6%',
    consistencyScore: 0.92,
    driftDetected: false,
    recommendation: 'Shadow model shows consistent improvement',
    deploymentReadiness: 'Ready for A/B testing'
  },
  
  // Prediction Discrepancy Analysis
  predictionDiscrepancies: {
    totalPredictions: 15420,
    matchingPredictions: 14280,
    differentPredictions: 1140,
    discrepancyRate: 7.4,
    highConfidenceMatches: 89.2,
    lowConfidenceMatches: 10.8,
    criticalDiscrepancies: 23
  }
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
        {/* Beautiful Horizontal Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="glass-card p-6 mb-8"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-12">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                  <GitBranch className="h-4 w-4 text-white" />
                </div>
                <div>
                  <p className="text-xs font-medium text-white/60 uppercase tracking-wider">Total Requests</p>
                  <p className="text-xl font-bold text-white">2,847</p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-green-600 rounded-lg flex items-center justify-center">
                  <CheckCircle className="h-4 w-4 text-white" />
                </div>
                <div>
                  <p className="text-xs font-medium text-white/60 uppercase tracking-wider">Avg Quality</p>
                  <p className="text-xl font-bold text-green-400">89.0%</p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <Clock className="h-4 w-4 text-white" />
                </div>
                <div>
                  <p className="text-xs font-medium text-white/60 uppercase tracking-wider">Avg Response</p>
                  <p className="text-xl font-bold text-purple-400">1250ms</p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-yellow-500 to-yellow-600 rounded-lg flex items-center justify-center">
                  <Database className="h-4 w-4 text-white" />
                </div>
                <div>
                  <p className="text-xs font-medium text-white/60 uppercase tracking-wider">Total Cost</p>
                  <p className="text-xl font-bold text-yellow-400">$12.47</p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-indigo-600 rounded-lg flex items-center justify-center">
                  <Activity className="h-4 w-4 text-white" />
                </div>
                <div>
                  <p className="text-xs font-medium text-white/60 uppercase tracking-wider">A/B Tests</p>
                  <p className="text-xl font-bold text-indigo-400">2</p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-pink-500 to-pink-600 rounded-lg flex items-center justify-center">
                <Clock className="h-4 w-4 text-white" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/60 uppercase tracking-wider">Last Request</p>
                <p className="text-sm font-medium text-white">30 seconds ago</p>
              </div>
            </div>
          </div>
        </motion.div>
          

        {/* Comprehensive ML Monitoring Sections */}
        
        {/* 1. Operational Monitoring */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="chart-container mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">🖥️ Operational Monitoring</h3>
            <div className="w-10 h-10 bg-blue-500/20 rounded-xl flex items-center justify-center">
              <Activity className="h-5 w-5 text-blue-400" />
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Infrastructure Health */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Infrastructure</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">CPU Usage</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.cpuUtilization}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Memory Usage</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.memoryUsage}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">GPU Usage</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.gpuUtilization}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Disk Usage</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.diskUsage}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Network Latency</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.networkLatency}ms</span>
                </div>
              </div>
            </div>

            {/* Pipeline Operations */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Pipeline Operations</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Total Runs</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.totalRuns}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Success Rate</span>
                  <span className="text-sm font-semibold text-green-400">{operationalMetrics.successRate}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Failed Runs</span>
                  <span className="text-sm font-semibold text-red-400">{operationalMetrics.failedRuns}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Avg Execution</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.avgExecutionTime}s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Pending Runs</span>
                  <span className="text-sm font-semibold text-yellow-400">{operationalMetrics.pendingRuns}</span>
                </div>
              </div>
            </div>

            {/* Model Deployment */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Model Deployment</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Models Deployed</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.modelsDeployed}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Active Pipelines</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.activePipelines}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Deployment Health</span>
                  <span className="text-sm font-semibold text-green-400">{operationalMetrics.deploymentHealth}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Uptime</span>
                  <span className="text-sm font-semibold text-green-400">{operationalMetrics.uptime}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Last Deployment</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.lastDeployment}</span>
                </div>
              </div>
            </div>

            {/* API & Service Health */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">API & Service</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">API Requests</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.apiRequests.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Avg Response</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.avgResponseTime}ms</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Error Rate</span>
                  <span className="text-sm font-semibold text-red-400">{operationalMetrics.errorRate}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Last Request</span>
                  <span className="text-sm font-semibold text-white">{operationalMetrics.lastRequest}</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Resource Monitoring Charts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.75 }}
          className="chart-container mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">📈 Smoothed Resource Monitoring</h3>
            <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
              <Monitor className="h-5 w-5 text-purple-400" />
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* GPU Usage Chart */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">GPU Usage Over Time</h4>
              <div className="h-56">
                <Line 
                  data={{
                    labels: resourceMonitoringData.hours.map(h => `${h}:00`),
                    datasets: [{
                      label: 'GPU Usage (%)',
                      data: resourceMonitoringData.gpuUsage,
                      borderColor: '#f97316',
                      backgroundColor: 'rgba(249, 115, 22, 0.1)',
                      borderWidth: 2,
                      fill: true,
                      tension: 0.4,
                      pointBackgroundColor: '#f97316',
                      pointBorderColor: '#ffffff',
                      pointBorderWidth: 2,
                      pointRadius: 3,
                      pointHoverRadius: 5,
                    }]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#f97316',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Hour',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: '% Usage',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return `${value}%`
                          }
                        },
                        min: 0,
                        max: 100
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* CPU Usage Chart */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">CPU Usage Over Time</h4>
              <div className="h-56">
                <Line 
                  data={{
                    labels: resourceMonitoringData.hours.map(h => `${h}:00`),
                    datasets: [{
                      label: 'CPU Usage (%)',
                      data: resourceMonitoringData.cpuUsage,
                      borderColor: '#f97316',
                      backgroundColor: 'rgba(249, 115, 22, 0.1)',
                      borderWidth: 2,
                      fill: true,
                      tension: 0.4,
                      pointBackgroundColor: '#f97316',
                      pointBorderColor: '#ffffff',
                      pointBorderWidth: 2,
                      pointRadius: 3,
                      pointHoverRadius: 5,
                    }]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#f97316',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Hour',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: '% Usage',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return `${value}%`
                          }
                        },
                        min: 0,
                        max: 100
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* Memory Usage Chart */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Memory Usage Over Time</h4>
              <div className="h-56">
                <Line 
                  data={{
                    labels: resourceMonitoringData.hours.map(h => `${h}:00`),
                    datasets: [{
                      label: 'Memory Usage (%)',
                      data: resourceMonitoringData.memoryUsage,
                      borderColor: '#22c55e',
                      backgroundColor: 'rgba(34, 197, 94, 0.1)',
                      borderWidth: 2,
                      fill: true,
                      tension: 0.4,
                      pointBackgroundColor: '#22c55e',
                      pointBorderColor: '#ffffff',
                      pointBorderWidth: 2,
                      pointRadius: 3,
                      pointHoverRadius: 5,
                    }]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#22c55e',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Hour',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: '% Usage',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return `${value}%`
                          }
                        },
                        min: 0,
                        max: 100
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* System Uptime Chart */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">System Uptime</h4>
              <div className="h-56">
                <Line 
                  data={{
                    labels: resourceMonitoringData.hours.map(h => `${h}:00`),
                    datasets: [{
                      label: 'Cumulative Uptime',
                      data: resourceMonitoringData.systemUptime,
                      borderColor: '#a855f7',
                      backgroundColor: 'rgba(168, 85, 247, 0.1)',
                      borderWidth: 2,
                      fill: true,
                      tension: 0.1,
                      pointBackgroundColor: '#a855f7',
                      pointBorderColor: '#ffffff',
                      pointBorderWidth: 2,
                      pointRadius: 3,
                      pointHoverRadius: 5,
                    }]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#a855f7',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {
                          label: function(context: any) {
                            return `Uptime: ${context.parsed.y} hours`
                          }
                        }
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Hour',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Cumulative Uptime',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return `${value}h`
                          }
                        },
                        min: 0,
                        max: 24
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>
          </div>
        </motion.div>

        {/* 2. Drift Monitoring */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="chart-container mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">📊 Drift Monitoring</h3>
            <div className="w-10 h-10 bg-orange-500/20 rounded-xl flex items-center justify-center">
              <TrendingUp className="h-5 w-5 text-orange-400" />
            </div>
          </div>
          
          {/* Feature Distribution Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* General Feature Distribution */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Feature Distribution: Train vs New Data</h4>
              <div className="h-64">
                <Bar 
                  data={{
                    labels: featureDistributionData.featureValues.map(v => v.toString()),
                    datasets: [
                      {
                        label: 'Train Data',
                        data: featureDistributionData.trainData,
                        backgroundColor: 'rgba(59, 130, 246, 0.8)',
                        borderColor: '#3b82f6',
                        borderWidth: 1,
                        borderRadius: 4,
                      },
                      {
                        label: 'New Data',
                        data: featureDistributionData.newData,
                        backgroundColor: 'rgba(249, 115, 22, 0.8)',
                        borderColor: '#f97316',
                        borderWidth: 1,
                        borderRadius: 4,
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top' as const,
                        labels: {
                          color: 'rgba(255, 255, 255, 0.8)',
                          font: { size: 12 },
                          usePointStyle: true,
                        }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#f97316',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Feature Value',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Count',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 }
                        }
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* Battery Voltage Distribution */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Battery Voltage Distribution</h4>
              <div className="h-64">
                <Bar 
                  data={{
                    labels: featureDistributionData.featureValues.map(v => v.toString()),
                    datasets: [
                      {
                        label: 'Train Data',
                        data: featureDistributionData.batteryFeatures.voltage.trainData,
                        backgroundColor: 'rgba(34, 197, 94, 0.8)',
                        borderColor: '#22c55e',
                        borderWidth: 1,
                        borderRadius: 4,
                      },
                      {
                        label: 'New Data',
                        data: featureDistributionData.batteryFeatures.voltage.newData,
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        borderColor: '#ef4444',
                        borderWidth: 1,
                        borderRadius: 4,
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top' as const,
                        labels: {
                          color: 'rgba(255, 255, 255, 0.8)',
                          font: { size: 12 },
                          usePointStyle: true,
                        }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#ef4444',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Voltage Level',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Count',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 }
                        }
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>
          </div>

          {/* Drift Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Feature Drift */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Feature Drift</h4>
              <div className="space-y-3">
                {Object.entries(driftMetrics.featureDrift).map(([feature, data]) => (
                  <div key={feature} className="flex justify-between items-center">
                    <span className="text-sm text-white/70 capitalize">{feature}</span>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        data.status === 'low' ? 'bg-green-500/20 text-green-400' :
                        data.status === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {data.score.toFixed(2)}
                      </span>
                      <span className={`text-xs ${
                        data.trend === 'increasing' ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {data.trend === 'increasing' ? '↗' : '→'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Drift */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Model Drift</h4>
              <div className="space-y-3">
                {Object.entries(driftMetrics.modelDrift).map(([model, data]) => (
                  <div key={model} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-white/70 capitalize">{model.replace(/([A-Z])/g, ' $1')}</span>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        data.status === 'low' ? 'bg-green-500/20 text-green-400' :
                        data.status === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {data.score.toFixed(2)}
                      </span>
                    </div>
                    <div className="text-xs text-white/50">{data.lastCheck}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Concept Drift */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Concept Drift</h4>
              <div className="space-y-3">
                {Object.entries(driftMetrics.conceptDrift).map(([type, data]) => (
                  <div key={type} className="flex justify-between items-center">
                    <span className="text-sm text-white/70 capitalize">{type}</span>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        data.status === 'low' ? 'bg-green-500/20 text-green-400' :
                        data.status === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {data.score.toFixed(2)}
                      </span>
                      <span className={`text-xs ${
                        data.trend === 'increasing' ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {data.trend === 'increasing' ? '↗' : '→'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Data Quality */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Data Quality</h4>
              <div className="space-y-3">
                {Object.entries(driftMetrics.dataQuality).map(([metric, value]) => (
                  <div key={metric} className="flex justify-between items-center">
                    <span className="text-sm text-white/70 capitalize">{metric}</span>
                    <span className="text-sm font-semibold text-white">{value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* 3. Model Performance Monitoring */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="chart-container mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">🎯 Model Performance Monitoring</h3>
            <div className="w-10 h-10 bg-green-500/20 rounded-xl flex items-center justify-center">
              <BarChart3 className="h-5 w-5 text-green-400" />
            </div>
          </div>
          
          {/* Proxy Metrics Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Model Confidence Over Time */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Model Confidence Over Time</h4>
              <div className="h-64">
                <Line 
                  data={{
                    labels: proxyMetricsData.days.map(d => `Day ${d}`),
                    datasets: [
                      {
                        label: 'Avg Confidence',
                        data: proxyMetricsData.modelConfidence,
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249, 115, 22, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: '#f97316',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                      },
                      {
                        label: 'Risk Threshold',
                        data: proxyMetricsData.riskThreshold,
                        borderColor: '#ef4444',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top' as const,
                        labels: {
                          color: 'rgba(255, 255, 255, 0.8)',
                          font: { size: 12 },
                          usePointStyle: true,
                        }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#f97316',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Day',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Confidence',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return value.toFixed(1)
                          }
                        },
                        min: 0,
                        max: 1
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* Positive Prediction Rate */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Positive Prediction Rate</h4>
              <div className="h-64">
                <Line 
                  data={{
                    labels: proxyMetricsData.days.map(d => `Day ${d}`),
                    datasets: [
                      {
                        label: 'Positive Rate',
                        data: proxyMetricsData.positiveRate,
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249, 115, 22, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: '#f97316',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                      },
                      {
                        label: 'Expected Max',
                        data: proxyMetricsData.expectedMax,
                        borderColor: '#ef4444',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top' as const,
                        labels: {
                          color: 'rgba(255, 255, 255, 0.8)',
                          font: { size: 12 },
                          usePointStyle: true,
                        }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#f97316',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Day',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Rate',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return value.toFixed(3)
                          }
                        },
                        min: 0,
                        max: 0.08
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* Regression Output Monitoring */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Regression Output Monitoring</h4>
              <div className="h-64">
                <Line 
                  data={{
                    labels: proxyMetricsData.days.map(d => `Day ${d}`),
                    datasets: [
                      {
                        label: 'Predicted Value',
                        data: proxyMetricsData.predictedValue,
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: '#22c55e',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                      },
                      {
                        label: 'Suspicious Threshold',
                        data: proxyMetricsData.suspiciousThreshold,
                        borderColor: '#ef4444',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top' as const,
                        labels: {
                          color: 'rgba(255, 255, 255, 0.8)',
                          font: { size: 12 },
                          usePointStyle: true,
                        }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#22c55e',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Day',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Output Value',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 }
                        },
                        min: 300,
                        max: 1100
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* Ground Truth Accuracy */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Ground Truth Accuracy (Backtesting)</h4>
              <div className="h-64">
                <Line 
                  data={{
                    labels: proxyMetricsData.days.map(d => `Day ${d}`),
                    datasets: [
                      {
                        label: 'Backtested Accuracy',
                        data: proxyMetricsData.backtestedAccuracy,
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: '#a855f7',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                      },
                      {
                        label: 'Min Acceptable',
                        data: proxyMetricsData.minAcceptable,
                        borderColor: '#ef4444',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top' as const,
                        labels: {
                          color: 'rgba(255, 255, 255, 0.8)',
                          font: { size: 12 },
                          usePointStyle: true,
                        }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#a855f7',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Day',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Accuracy',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return value.toFixed(2)
                          }
                        },
                        min: 0.75,
                        max: 0.92
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>
          </div>

          {/* Performance Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Real-time Performance */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Real-time Performance</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Predictions/sec</span>
                  <span className="text-sm font-semibold text-white">{performanceMetrics.realTimeMetrics.predictionsPerSecond}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Avg Latency</span>
                  <span className="text-sm font-semibold text-white">{performanceMetrics.realTimeMetrics.avgPredictionLatency}ms</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Throughput</span>
                  <span className="text-sm font-semibold text-white">{performanceMetrics.realTimeMetrics.throughput}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Error Rate</span>
                  <span className="text-sm font-semibold text-red-400">{performanceMetrics.realTimeMetrics.errorRate}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Confidence Score</span>
                  <span className="text-sm font-semibold text-green-400">{(performanceMetrics.realTimeMetrics.confidenceScore * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Business Metrics */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Business Metrics</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Cost/Prediction</span>
                  <span className="text-sm font-semibold text-white">${performanceMetrics.businessMetrics.costPerPrediction}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Revenue Impact</span>
                  <span className="text-sm font-semibold text-green-400">${performanceMetrics.businessMetrics.revenueImpact}K</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">User Satisfaction</span>
                  <span className="text-sm font-semibold text-white">{performanceMetrics.businessMetrics.userSatisfaction}/5</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Model ROI</span>
                  <span className="text-sm font-semibold text-green-400">{performanceMetrics.businessMetrics.modelROI}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Time to Value</span>
                  <span className="text-sm font-semibold text-white">{performanceMetrics.businessMetrics.timeToValue} days</span>
                </div>
              </div>
            </div>

            {/* Performance Trend */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Performance Trend (6 days)</h4>
              <div className="space-y-3">
                {performanceMetrics.accuracyTrend.slice(-3).map((day, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-white/70">{day.date}</span>
                      <span className="text-sm font-semibold text-white">{(day.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between items-center text-xs text-white/50">
                      <span>P: {(day.precision * 100).toFixed(1)}%</span>
                      <span>R: {(day.recall * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* 4. Shadow Model Monitoring */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
          className="chart-container mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">👥 Shadow Model Monitoring</h3>
            <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
              <GitCompare className="h-5 w-5 text-purple-400" />
            </div>
          </div>
          
          {/* Shadow Model vs Live Model Accuracy Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Main Accuracy Comparison Chart */}
            <div className="lg:col-span-2 space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Shadow Model vs Live Model Accuracy</h4>
              <div className="h-80">
                <Line 
                  data={{
                    labels: shadowModelData.days.map(d => `Day ${d}`),
                    datasets: [
                      {
                        label: 'Live Model',
                        data: shadowModelData.liveModelAccuracy,
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249, 115, 22, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: '#f97316',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                      },
                      {
                        label: 'Shadow Model',
                        data: shadowModelData.shadowModelAccuracy,
                        borderColor: '#dc2626',
                        backgroundColor: 'rgba(220, 38, 38, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: '#dc2626',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top' as const,
                        labels: {
                          color: 'rgba(255, 255, 255, 0.8)',
                          font: { size: 12 },
                          usePointStyle: true,
                        }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#dc2626',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Day',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Accuracy',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return value.toFixed(2)
                          }
                        },
                        min: 0.82,
                        max: 0.92
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* Shadow Model Metrics */}
            <div className="space-y-6">
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Model Comparison</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Avg Live Accuracy</span>
                    <span className="text-sm font-semibold text-white">{(shadowModelData.comparisonMetrics.avgLiveAccuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Avg Shadow Accuracy</span>
                    <span className="text-sm font-semibold text-green-400">{(shadowModelData.comparisonMetrics.avgShadowAccuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Accuracy Gap</span>
                    <span className="text-sm font-semibold text-green-400">{shadowModelData.comparisonMetrics.shadowAdvantage}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Consistency Score</span>
                    <span className="text-sm font-semibold text-blue-400">{(shadowModelData.comparisonMetrics.consistencyScore * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Drift Detected</span>
                    <span className={`text-sm font-semibold ${shadowModelData.comparisonMetrics.driftDetected ? 'text-red-400' : 'text-green-400'}`}>
                      {shadowModelData.comparisonMetrics.driftDetected ? 'Yes' : 'No'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Prediction Analysis</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Total Predictions</span>
                    <span className="text-sm font-semibold text-white">{shadowModelData.predictionDiscrepancies.totalPredictions.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Matching Predictions</span>
                    <span className="text-sm font-semibold text-green-400">{shadowModelData.predictionDiscrepancies.matchingPredictions.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Different Predictions</span>
                    <span className="text-sm font-semibold text-orange-400">{shadowModelData.predictionDiscrepancies.differentPredictions.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Discrepancy Rate</span>
                    <span className="text-sm font-semibold text-orange-400">{shadowModelData.predictionDiscrepancies.discrepancyRate}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-white/70">Critical Discrepancies</span>
                    <span className="text-sm font-semibold text-red-400">{shadowModelData.predictionDiscrepancies.criticalDiscrepancies}</span>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Recommendation</h4>
                <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <p className="text-sm text-green-400 font-medium mb-2">
                    {shadowModelData.comparisonMetrics.recommendation}
                  </p>
                  <p className="text-xs text-white/70">
                    Status: <span className="text-green-400 font-medium">{shadowModelData.comparisonMetrics.deploymentReadiness}</span>
                  </p>
                </div>
              </div>
            </div>
          </div>


        </motion.div>

        {/* 5. Alerting and Response Monitoring */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.1 }}
          className="chart-container mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">🚨 Alerting and Response Monitoring</h3>
            <div className="w-10 h-10 bg-red-500/20 rounded-xl flex items-center justify-center">
              <AlertTriangle className="h-5 w-5 text-red-400" />
            </div>
          </div>
          
          {/* Alert Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Model Accuracy with Thresholds */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Model Accuracy with Thresholds</h4>
              <div className="h-80">
                <Line 
                  data={{
                    labels: alertingData.days.map(d => `Day ${d}`),
                    datasets: [
                      {
                        label: 'Model Accuracy',
                        data: alertingData.modelAccuracy,
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249, 115, 22, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointBackgroundColor: '#f97316',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                      },
                      {
                        label: 'Min Acceptable',
                        data: alertingData.minAcceptable,
                        borderColor: '#ef4444',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: true,
                        position: 'top' as const,
                        labels: {
                          color: 'rgba(255, 255, 255, 0.8)',
                          font: { size: 12 },
                          usePointStyle: true,
                        }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#f97316',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                      },
                      annotation: {
                        annotations: {
                          rollbackLine: {
                            type: 'line',
                            xMin: 30,
                            xMax: 30,
                            borderColor: '#ef4444',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: {
                              content: 'Rollback?',
                              position: 'bottom',
                              color: '#ef4444',
                              font: { size: 12, weight: 'bold' }
                            }
                          }
                        }
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Day',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Accuracy',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 },
                          callback: function(value: any) {
                            return value.toFixed(2)
                          }
                        },
                        min: 0.75,
                        max: 0.9
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
              </div>
            </div>

            {/* Alert Count */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Alert Count</h4>
              <div className="h-80">
                <Bar 
                  data={{
                    labels: alertingData.days.map(d => `Day ${d}`),
                    datasets: [{
                      label: 'Alerts',
                      data: alertingData.alertCount,
                      backgroundColor: 'rgba(59, 130, 246, 0.8)',
                      borderColor: '#3b82f6',
                      borderWidth: 1,
                      borderRadius: 4,
                    }]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#3b82f6',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                      }
                    },
                    scales: {
                      x: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Day',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                      },
                      y: {
                        display: true,
                        title: {
                          display: true,
                          text: 'Alert Count',
                          color: 'rgba(255, 255, 255, 0.7)',
                          font: { size: 12, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                        ticks: { 
                          color: 'rgba(255, 255, 255, 0.8)', 
                          font: { size: 10 }
                        },
                        min: 0,
                        max: 12
                      }
                    },
                    interaction: { intersect: false, mode: 'index' as const }
                  }}
                />
                {/* Alert Threshold Line */}
                <div className="relative -mt-80 h-80 pointer-events-none">
                  <div className="absolute top-0 left-0 right-0 h-px bg-orange-500 border-dashed border-orange-500" 
                       style={{ top: `${(1 - 5/12) * 100}%` }}>
                  </div>
                  <div className="absolute text-orange-400 text-xs font-medium" 
                       style={{ top: `${(1 - 5/12) * 100 - 2}%`, right: '10px' }}>
                    Alert Threshold (5)
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Performance Metric with Distribution Shift */}
          <div className="space-y-4 mb-8">
            <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Performance Metric with Distribution Shift</h4>
            <div className="h-80">
              <Line 
                data={{
                  labels: Array.from({ length: 60 }, (_, i) => `Day ${i}`),
                  datasets: [
                    {
                      label: 'Model Accuracy',
                      data: alertingData.modelAccuracyWithShift,
                      borderColor: '#3b82f6',
                      backgroundColor: 'rgba(59, 130, 246, 0.1)',
                      borderWidth: 2,
                      fill: false,
                      tension: 0.4,
                      pointBackgroundColor: '#3b82f6',
                      pointBorderColor: '#ffffff',
                      pointBorderWidth: 2,
                      pointRadius: 3,
                      pointHoverRadius: 5,
                    },
                    {
                      label: 'Agreement with Ground Truth (Doctor)',
                      data: alertingData.agreementWithGroundTruth,
                      borderColor: '#ef4444',
                      backgroundColor: 'transparent',
                      borderWidth: 2,
                      borderDash: [5, 5],
                      fill: false,
                      tension: 0.4,
                      pointBackgroundColor: '#ef4444',
                      pointBorderColor: '#ffffff',
                      pointBorderWidth: 2,
                      pointRadius: 3,
                      pointHoverRadius: 5,
                    }
                  ]
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: true,
                      position: 'top' as const,
                      labels: {
                        color: 'rgba(255, 255, 255, 0.8)',
                        font: { size: 12 },
                        usePointStyle: true,
                      }
                    },
                    tooltip: {
                      backgroundColor: 'rgba(0, 0, 0, 0.8)',
                      titleColor: '#ffffff',
                      bodyColor: '#ffffff',
                      borderColor: '#3b82f6',
                      borderWidth: 1,
                      cornerRadius: 8,
                      displayColors: true,
                    }
                  },
                  scales: {
                    x: {
                      display: true,
                      title: {
                        display: true,
                        text: 'Day',
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: { size: 12, weight: 'bold' }
                      },
                      grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                      ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 10 } }
                    },
                    y: {
                      display: true,
                      title: {
                        display: true,
                        text: 'Performance Metric',
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: { size: 12, weight: 'bold' }
                      },
                      grid: { color: 'rgba(255, 255, 255, 0.1)', drawBorder: false },
                      ticks: { 
                        color: 'rgba(255, 255, 255, 0.8)', 
                        font: { size: 10 },
                        callback: function(value: any) {
                          return value.toFixed(2)
                        }
                      },
                      min: 0.7,
                      max: 1.0
                    }
                  },
                  interaction: { intersect: false, mode: 'index' as const }
                }}
              />
              {/* Distribution Shift Annotation */}
              <div className="relative -mt-80 h-80 pointer-events-none">
                <div className="absolute top-0 left-0 right-0 h-full bg-gray-500/10" 
                     style={{ left: `${(30/60) * 100}%` }}>
                </div>
                <div className="absolute top-4 text-white text-sm font-medium" 
                     style={{ left: `${(30/60) * 100 + 2}%` }}>
                  Distribution Shift Detected
                </div>
                <div className="absolute top-0 bottom-0 w-px bg-black border-dashed border-black" 
                     style={{ left: `${(30/60) * 100}%` }}>
                </div>
              </div>
            </div>
          </div>

          {/* Alert Response Policies and Status */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Current Alerts Status */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Current Alerts</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Active Alerts</span>
                  <span className="text-sm font-semibold text-red-400">{alertingData.currentAlerts.activeAlerts}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Critical Alerts</span>
                  <span className="text-sm font-semibold text-red-400">{alertingData.currentAlerts.criticalAlerts}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Data Drift Alerts</span>
                  <span className="text-sm font-semibold text-orange-400">{alertingData.currentAlerts.dataDriftAlerts}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Performance Alerts</span>
                  <span className="text-sm font-semibold text-yellow-400">{alertingData.currentAlerts.performanceAlerts}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Last Alert</span>
                  <span className="text-sm font-semibold text-white">{alertingData.currentAlerts.lastAlertTime}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Escalation Level</span>
                  <span className="text-sm font-semibold text-orange-400">{alertingData.currentAlerts.escalationLevel}</span>
                </div>
              </div>
            </div>

            {/* Response Policies */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Response Policies</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Data Drift Threshold</span>
                  <span className="text-sm font-semibold text-white">{(alertingData.responsePolicies.dataDriftThreshold * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Accuracy Threshold</span>
                  <span className="text-sm font-semibold text-white">{(alertingData.responsePolicies.accuracyThreshold * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Alert Threshold</span>
                  <span className="text-sm font-semibold text-white">{alertingData.responsePolicies.alertThreshold}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Auto Retrain</span>
                  <span className={`text-sm font-semibold ${alertingData.responsePolicies.autoRetrainEnabled ? 'text-green-400' : 'text-red-400'}`}>
                    {alertingData.responsePolicies.autoRetrainEnabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Auto Rollback</span>
                  <span className={`text-sm font-semibold ${alertingData.responsePolicies.autoRollbackEnabled ? 'text-green-400' : 'text-red-400'}`}>
                    {alertingData.responsePolicies.autoRollbackEnabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Rollback Threshold</span>
                  <span className="text-sm font-semibold text-white">{(alertingData.responsePolicies.rollbackThreshold * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            {/* Automated Actions */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Automated Actions</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Model Rollbacks</span>
                  <span className="text-sm font-semibold text-red-400">{alertingData.automatedActions.modelRollbacks}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Auto Retrains</span>
                  <span className="text-sm font-semibold text-green-400">{alertingData.automatedActions.autoRetrains}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">Engineer Notifications</span>
                  <span className="text-sm font-semibold text-blue-400">{alertingData.automatedActions.engineerNotifications}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white/70">System Scaling</span>
                  <span className="text-sm font-semibold text-purple-400">{alertingData.automatedActions.systemScaling}</span>
                </div>
                <div className="pt-2 border-t border-white/10">
                  <p className="text-xs text-white/60">Last Action:</p>
                  <p className="text-xs text-white/80">{alertingData.automatedActions.lastAction}</p>
                </div>
              </div>
            </div>

            {/* Runbook Actions */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-white border-b border-white/20 pb-2">Runbook Actions</h4>
              <div className="space-y-3">
                <button className="w-full btn-primary text-sm">
                  🔄 Trigger Auto Retrain
                </button>
                <button className="w-full btn-secondary text-sm">
                  ⚡ Scale System Resources
                </button>
                <button className="w-full btn-secondary text-sm">
                  🔍 Investigate Data Pipeline
                </button>
                <button className="w-full btn-secondary text-sm">
                  📧 Notify Engineering Team
                </button>
                <button className="w-full btn-secondary text-sm">
                  📊 Generate Alert Report
                </button>
                <button className="w-full btn-secondary text-sm">
                  🔧 Manual Intervention
                </button>
              </div>
            </div>
          </div>
        </motion.div>

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