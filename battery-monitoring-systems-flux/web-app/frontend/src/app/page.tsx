'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Users,
  Battery,
  AlertCircle,
  Shield,
  Clock,
  TrendingUp,
  Zap,
  Thermometer,
  MessageCircle,
  Monitor,
  Brain
} from 'lucide-react'
import Link from 'next/link'

// Sample data for charts
const voltageData = [
  { time: '00:00', voltage: 3.65, cell: 'Cell-01' },
  { time: '02:00', voltage: 3.68, cell: 'Cell-01' },
  { time: '04:00', voltage: 3.62, cell: 'Cell-01' },
  { time: '06:00', voltage: 3.70, cell: 'Cell-01' },
  { time: '08:00', voltage: 3.67, cell: 'Cell-01' },
  { time: '10:00', voltage: 3.64, cell: 'Cell-01' },
  { time: '12:00', voltage: 3.69, cell: 'Cell-01' },
  { time: '14:00', voltage: 3.66, cell: 'Cell-01' },
  { time: '16:00', voltage: 3.63, cell: 'Cell-01' },
  { time: '18:00', voltage: 3.71, cell: 'Cell-01' },
  { time: '20:00', voltage: 3.68, cell: 'Cell-01' },
  { time: '22:00', voltage: 3.65, cell: 'Cell-01' }
]

const temperatureData = [
  { time: '00:00', temp: 25, cell: 'Cell-01' },
  { time: '02:00', temp: 26, cell: 'Cell-01' },
  { time: '04:00', temp: 24, cell: 'Cell-01' },
  { time: '06:00', temp: 27, cell: 'Cell-01' },
  { time: '08:00', temp: 28, cell: 'Cell-01' },
  { time: '10:00', temp: 29, cell: 'Cell-01' },
  { time: '12:00', temp: 31, cell: 'Cell-01' },
  { time: '14:00', temp: 30, cell: 'Cell-01' },
  { time: '16:00', temp: 28, cell: 'Cell-01' },
  { time: '18:00', temp: 26, cell: 'Cell-01' },
  { time: '20:00', temp: 25, cell: 'Cell-01' },
  { time: '22:00', temp: 24, cell: 'Cell-01' }
]

const stats = {
  devices: '12',
  cells: '48',
  alerts: '2',
  status: 'Healthy',
  uptime: '99.8%',
  efficiency: '94.2%'
}

// Chart component with dynamic import
const VoltageChart = () => {
  const [Chart, setChart] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadChart = async () => {
      try {
        const recharts = await import('recharts')
        setChart({
          AreaChart: recharts.AreaChart,
          Area: recharts.Area,
          XAxis: recharts.XAxis,
          YAxis: recharts.YAxis,
          CartesianGrid: recharts.CartesianGrid,
          Tooltip: recharts.Tooltip,
          ResponsiveContainer: recharts.ResponsiveContainer
        })
      } catch (error) {
        console.error('Failed to load chart:', error)
      } finally {
        setIsLoading(false)
      }
    }
    loadChart()
  }, [])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-400"></div>
      </div>
    )
  }

  if (!Chart) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-white/50 text-sm">Chart failed to load</p>
      </div>
    )
  }

  const { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } = Chart

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={voltageData}>
        <defs>
          <linearGradient id="voltageGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ffd700" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#ffd700" stopOpacity={0.1}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff1a" />
        <XAxis 
          dataKey="time" 
          stroke="#ffffff80" 
          fontSize={12}
          tickLine={false}
        />
        <YAxis 
          stroke="#ffffff80" 
          fontSize={12}
          tickLine={false}
          domain={[3.5, 3.8]}
        />
        <Tooltip 
          contentStyle={{
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '8px',
            color: 'white'
          }}
        />
        <Area 
          type="monotone" 
          dataKey="voltage" 
          stroke="#ffd700" 
          strokeWidth={3}
          fill="url(#voltageGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

const TemperatureChart = () => {
  const [Chart, setChart] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadChart = async () => {
      try {
        const recharts = await import('recharts')
        setChart({
          AreaChart: recharts.AreaChart,
          Area: recharts.Area,
          XAxis: recharts.XAxis,
          YAxis: recharts.YAxis,
          CartesianGrid: recharts.CartesianGrid,
          Tooltip: recharts.Tooltip,
          ResponsiveContainer: recharts.ResponsiveContainer
        })
      } catch (error) {
        console.error('Failed to load chart:', error)
      } finally {
        setIsLoading(false)
      }
    }
    loadChart()
  }, [])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-400"></div>
      </div>
    )
  }

  if (!Chart) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-white/50 text-sm">Chart failed to load</p>
      </div>
    )
  }

  const { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } = Chart

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={temperatureData}>
        <defs>
          <linearGradient id="tempGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ff6b6b" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#ff6b6b" stopOpacity={0.1}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff1a" />
        <XAxis 
          dataKey="time" 
          stroke="#ffffff80" 
          fontSize={12}
          tickLine={false}
        />
        <YAxis 
          stroke="#ffffff80" 
          fontSize={12}
          tickLine={false}
          domain={[20, 35]}
        />
        <Tooltip 
          contentStyle={{
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '8px',
            color: 'white'
          }}
        />
        <Area 
          type="monotone" 
          dataKey="temp" 
          stroke="#ff6b6b" 
          strokeWidth={3}
          fill="url(#tempGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export default function Dashboard() {
  const [isConnected, setIsConnected] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="header">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg animate-glow">
                  <Battery className="h-7 w-7 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold gradient-text tracking-tight">Battery Monitoring System</h1>
                <p className="text-sm text-white/70 font-medium">Real-time ML/LLM & MLOps Dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-sm font-medium text-white/80">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <Link href="/mlops">
                  <button className="btn-secondary flex items-center space-x-2 group">
                    <Monitor className="h-5 w-5 group-hover:scale-110 transition-transform" />
                    <span>MLOps</span>
                  </button>
                </Link>
                <Link href="/llm">
                  <button className="btn-secondary flex items-center space-x-2 group">
                    <Brain className="h-5 w-5 group-hover:scale-110 transition-transform" />
                    <span>LLM</span>
                  </button>
                </Link>
                <Link href="/chat">
                  <button className="btn-primary flex items-center space-x-2 group">
                    <MessageCircle className="h-5 w-5 group-hover:scale-110 transition-transform" />
                    <span>Chat with AI</span>
                  </button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Section */}
        <div className="flex flex-wrap items-center gap-4 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <Users className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Total Devices</p>
                <p className="text-xl font-bold text-white">{stats.devices}</p>
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
              <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                <Battery className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Total Cells</p>
                <p className="text-xl font-bold text-white">{stats.cells}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center">
                <AlertCircle className="h-5 w-5 text-red-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Active Alerts</p>
                <p className="text-xl font-bold text-white">{stats.alerts}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                <Shield className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">System Status</p>
                <p className="text-xl font-bold gradient-text-success">{stats.status}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                <Clock className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">System Uptime</p>
                <p className="text-xl font-bold gradient-text-success">{stats.uptime}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="stat-card flex-shrink-0"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Efficiency Score</p>
                <p className="text-xl font-bold gradient-text">{stats.efficiency}</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Voltage Trend</h3>
              <div className="w-10 h-10 bg-yellow-500/20 rounded-xl flex items-center justify-center">
                <Zap className="h-5 w-5 text-yellow-400" />
              </div>
            </div>
            <div className="h-64 bg-white/5 rounded-xl flex items-center justify-center p-4">
              <VoltageChart />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="chart-container"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Temperature Trend</h3>
              <div className="w-10 h-10 bg-red-500/20 rounded-xl flex items-center justify-center">
                <Thermometer className="h-5 w-5 text-red-400" />
              </div>
            </div>
            <div className="h-64 bg-white/5 rounded-xl flex items-center justify-center p-4">
              <TemperatureChart />
            </div>
          </motion.div>
        </div>

        {/* Data Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="data-table"
        >
          <div className="px-6 py-4 border-b border-white/10">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold text-white">Battery Data</h3>
              <div className="flex items-center space-x-4">
                <select className="filter-select">
                  <option>All Devices</option>
                </select>
                <select className="filter-select">
                  <option>All Cells</option>
                </select>
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-white/5">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-white/70 uppercase tracking-wider">Device</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-white/70 uppercase tracking-wider">Cell</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-white/70 uppercase tracking-wider">Voltage</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-white/70 uppercase tracking-wider">Temperature</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-white/70 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-white/70 uppercase tracking-wider">Last Updated</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                <tr className="hover:bg-white/5 transition-colors">
                  <td className="px-6 py-4 text-sm text-white">Device-001</td>
                  <td className="px-6 py-4 text-sm text-white">Cell-01</td>
                  <td className="px-6 py-4 text-sm text-white">3.65V</td>
                  <td className="px-6 py-4 text-sm text-white">25°C</td>
                  <td className="px-6 py-4">
                    <span className="status-badge status-healthy">
                      Healthy
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-white/60">2 min ago</td>
                </tr>
                <tr className="hover:bg-white/5 transition-colors">
                  <td className="px-6 py-4 text-sm text-white">Device-001</td>
                  <td className="px-6 py-4 text-sm text-white">Cell-02</td>
                  <td className="px-6 py-4 text-sm text-white">3.62V</td>
                  <td className="px-6 py-4 text-sm text-white">26°C</td>
                  <td className="px-6 py-4">
                    <span className="status-badge status-warning">
                      Warning
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-white/60">1 min ago</td>
                </tr>
              </tbody>
            </table>
          </div>
        </motion.div>
      </main>

      {/* Floating Chat Button */}
      <Link href="/chat">
        <motion.button
          className="fixed bottom-8 right-8 w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full shadow-2xl hover:shadow-3xl transition-all duration-300 z-50 flex items-center justify-center group cursor-pointer border-4 border-white/20 animate-glow"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <MessageCircle className="h-7 w-7 text-white group-hover:scale-110 transition-transform" />
          <div className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center border-2 border-white">
            <span className="text-xs text-white font-bold">AI</span>
          </div>
          <div className="absolute -bottom-12 left-1/2 transform -translate-x-1/2 bg-black/90 text-white text-sm px-3 py-2 rounded-lg whitespace-nowrap z-50 font-medium opacity-0 group-hover:opacity-100 transition-opacity">
            🤖 AI Assistant
          </div>
        </motion.button>
      </Link>
    </div>
  )
}
