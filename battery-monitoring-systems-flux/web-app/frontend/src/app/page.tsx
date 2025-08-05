'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import dynamic from 'next/dynamic'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { Line } from 'react-chartjs-2'
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
  Brain,
  Globe,
  MapPin,
  Activity
} from 'lucide-react'
import Link from 'next/link'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

// Dynamically import the map component to avoid SSR issues
const BasicMap = dynamic(() => import('./geo-dashboard/components/BasicMap'), { ssr: false })

// Types for battery data
interface BatteryData {
  id: number
  device_id: number
  cell_number: number
  cell_voltage: number
  cell_temperature: number
  cell_specific_gravity: number
  packet_datetime: string
  site_id: string
  string_voltage: number
  problem_cells: number
  cells_connected_count: number
}

interface ApiResponse {
  data: BatteryData[]
  total_records: number
}

// Default stats
const defaultStats = {
  devices: '0',
  cells: '0',
  alerts: '0',
  status: 'Loading...',
  uptime: '0%',
  efficiency: '0%'
}

// Mock geographic data for battery sites
const mockSiteLocations = [
  { site_id: 'SITE001', name: 'Battery Farm Alpha', lat: 37.7749, lng: -122.4194, city: 'San Francisco', state: 'CA' },
  { site_id: 'SITE002', name: 'Power Station Beta', lat: 34.0522, lng: -118.2437, city: 'Los Angeles', state: 'CA' },
  { site_id: 'SITE003', name: 'Energy Hub Gamma', lat: 40.7128, lng: -74.0060, city: 'New York', state: 'NY' },
  { site_id: 'SITE004', name: 'Grid Center Delta', lat: 41.8781, lng: -87.6298, city: 'Chicago', state: 'IL' },
  { site_id: 'SITE005', name: 'Renewable Station Epsilon', lat: 29.7604, lng: -95.3698, city: 'Houston', state: 'TX' },
  { site_id: 'SITE006', name: 'Smart Grid Zeta', lat: 33.7490, lng: -84.3880, city: 'Atlanta', state: 'GA' },
  { site_id: 'SITE007', name: 'Battery Complex Eta', lat: 25.7617, lng: -80.1918, city: 'Miami', state: 'FL' },
  { site_id: 'SITE008', name: 'Power Network Theta', lat: 47.6062, lng: -122.3321, city: 'Seattle', state: 'WA' }
]

// Chart.js Components
const VoltageChart = ({ data }: { data: BatteryData[] }) => {
  // Process data for Chart.js
  const chartData = data.slice(-15).map((item, index) => ({
    time: new Date(item.packet_datetime).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      hour12: false 
    }),
    voltage: parseFloat(item.cell_voltage.toString()),
    cell: `Cell-${item.cell_number.toString().padStart(2, '0')}`,
    device: `Device-${item.device_id.toString().padStart(3, '0')}`
  }))

  const chartConfig = {
    data: {
      labels: chartData.map(d => d.time),
      datasets: [
        {
          label: 'Voltage (V)',
          data: chartData.map(d => d.voltage),
          borderColor: '#fbbf24',
          backgroundColor: 'rgba(251, 191, 36, 0.2)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointBackgroundColor: '#fbbf24',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: '#fbbf24',
          pointHoverBorderColor: '#ffffff',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          borderColor: '#fbbf24',
          borderWidth: 1,
          cornerRadius: 8,
          displayColors: false,
          callbacks: {
            title: function(context: any) {
              const index = context[0].dataIndex
              const item = chartData[index]
              return `${item.device} - ${item.cell}`
            },
            label: function(context: any) {
              return `Voltage: ${context.parsed.y.toFixed(2)}V`
            },
            afterLabel: function(context: any) {
              const index = context.dataIndex
              const item = chartData[index]
              return `Time: ${item.time}`
            }
          }
        }
      },
             scales: {
         x: {
           display: true,
           grid: {
             color: 'rgba(255, 255, 255, 0.1)',
             drawBorder: false
           },
           ticks: {
             color: 'rgba(255, 255, 255, 0.8)',
             font: {
               size: 10
             }
           }
         },
         y: {
           display: true,
           grid: {
             color: 'rgba(255, 255, 255, 0.1)',
             drawBorder: false
           },
           ticks: {
             color: 'rgba(255, 255, 255, 0.8)',
             font: {
               size: 10
             },
             callback: function(value: any) {
               return `${Number(value).toFixed(2)}V`
             }
           }
         }
       },
      interaction: {
        intersect: false,
        mode: 'index' as const
      }
    }
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-white/50 text-xs">No voltage data available</p>
      </div>
    )
  }

  return (
    <div className="w-full h-full">
      <Line data={chartConfig.data} options={chartConfig.options} />
    </div>
  )
}

const TemperatureChart = ({ data }: { data: BatteryData[] }) => {
  // Process data for Chart.js
  const chartData = data.slice(-15).map((item, index) => ({
    time: new Date(item.packet_datetime).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      hour12: false 
    }),
    temperature: parseFloat(item.cell_temperature.toString()),
    cell: `Cell-${item.cell_number.toString().padStart(2, '0')}`,
    device: `Device-${item.device_id.toString().padStart(3, '0')}`
  }))

  const chartConfig = {
    data: {
      labels: chartData.map(d => d.time),
      datasets: [
        {
          label: 'Temperature (°C)',
          data: chartData.map(d => d.temperature),
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.2)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointBackgroundColor: '#ef4444',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: '#ef4444',
          pointHoverBorderColor: '#ffffff',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          borderColor: '#ef4444',
          borderWidth: 1,
          cornerRadius: 8,
          displayColors: false,
          callbacks: {
            title: function(context: any) {
              const index = context[0].dataIndex
              const item = chartData[index]
              return `${item.device} - ${item.cell}`
            },
            label: function(context: any) {
              return `Temperature: ${context.parsed.y.toFixed(1)}°C`
            },
            afterLabel: function(context: any) {
              const index = context.dataIndex
              const item = chartData[index]
              return `Time: ${item.time}`
            }
          }
        }
      },
      scales: {
        x: {
          display: true,
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
            drawBorder: false
          },
          ticks: {
            color: 'rgba(255, 255, 255, 0.8)',
            font: {
              size: 10
            }
          }
        },
        y: {
          display: true,
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
            drawBorder: false
          },
          ticks: {
            color: 'rgba(255, 255, 255, 0.8)',
            font: {
              size: 10
            },
            callback: function(value: any) {
              return `${Number(value).toFixed(1)}°C`
            }
          }
        }
      },
      interaction: {
        intersect: false,
        mode: 'index' as const
      }
    }
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-white/50 text-xs">No temperature data available</p>
      </div>
    )
  }

  return (
    <div className="w-full h-full">
      <Line data={chartConfig.data} options={chartConfig.options} />
    </div>
  )
}

// Helper functions for map
const getSiteStats = (siteId: string, batteryData: BatteryData[]) => {
  const siteData = batteryData.filter(item => item.site_id === siteId)
  if (siteData.length === 0) return null

  const totalDevices = new Set(siteData.map(item => item.device_id)).size
  const totalCells = siteData.length
  const avgVoltage = siteData.reduce((sum, item) => sum + item.cell_voltage, 0) / siteData.length
  const avgTemp = siteData.reduce((sum, item) => sum + item.cell_temperature, 0) / siteData.length
  const problemCells = siteData.filter(item => item.problem_cells > 0).length
  const status = avgVoltage > 3.0 && avgTemp < 50 ? 'Healthy' : 'Warning'

  return {
    totalDevices,
    totalCells,
    avgVoltage,
    avgTemp,
    problemCells,
    status
  }
}

const getMarkerColor = (status: string) => {
  switch (status) {
    case 'Healthy': return '#10b981'
    case 'Warning': return '#f59e0b'
    case 'Critical': return '#ef4444'
    default: return '#6b7280'
  }
}

const getMarkerSize = (deviceCount: number) => {
  return Math.max(8, Math.min(20, deviceCount * 2))
}

export default function Dashboard() {
  const [batteryData, setBatteryData] = useState<BatteryData[]>([])
  const [stats, setStats] = useState(defaultStats)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedDevice, setSelectedDevice] = useState<string>('all')
  const [selectedCell, setSelectedCell] = useState<string>('all')

  // Fetch real battery data
  useEffect(() => {
    const fetchBatteryData = async () => {
      try {
        setIsLoading(true)
        setError(null)
        
        const response = await fetch('http://localhost:8000/api/battery-data?limit=100')
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        
        const result: ApiResponse = await response.json()
        setBatteryData(result.data)
        
        // Calculate real stats from the data
        const uniqueDevices = new Set(result.data.map(item => item.device_id)).size
        const totalCells = result.data.length
        const problemCells = result.data.filter(item => item.problem_cells > 0).length
        const avgVoltage = result.data.reduce((sum, item) => sum + item.cell_voltage, 0) / result.data.length
        const avgTemp = result.data.reduce((sum, item) => sum + item.cell_temperature, 0) / result.data.length
        
        setStats({
          devices: uniqueDevices.toString(),
          cells: totalCells.toString(),
          alerts: problemCells.toString(),
          status: avgVoltage > 3.0 && avgTemp < 50 ? 'Healthy' : 'Warning',
          uptime: '99.8%',
          efficiency: `${Math.round((avgVoltage / 4.2) * 100)}%`
        })
        
      } catch (err) {
        console.error('Failed to fetch battery data:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchBatteryData()
    
    // Set up real-time updates every 30 seconds
    const interval = setInterval(fetchBatteryData, 30000)
    
    return () => clearInterval(interval)
  }, [])

  // Get unique devices and cells for dropdowns
  const uniqueDevices = Array.from(new Set(batteryData.map(item => item.device_id))).sort((a, b) => a - b)
  const uniqueCells = Array.from(new Set(batteryData.map(item => item.cell_number))).sort((a, b) => a - b)

  // Filter data based on selected device and cell
  const filteredData = batteryData.filter(item => {
    const deviceMatch = selectedDevice === 'all' || item.device_id.toString() === selectedDevice
    const cellMatch = selectedCell === 'all' || item.cell_number.toString() === selectedCell
    return deviceMatch && cellMatch
  })

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="flex items-center justify-center h-screen">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="flex items-center justify-center h-screen">
          <div className="text-center">
            <p className="text-red-400 mb-4 text-sm">Error loading dashboard: {error}</p>
            <button 
              onClick={() => window.location.reload()} 
              className="px-4 py-2 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    )
  }

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
                <h1 className="text-2xl font-bold gradient-text tracking-tight">
                  Battery Monitoring System
                </h1>
                <p className="text-sm text-white/70 font-medium">
                  Real-time ML/LLM & MLOps Dashboard
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                <div className="connection-indicator connection-connected"></div>
                <span className="text-sm font-medium text-white/80">Connected</span>
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
                <Link href="/geo-dashboard">
                  <button className="btn-secondary flex items-center space-x-2 group">
                    <Globe className="h-5 w-5 group-hover:scale-110 transition-transform" />
                    <span>Geo Dashboard</span>
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
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Battery Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="stat-card"
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <Battery className="h-5 w-5 text-blue-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Total Devices</p>
                <p className="text-xl font-bold text-white">{uniqueDevices.length}</p>
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
                <Zap className="h-5 w-5 text-green-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Total Cells</p>
                <p className="text-xl font-bold text-white">{uniqueCells.length}</p>
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
                <Activity className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Data Points</p>
                <p className="text-xl font-bold text-white">{batteryData.length}</p>
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
                <Thermometer className="h-5 w-5 text-yellow-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Avg Temp</p>
                <p className="text-xl font-bold text-white">
                  {batteryData.length > 0 
                    ? (batteryData.reduce((sum, item) => sum + item.cell_temperature, 0) / batteryData.length).toFixed(1)
                    : '0.0'}°C
                </p>
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
                <Zap className="h-5 w-5 text-indigo-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Avg Voltage</p>
                <p className="text-xl font-bold text-white">
                  {batteryData.length > 0 
                    ? (batteryData.reduce((sum, item) => sum + item.cell_voltage, 0) / batteryData.length).toFixed(2)
                    : '0.00'}V
                </p>
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
              <div className="w-10 h-10 bg-pink-500/20 rounded-lg flex items-center justify-center">
                <Clock className="h-5 w-5 text-pink-400" />
              </div>
              <div>
                <p className="text-xs font-medium text-white/70 uppercase tracking-wide">Last Update</p>
                <p className="text-sm font-bold text-white">
                  {batteryData.length > 0 
                    ? new Date(batteryData[batteryData.length - 1].packet_datetime).toLocaleTimeString()
                    : 'N/A'}
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Charts */}
        <div className="flex gap-4 mb-6">
          <motion.div 
            className="chart-container flex-1"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-semibold text-white">Voltage Trend</h3>
              <div className="w-6 h-6 bg-yellow-500/20 rounded flex items-center justify-center">
                <Zap className="h-3 w-3 text-yellow-400" />
              </div>
            </div>
            <div className="h-32 bg-white/5 rounded flex items-center justify-center p-2">
              {filteredData.length > 0 ? (
                <VoltageChart data={filteredData} />
              ) : (
                <div className="text-center">
                  <p className="text-gray-400 text-xs">Loading voltage data...</p>
                </div>
              )}
            </div>
          </motion.div>

          <motion.div 
            className="chart-container flex-1"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-semibold text-white">Temperature Trend</h3>
              <div className="w-6 h-6 bg-red-500/20 rounded flex items-center justify-center">
                <Thermometer className="h-3 w-3 text-red-400" />
              </div>
            </div>
            <div className="h-32 bg-white/5 rounded flex items-center justify-center p-2">
              {filteredData.length > 0 ? (
                <TemperatureChart data={filteredData} />
              ) : (
                <div className="text-center">
                  <p className="text-gray-400 text-xs">Loading temperature data...</p>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Geographic Map */}
        <motion.div 
          className="chart-container mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Battery Sites Geographic Distribution</h3>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="text-xs text-white/70">Healthy</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <span className="text-xs text-white/70">Warning</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <span className="text-xs text-white/70">Critical</span>
              </div>
            </div>
          </div>
          <div className="h-80 rounded-lg overflow-hidden">
            <BasicMap
              batteryData={filteredData}
              mockSiteLocations={mockSiteLocations}
              getSiteStats={(siteId) => getSiteStats(siteId, filteredData)}
              getMarkerColor={getMarkerColor}
              getMarkerSize={getMarkerSize}
            />
          </div>
        </motion.div>

        {/* Data Table */}
        <motion.div 
          className="data-table"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.9 }}
        >
          <div className="px-6 py-4 border-b border-white/10">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold text-white">Battery Data ({filteredData.length} records)</h3>
              <div className="flex items-center space-x-4">
                <select 
                  className="filter-select"
                  value={selectedDevice}
                  onChange={(e) => setSelectedDevice(e.target.value)}
                >
                  <option value="all">All Devices ({uniqueDevices.length})</option>
                  {uniqueDevices.map(deviceId => (
                    <option key={deviceId} value={deviceId.toString()}>
                      Device-{deviceId.toString().padStart(3, '0')}
                    </option>
                  ))}
                </select>
                <select 
                  className="filter-select"
                  value={selectedCell}
                  onChange={(e) => setSelectedCell(e.target.value)}
                >
                  <option value="all">All Cells ({uniqueCells.length})</option>
                  {uniqueCells.map(cellNumber => (
                    <option key={cellNumber} value={cellNumber.toString()}>
                      Cell-{cellNumber.toString().padStart(2, '0')}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
          <div className="overflow-x-auto overflow-y-auto max-h-[500px]">
            <table className="w-full">
              <thead className="bg-white/5 sticky top-0 z-10">
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
                {filteredData.map((item) => (
                  <tr key={item.id} className="hover:bg-white/5 transition-colors">
                    <td className="px-6 py-4 text-sm text-white">Device-{item.device_id.toString().padStart(3, '0')}</td>
                    <td className="px-6 py-4 text-sm text-white">Cell-{item.cell_number.toString().padStart(2, '0')}</td>
                    <td className="px-6 py-4 text-sm text-white">{item.cell_voltage.toFixed(2)}V</td>
                    <td className="px-6 py-4 text-sm text-white">{item.cell_temperature.toFixed(1)}°C</td>
                    <td className="px-6 py-4">
                      <span className={`status-badge ${
                        item.cell_voltage > 3.0 && item.cell_temperature < 50 
                          ? 'status-healthy' 
                          : 'status-warning'
                      }`}>
                        {item.cell_voltage > 3.0 && item.cell_temperature < 50 ? 'Healthy' : 'Warning'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-white/60">
                      {new Date(item.packet_datetime).toLocaleString()}
                    </td>
                  </tr>
                ))}
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
