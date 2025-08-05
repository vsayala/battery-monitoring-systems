'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import * as d3 from 'd3'
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

// D3 Chart Components
const VoltageChart = ({ data }: { data: BatteryData[] }) => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return

    // Clear previous chart
    d3.select(svgRef.current).selectAll("*").remove()

    // Process data
    const chartData = data.slice(0, 20).map((item, index) => {
      const timestamp = new Date(item.packet_datetime)
      return {
        time: timestamp,
        voltage: item.cell_voltage,
        cell: `Cell-${item.cell_number.toString().padStart(2, '0')}`
      }
    })

    // Set up dimensions
    const margin = { top: 20, right: 30, bottom: 30, left: 40 }
    const width = 600 - margin.left - margin.right
    const height = 300 - margin.top - margin.bottom

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(chartData, d => d.time) as [Date, Date])
      .range([0, width])

    const yScale = d3.scaleLinear()
      .domain([d3.min(chartData, d => d.voltage)! - 0.1, d3.max(chartData, d => d.voltage)! + 0.1])
      .range([height, 0])

    // Line generator
    const line = d3.line<{time: Date, voltage: number}>()
      .x(d => xScale(d.time))
      .y(d => yScale(d.voltage))
      .curve(d3.curveMonotoneX)

    // Add grid
    svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickSize(-height).tickFormat(() => ''))
      .style('stroke', 'rgba(255,255,255,0.1)')
      .style('stroke-width', 1)

    svg.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(() => ''))
      .style('stroke', 'rgba(255,255,255,0.1)')
      .style('stroke-width', 1)

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat((d: any) => {
        const date = d as Date;
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
      }))
      .style('color', 'rgba(255,255,255,0.7)')
      .style('font-size', '10px')

    svg.append('g')
      .call(d3.axisLeft(yScale))
      .style('color', 'rgba(255,255,255,0.7)')
      .style('font-size', '10px')

    // Add area
    const area = d3.area<{time: Date, voltage: number}>()
      .x(d => xScale(d.time))
      .y0(height)
      .y1(d => yScale(d.voltage))
      .curve(d3.curveMonotoneX)

    svg.append('path')
      .datum(chartData)
      .attr('fill', 'rgba(251, 191, 36, 0.3)')
      .attr('d', area)

    // Add line
    svg.append('path')
      .datum(chartData)
      .attr('fill', 'none')
      .attr('stroke', '#fbbf24')
      .attr('stroke-width', 2)
      .attr('d', line)

    // Add dots
    svg.selectAll('.dot')
      .data(chartData)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.time))
      .attr('cy', d => yScale(d.voltage))
      .attr('r', 3)
      .attr('fill', '#fbbf24')
      .style('opacity', 0.7)

  }, [data])

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-white/50 text-xs">No voltage data available</p>
      </div>
    )
  }

  return (
    <div className="w-full h-full flex items-center justify-center">
      <svg ref={svgRef} className="w-full h-full"></svg>
    </div>
  )
}

const TemperatureChart = ({ data }: { data: BatteryData[] }) => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return

    // Clear previous chart
    d3.select(svgRef.current).selectAll("*").remove()

    // Process data
    const chartData = data.slice(0, 20).map((item, index) => {
      const timestamp = new Date(item.packet_datetime)
      return {
        time: timestamp,
        temp: item.cell_temperature,
        cell: `Cell-${item.cell_number.toString().padStart(2, '0')}`
      }
    })

    // Set up dimensions
    const margin = { top: 20, right: 30, bottom: 30, left: 40 }
    const width = 600 - margin.left - margin.right
    const height = 300 - margin.top - margin.bottom

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(chartData, d => d.time) as [Date, Date])
      .range([0, width])

    const yScale = d3.scaleLinear()
      .domain([d3.min(chartData, d => d.temp)! - 1, d3.max(chartData, d => d.temp)! + 1])
      .range([height, 0])

    // Line generator
    const line = d3.line<{time: Date, temp: number}>()
      .x(d => xScale(d.time))
      .y(d => yScale(d.temp))
      .curve(d3.curveMonotoneX)

    // Add grid
    svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickSize(-height).tickFormat(() => ''))
      .style('stroke', 'rgba(255,255,255,0.1)')
      .style('stroke-width', 1)

    svg.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(() => ''))
      .style('stroke', 'rgba(255,255,255,0.1)')
      .style('stroke-width', 1)

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat((d: any) => {
        const date = d as Date;
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
      }))
      .style('color', 'rgba(255,255,255,0.7)')
      .style('font-size', '10px')

    svg.append('g')
      .call(d3.axisLeft(yScale))
      .style('color', 'rgba(255,255,255,0.7)')
      .style('font-size', '10px')

    // Add area
    const area = d3.area<{time: Date, temp: number}>()
      .x(d => xScale(d.time))
      .y0(height)
      .y1(d => yScale(d.temp))
      .curve(d3.curveMonotoneX)

    svg.append('path')
      .datum(chartData)
      .attr('fill', 'rgba(239, 68, 68, 0.3)')
      .attr('d', area)

    // Add line
    svg.append('path')
      .datum(chartData)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 2)
      .attr('d', line)

    // Add dots
    svg.selectAll('.dot')
      .data(chartData)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.time))
      .attr('cy', d => yScale(d.temp))
      .attr('r', 3)
      .attr('fill', '#ef4444')
      .style('opacity', 0.7)

  }, [data])

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-white/50 text-xs">No temperature data available</p>
      </div>
    )
  }

  return (
    <div className="w-full h-full flex items-center justify-center">
      <svg ref={svgRef} className="w-full h-full"></svg>
    </div>
  )
}

export default function Dashboard() {
  const [batteryData, setBatteryData] = useState<BatteryData[]>([])
  const [stats, setStats] = useState(defaultStats)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

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
        {/* Stats Cards */}
        <div className="flex flex-wrap items-center gap-4 mb-8">
          <motion.div 
            className="stat-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
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
            className="stat-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
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
            className="stat-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
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
            className="stat-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
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
            className="stat-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
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
            className="stat-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
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

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <motion.div 
            className="chart-container"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Voltage Trend</h3>
              <div className="w-10 h-10 bg-yellow-500/20 rounded-xl flex items-center justify-center">
                <Zap className="h-5 w-5 text-yellow-400" />
              </div>
            </div>
            <div className="h-64 bg-white/5 rounded-xl flex items-center justify-center p-4">
              <VoltageChart data={batteryData} />
            </div>
          </motion.div>

          <motion.div 
            className="chart-container"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Temperature Trend</h3>
              <div className="w-10 h-10 bg-red-500/20 rounded-xl flex items-center justify-center">
                <Thermometer className="h-5 w-5 text-red-400" />
              </div>
            </div>
            <div className="h-64 bg-white/5 rounded-xl flex items-center justify-center p-4">
              <TemperatureChart data={batteryData} />
            </div>
          </motion.div>
        </div>

        {/* Data Table */}
        <motion.div 
          className="data-table"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          <div className="px-6 py-4 border-b border-white/10">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold text-white">Battery Data ({batteryData.length} records)</h3>
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
          <div className="overflow-x-auto overflow-y-auto max-h-[600px]">
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
                {batteryData.map((item) => (
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
