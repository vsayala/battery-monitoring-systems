'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import dynamic from 'next/dynamic'
import {
  MapPin,
  Battery,
  AlertCircle,
  TrendingUp,
  Globe,
  Layers,
  Filter,
  RefreshCw,
  Zap,
  Thermometer,
  Shield,
  Users
} from 'lucide-react'
import Link from 'next/link'
import { BatteryData, ApiResponse } from './types'

// Dynamically import the basic map to avoid SSR issues
const BasicMap = dynamic(() => import('./components/BasicMap'), { ssr: false })

// Import chart components
import VoltageChart from './components/VoltageChart'
import TemperatureChart from './components/TemperatureChart'

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

// Geo Dashboard Component
const GeoDashboard = () => {
  const [batteryData, setBatteryData] = useState<BatteryData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedSite, setSelectedSite] = useState<string>('all')
  const [mapView, setMapView] = useState<'markers' | 'heatmap' | 'clusters'>('markers')

  // Fetch battery data
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
        
      } catch (err) {
        console.error('Failed to fetch battery data:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchBatteryData()
    const interval = setInterval(fetchBatteryData, 30000)
    return () => clearInterval(interval)
  }, [])

  // Calculate stats for each site
  const getSiteStats = (siteId: string) => {
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

  // Get marker color based on status
  const getMarkerColor = (status: string) => {
    switch (status) {
      case 'Healthy': return '#10b981'
      case 'Warning': return '#f59e0b'
      case 'Critical': return '#ef4444'
      default: return '#6b7280'
    }
  }

  // Get marker size based on device count
  const getMarkerSize = (deviceCount: number) => {
    return Math.max(8, Math.min(20, deviceCount * 2))
  }



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
            <p className="text-red-400 mb-4 text-sm">Error loading geo dashboard: {error}</p>
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
                  <Globe className="h-7 w-7 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold gradient-text tracking-tight">
                  Battery Monitoring Geo Dashboard
                </h1>
                <p className="text-sm text-white/70 font-medium">
                  Geographic visualization of battery monitoring sites
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                <div className="connection-indicator connection-connected"></div>
                <span className="text-sm font-medium text-white/80">Connected</span>
              </div>
              <div className="flex items-center space-x-3">
                <Link href="/">
                  <button className="btn-secondary flex items-center space-x-2 group">
                    <Battery className="h-5 w-5 group-hover:scale-110 transition-transform" />
                    <span>Dashboard</span>
                  </button>
                </Link>
                <Link href="/mlops">
                  <button className="btn-secondary flex items-center space-x-2 group">
                    <TrendingUp className="h-5 w-5 group-hover:scale-110 transition-transform" />
                    <span>MLOps</span>
                  </button>
                </Link>
                <Link href="/chat">
                  <button className="btn-primary flex items-center space-x-2 group">
                    <AlertCircle className="h-5 w-5 group-hover:scale-110 transition-transform" />
                    <span>AI Assistant</span>
                  </button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Controls */}
        <div className="flex flex-wrap items-center gap-4 mb-8">
          <motion.div 
            className="glass-card p-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center space-x-3">
              <Filter className="h-5 w-5 text-blue-400" />
              <select 
                className="filter-select"
                value={selectedSite}
                onChange={(e) => setSelectedSite(e.target.value)}
              >
                <option value="all">All Sites ({mockSiteLocations.length})</option>
                {mockSiteLocations.map(site => (
                  <option key={site.site_id} value={site.site_id}>
                    {site.name} - {site.city}, {site.state}
                  </option>
                ))}
              </select>
            </div>
          </motion.div>

          <motion.div 
            className="glass-card p-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <div className="flex items-center space-x-3">
              <Layers className="h-5 w-5 text-green-400" />
              <select 
                className="filter-select"
                value={mapView}
                onChange={(e) => setMapView(e.target.value as any)}
              >
                <option value="markers">Markers View</option>
                <option value="heatmap">Heatmap View</option>
                <option value="clusters">Clusters View</option>
              </select>
            </div>
          </motion.div>

          <motion.div 
            className="glass-card p-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <div className="flex items-center space-x-3">
              <RefreshCw className="h-5 w-5 text-purple-400" />
              <span className="text-sm text-white/70">Auto-refresh: 30s</span>
            </div>
          </motion.div>
        </div>

        {/* Map Container */}
        <motion.div 
          className="glass-card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-white">Battery Sites Geographic Distribution</h3>
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
          
          <div className="h-96 rounded-xl overflow-hidden">
            <BasicMap
              batteryData={batteryData}
              mockSiteLocations={mockSiteLocations}
              getSiteStats={getSiteStats}
              getMarkerColor={getMarkerColor}
              getMarkerSize={getMarkerSize}
            />
          </div>
        </motion.div>

        {/* Voltage and Temperature Trends */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
          {/* Voltage Trend Chart */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Voltage Trend</h3>
              <Zap className="h-6 w-6 text-blue-400" />
            </div>
            <div className="h-64">
              <VoltageChart data={batteryData} />
            </div>
          </motion.div>

          {/* Temperature Trend Chart */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white">Temperature Trend</h3>
              <Thermometer className="h-6 w-6 text-red-400" />
            </div>
            <div className="h-64">
              <TemperatureChart data={batteryData} />
            </div>
          </motion.div>
        </div>

        {/* Site Statistics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-8">
          {mockSiteLocations.map((site, index) => {
            const stats = getSiteStats(site.site_id)
            if (!stats) return null

            return (
              <motion.div 
                key={site.site_id}
                className="glass-card p-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                      <MapPin className="h-5 w-5 text-blue-400" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-white">{site.name}</h4>
                      <p className="text-xs text-white/60">{site.city}, {site.state}</p>
                    </div>
                  </div>
                  <div className={`w-3 h-3 rounded-full ${
                    stats.status === 'Healthy' ? 'bg-green-500' : 
                    stats.status === 'Warning' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}></div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">{stats.totalDevices}</p>
                    <p className="text-xs text-white/60">Devices</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">{stats.totalCells}</p>
                    <p className="text-xs text-white/60">Cells</p>
                  </div>
                  <div className="text-center">
                    <p className="text-lg font-bold text-white">{stats.avgVoltage.toFixed(2)}V</p>
                    <p className="text-xs text-white/60">Avg Voltage</p>
                  </div>
                  <div className="text-center">
                    <p className="text-lg font-bold text-white">{stats.avgTemp.toFixed(1)}°C</p>
                    <p className="text-xs text-white/60">Avg Temp</p>
                  </div>
                </div>
                
                <div className="mt-4 pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white/70">Status</span>
                    <span className={`text-sm font-semibold ${
                      stats.status === 'Healthy' ? 'text-green-400' : 
                      stats.status === 'Warning' ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {stats.status}
                    </span>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </main>

      {/* Floating Back Button */}
      <Link href="/">
        <motion.button 
          className="fixed bottom-8 left-8 w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full shadow-2xl hover:shadow-3xl transition-all duration-300 z-50 flex items-center justify-center group cursor-pointer border-4 border-white/20 animate-glow"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <Battery className="h-7 w-7 text-white group-hover:scale-110 transition-transform" />
        </motion.button>
      </Link>
    </div>
  )
}

export default GeoDashboard 