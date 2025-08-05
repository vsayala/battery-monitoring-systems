'use client'

import { useEffect, useRef, useState } from 'react'
import { BatteryData } from '../types'

interface SimpleMapProps {
  batteryData: BatteryData[]
  mockSiteLocations: any[]
  getSiteStats: (siteId: string) => any
  getMarkerColor: (status: string) => string
  getMarkerSize: (deviceCount: number) => number
}

const SimpleMap = ({ 
  batteryData, 
  mockSiteLocations, 
  getSiteStats, 
  getMarkerColor, 
  getMarkerSize 
}: SimpleMapProps) => {
  const [isLoaded, setIsLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<any>(null)

  useEffect(() => {
    if (typeof window !== 'undefined' && !isLoaded) {
      const loadMap = async () => {
        try {
          console.log('Loading map...')
          
          // Load Leaflet CSS if not already loaded
          if (!document.querySelector('link[href*="leaflet.css"]')) {
            const link = document.createElement('link')
            link.rel = 'stylesheet'
            link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'
            link.integrity = 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY='
            link.crossOrigin = ''
            document.head.appendChild(link)
          }
          
          // Dynamically import Leaflet only on client side
          const L = await import('leaflet')
          console.log('Leaflet imported successfully')
          
          if (mapRef.current && !mapInstanceRef.current) {
            // Create unique container ID
            const containerId = `map-${Date.now()}`
            mapRef.current.id = containerId
            
            console.log('Initializing map with container:', containerId)
            
            // Initialize map
            mapInstanceRef.current = L.map(containerId).setView([39.8283, -98.5795], 4)
            
            // Add tile layer
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
              attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(mapInstanceRef.current)
            
            console.log('Adding markers for', mockSiteLocations.length, 'sites')
            
            // Add markers
            mockSiteLocations.forEach(site => {
              const stats = getSiteStats(site.site_id)
              if (!stats) return
              
              const marker = L.circleMarker([site.lat, site.lng], {
                radius: getMarkerSize(stats.totalDevices),
                fillColor: getMarkerColor(stats.status),
                color: getMarkerColor(stats.status),
                weight: 2,
                opacity: 0.8,
                fillOpacity: 0.6
              }).addTo(mapInstanceRef.current)
              
              // Add popup
              const popupContent = `
                <div class="p-2">
                  <h3 class="font-bold text-lg mb-2">${site.name}</h3>
                  <p class="text-sm text-gray-600 mb-3">${site.city}, ${site.state}</p>
                  <div class="space-y-1 text-sm">
                    <div class="flex justify-between">
                      <span>Devices:</span>
                      <span class="font-semibold">${stats.totalDevices}</span>
                    </div>
                    <div class="flex justify-between">
                      <span>Cells:</span>
                      <span class="font-semibold">${stats.totalCells}</span>
                    </div>
                    <div class="flex justify-between">
                      <span>Avg Voltage:</span>
                      <span class="font-semibold">${stats.avgVoltage.toFixed(2)}V</span>
                    </div>
                    <div class="flex justify-between">
                      <span>Avg Temp:</span>
                      <span class="font-semibold">${stats.avgTemp.toFixed(1)}°C</span>
                    </div>
                    <div class="flex justify-between">
                      <span>Status:</span>
                      <span class="font-semibold ${stats.status === 'Healthy' ? 'text-green-600' : stats.status === 'Warning' ? 'text-yellow-600' : 'text-red-600'}">${stats.status}</span>
                    </div>
                  </div>
                </div>
              `
              
              marker.bindPopup(popupContent)
            })
            
            console.log('Map loaded successfully')
            setIsLoaded(true)
          }
        } catch (error) {
          console.error('Error loading map:', error)
          setError(error instanceof Error ? error.message : 'Failed to load map')
        }
      }
      
      loadMap()
    }
    
    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.remove()
          mapInstanceRef.current = null
        } catch (error) {
          console.log('Map cleanup error:', error)
        }
      }
    }
  }, [isLoaded, mockSiteLocations, getSiteStats, getMarkerColor, getMarkerSize])

  if (error) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-gray-100 rounded-lg">
        <div className="text-center">
          <p className="text-red-600 mb-2">Error loading map: {error}</p>
          <button 
            onClick={() => {
              setError(null)
              setIsLoaded(false)
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div 
      ref={mapRef}
      className="h-full w-full rounded-lg bg-gray-100"
      style={{ 
        minHeight: '384px',
        height: '384px',
        width: '100%'
      }}
    >
      {!isLoaded && (
        <div className="h-full w-full flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-gray-600">Loading map...</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default SimpleMap 