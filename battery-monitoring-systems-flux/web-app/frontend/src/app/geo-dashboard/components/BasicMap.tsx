'use client'

import { useEffect, useRef } from 'react'
import { BatteryData } from '../types'

interface BasicMapProps {
  batteryData: BatteryData[]
  mockSiteLocations: any[]
  getSiteStats: (siteId: string) => any
  getMarkerColor: (status: string) => string
  getMarkerSize: (deviceCount: number) => number
}

const BasicMap = ({ 
  batteryData, 
  mockSiteLocations, 
  getSiteStats, 
  getMarkerColor, 
  getMarkerSize 
}: BasicMapProps) => {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<any>(null)

  useEffect(() => {
    let mounted = true

    const initMap = async () => {
      if (!mounted || !mapRef.current || mapInstanceRef.current) return

      try {
        // Import Leaflet
        const L = await import('leaflet')
        
        // Create map
        const map = L.map(mapRef.current).setView([39.8283, -98.5795], 4)
        mapInstanceRef.current = map

        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '© OpenStreetMap contributors'
        }).addTo(map)

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
          }).addTo(map)

          const popupContent = `
            <div style="padding: 8px;">
              <h3 style="font-weight: bold; margin-bottom: 8px;">${site.name}</h3>
              <p style="color: #666; margin-bottom: 12px;">${site.city}, ${site.state}</p>
              <div style="font-size: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span>Devices:</span>
                  <span style="font-weight: bold;">${stats.totalDevices}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span>Cells:</span>
                  <span style="font-weight: bold;">${stats.totalCells}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span>Avg Voltage:</span>
                  <span style="font-weight: bold;">${stats.avgVoltage.toFixed(2)}V</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span>Avg Temp:</span>
                  <span style="font-weight: bold;">${stats.avgTemp.toFixed(1)}°C</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                  <span>Status:</span>
                  <span style="font-weight: bold; color: ${stats.status === 'Healthy' ? '#10b981' : stats.status === 'Warning' ? '#f59e0b' : '#ef4444'};">${stats.status}</span>
                </div>
              </div>
            </div>
          `

          marker.bindPopup(popupContent)
        })

      } catch (error) {
        console.error('Map initialization error:', error)
      }
    }

    // Small delay to ensure DOM is ready
    const timer = setTimeout(initMap, 100)

    return () => {
      mounted = false
      clearTimeout(timer)
      
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.remove()
          mapInstanceRef.current = null
        } catch (error) {
          console.log('Map cleanup error:', error)
        }
      }
    }
  }, [mockSiteLocations, getSiteStats, getMarkerColor, getMarkerSize])

  return (
    <div 
      ref={mapRef}
      style={{ 
        height: '384px',
        width: '100%',
        borderRadius: '8px',
        backgroundColor: '#f3f4f6'
      }}
    />
  )
}

export default BasicMap 