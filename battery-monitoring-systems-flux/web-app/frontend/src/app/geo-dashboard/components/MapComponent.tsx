'use client'

import { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import { BatteryData } from '../types'

interface MapComponentProps {
  batteryData: BatteryData[]
  mockSiteLocations: any[]
  getSiteStats: (siteId: string) => any
  getMarkerColor: (status: string) => string
  getMarkerSize: (deviceCount: number) => number
}

const MapComponent = ({ 
  batteryData, 
  mockSiteLocations, 
  getSiteStats, 
  getMarkerColor, 
  getMarkerSize 
}: MapComponentProps) => {
  const mapRef = useRef<any>(null)

  // Cleanup function to prevent memory leaks
  useEffect(() => {
    return () => {
      if (mapRef.current) {
        try {
          mapRef.current.remove()
        } catch (error) {
          console.log('Map cleanup error:', error)
        }
      }
    }
  }, [])

  return (
    <MapContainer 
      center={[39.8283, -98.5795]} 
      zoom={4} 
      style={{ height: '100%', width: '100%' }}
      className="z-10"
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      
      {mockSiteLocations.map(site => {
        const stats = getSiteStats(site.site_id)
        if (!stats) return null

        return (
          <CircleMarker
            key={site.site_id}
            center={[site.lat, site.lng]}
            radius={getMarkerSize(stats.totalDevices)}
            fillColor={getMarkerColor(stats.status)}
            color={getMarkerColor(stats.status)}
            weight={2}
            opacity={0.8}
            fillOpacity={0.6}
          >
            <Popup>
              <div className="p-2">
                <h3 className="font-bold text-lg mb-2">{site.name}</h3>
                <p className="text-sm text-gray-600 mb-3">{site.city}, {site.state}</p>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Devices:</span>
                    <span className="font-semibold">{stats.totalDevices}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cells:</span>
                    <span className="font-semibold">{stats.totalCells}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg Voltage:</span>
                    <span className="font-semibold">{stats.avgVoltage.toFixed(2)}V</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg Temp:</span>
                    <span className="font-semibold">{stats.avgTemp.toFixed(1)}°C</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Status:</span>
                    <span className={`font-semibold ${
                      stats.status === 'Healthy' ? 'text-green-600' : 
                      stats.status === 'Warning' ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {stats.status}
                    </span>
                  </div>
                </div>
              </div>
            </Popup>
          </CircleMarker>
        )
      })}
    </MapContainer>
  )
}

export default MapComponent 