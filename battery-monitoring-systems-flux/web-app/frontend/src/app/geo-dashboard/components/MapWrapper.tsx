'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { BatteryData } from '../types'

// Dynamically import the map component with no SSR
const MapComponent = dynamic(() => import('./MapComponent'), { 
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center bg-gray-100 rounded-lg">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    </div>
  )
})

interface MapWrapperProps {
  batteryData: BatteryData[]
  mockSiteLocations: any[]
  getSiteStats: (siteId: string) => any
  getMarkerColor: (status: string) => string
  getMarkerSize: (deviceCount: number) => number
}

const MapWrapper = (props: MapWrapperProps) => {
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  if (!isClient) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-gray-100 rounded-lg">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return <MapComponent {...props} />
}

export default MapWrapper 