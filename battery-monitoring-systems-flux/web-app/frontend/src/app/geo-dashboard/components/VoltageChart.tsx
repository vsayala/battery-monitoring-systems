'use client'

import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { BatteryData } from '../types'

interface VoltageChartProps {
  data: BatteryData[]
}

const VoltageChart = ({ data }: VoltageChartProps) => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return

    // Clear previous chart
    d3.select(svgRef.current).selectAll("*").remove()

    // Prepare data
    const chartData = data
      .slice(-50) // Last 50 data points
      .map((item, index) => ({
        time: index,
        voltage: parseFloat(item.cell_voltage.toString()),
        datetime: item.packet_datetime
      }))

    if (chartData.length === 0) return

    // Set up dimensions
    const margin = { top: 20, right: 20, bottom: 30, left: 40 }
    const width = 400 - margin.left - margin.right
    const height = 200 - margin.top - margin.bottom

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, chartData.length - 1])
      .range([0, width])

    const yScale = d3.scaleLinear()
      .domain([d3.min(chartData, d => d.voltage) * 0.95, d3.max(chartData, d => d.voltage) * 1.05])
      .range([height, 0])

    // Create line generator
    const line = d3.line<{ time: number; voltage: number }>()
      .x(d => xScale(d.time))
      .y(d => yScale(d.voltage))
      .curve(d3.curveMonotoneX)

    // Add the line path
    svg.append('path')
      .datum(chartData)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', line)

    // Add area fill
    const area = d3.area<{ time: number; voltage: number }>()
      .x(d => xScale(d.time))
      .y0(height)
      .y1(d => yScale(d.voltage))
      .curve(d3.curveMonotoneX)

    svg.append('path')
      .datum(chartData)
      .attr('fill', 'url(#voltageGradient)')
      .attr('d', area)

    // Add gradient
    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient')
      .attr('id', 'voltageGradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%')

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#3b82f6')
      .attr('stop-opacity', 0.3)

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#3b82f6')
      .attr('stop-opacity', 0.1)

    // Add dots
    svg.selectAll('.dot')
      .data(chartData)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.time))
      .attr('cy', d => yScale(d.voltage))
      .attr('r', 3)
      .attr('fill', '#3b82f6')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 1)

    // Add axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(5)
      .tickFormat((d, i) => {
        if (i === 0 || i === chartData.length - 1) {
          return chartData[i]?.datetime ? 
            new Date(chartData[i].datetime).toLocaleTimeString() : ''
        }
        return ''
      })

    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => `${d.toFixed(2)}V`)

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis)
      .selectAll('text')
      .style('font-size', '10px')
      .style('fill', '#9ca3af')

    svg.append('g')
      .call(yAxis)
      .selectAll('text')
      .style('font-size', '10px')
      .style('fill', '#9ca3af')

    // Add grid lines
    svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale)
        .ticks(5)
        .tickSize(-height)
        .tickFormat('')
      )
      .style('stroke-dasharray', '3,3')
      .style('stroke', '#374151')
      .style('stroke-opacity', 0.3)

    svg.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale)
        .ticks(5)
        .tickSize(-width)
        .tickFormat('')
      )
      .style('stroke-dasharray', '3,3')
      .style('stroke', '#374151')
      .style('stroke-opacity', 0.3)

  }, [data])

  if (!data || data.length === 0) {
    return (
      <div className="h-full w-full flex items-center justify-center">
        <p className="text-gray-500 text-sm">No voltage data available</p>
      </div>
    )
  }

  return (
    <div className="w-full h-full">
      <svg ref={svgRef} className="w-full h-full"></svg>
    </div>
  )
}

export default VoltageChart 