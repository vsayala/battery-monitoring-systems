// Types for battery data
export interface BatteryData {
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

export interface ApiResponse {
  data: BatteryData[]
  total_records: number
} 