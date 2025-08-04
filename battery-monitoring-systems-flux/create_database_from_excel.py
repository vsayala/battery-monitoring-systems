#!/usr/bin/env python3
"""
Script to create battery_monitoring.db from Excel data with proper schema.
This matches the actual data structure from the Excel file.
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_database_from_excel():
    """Create SQLite database from Excel data with proper schema."""
    
    # File paths
    excel_file = 'data/data.xlsx'
    db_file = 'battery_monitoring.db'
    
    logger.info(f"Reading Excel file: {excel_file}")
    
    try:
        # Read Excel data
        df = pd.read_excel(excel_file)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Display column information
        logger.info("Columns found:")
        for i, col in enumerate(df.columns, 1):
            logger.info(f"  {i:2d}. {col}")
        
        # Create SQLite database
        logger.info(f"Creating database: {db_file}")
        
        # Remove existing database if it exists
        if os.path.exists(db_file):
            os.remove(db_file)
            logger.info("Removed existing database file")
        
        # Connect to SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Create table with proper schema based on actual data
        create_table_sql = """
        CREATE TABLE battery_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            packet_id INTEGER,
            start_packet TEXT,
            protocol_version REAL,
            data_identifier TEXT,
            site_id TEXT,
            time TEXT,
            date TEXT,
            packet_datetime TEXT,
            device_id INTEGER,
            bms_manufacturer_id TEXT,
            serial_number TEXT,
            installation_date TEXT,
            cells_connected_count INTEGER,
            problem_cells INTEGER,
            cell_number INTEGER,
            cell_voltage REAL,
            cell_temperature REAL,
            cell_specific_gravity REAL,
            cell_server_time TEXT,
            string_voltage REAL,
            system_peak_current_in_charge_one_cycle REAL,
            average_discharging_current REAL,
            average_charging_current REAL,
            ah_in_for_one_charge_cycle REAL,
            ah_out_for_one_discharge_cycle REAL,
            cumulative_ah_in REAL,
            cumulative_ah_out REAL,
            charge_time_cycle REAL,
            discharge_time_cycle REAL,
            total_charging_energy REAL,
            total_discharging_energy REAL,
            every_hour_avg_temp REAL,
            cumulative_total_avg_temp_every_hour REAL,
            charge_or_discharge_cycle TEXT,
            soc_latest_value_for_every_cycle REAL,
            dod_latest_value_for_every_cycle REAL,
            system_peak_current_in_discharge_one_cycle REAL,
            instantaneous_current REAL,
            ambient_temperature REAL,
            battery_run_hours REAL,
            bms_bank_discharge_cycle INTEGER,
            bms_ambient_temperature_hn REAL,
            bms_soc_ln REAL,
            bms_string_voltage_lnh REAL,
            bms_string_current_hn REAL,
            bms_bms_sed_communication TEXT,
            bms_cell_communication TEXT,
            bms_cell_voltage_ln REAL,
            bms_cell_voltage_nh REAL,
            bms_cell_temperature_hn REAL,
            bms_buzzer TEXT,
            charger_id TEXT,
            charger_device_id TEXT,
            ac_voltage REAL,
            ac_current REAL,
            frequency REAL,
            energy REAL,
            charger_input_mains TEXT,
            charger_input_phase TEXT,
            charger_dc_voltage_oln REAL,
            charger_ac_voltage_uln REAL,
            charger_load REAL,
            charger_trip TEXT,
            charger_output_mccb TEXT,
            charger_battery_condition TEXT,
            charger_test_push_button TEXT,
            charger_reset_push_button TEXT,
            charger_alarm_supply_fuse BOOLEAN,
            charger_filter_fuse BOOLEAN,
            charger_output_fuse BOOLEAN,
            charger_input_fuse BOOLEAN,
            charger_rectifier_fuse TEXT,
            server_time TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_table_sql)
        logger.info("Created battery_data table with proper schema")
        
        # Clean and prepare data for insertion
        logger.info("Preparing data for insertion...")
        
        # Rename columns to match database schema (snake_case)
        column_mapping = {
            'PacketID': 'packet_id',
            'StartPacket': 'start_packet',
            'ProtocolVersion': 'protocol_version',
            'DataIdentifier': 'data_identifier',
            'SiteID': 'site_id',
            'Time': 'time',
            'Date': 'date',
            'PacketDateTime': 'packet_datetime',
            'DeviceID': 'device_id',
            'BMSManufacturerID': 'bms_manufacturer_id',
            'SerialNumber': 'serial_number',
            'InstallationDate': 'installation_date',
            'CellsConnectedCount': 'cells_connected_count',
            'ProblemCells': 'problem_cells',
            'CellNumber': 'cell_number',
            'CellVoltage': 'cell_voltage',
            'CellTemperature': 'cell_temperature',
            'CellSpecificGravity': 'cell_specific_gravity',
            'CellServerTime': 'cell_server_time',
            'StringVoltage': 'string_voltage',
            'SystemPeakCurrentInChargeOneCycle': 'system_peak_current_in_charge_one_cycle',
            'AverageDischargingCurrent': 'average_discharging_current',
            'AverageChargingCurrent': 'average_charging_current',
            'AhInForOneChargeCycle': 'ah_in_for_one_charge_cycle',
            'AhOutForOneDischargeCycle': 'ah_out_for_one_discharge_cycle',
            'CumulativeAHIn': 'cumulative_ah_in',
            'CumulativeAHOut': 'cumulative_ah_out',
            'ChargeTimeCycle': 'charge_time_cycle',
            'DischargeTimeCycle': 'discharge_time_cycle',
            'TotalChargingEnergy': 'total_charging_energy',
            'TotalDischargingEnergy': 'total_discharging_energy',
            'EveryHourAvgTemp': 'every_hour_avg_temp',
            'CumulativeTotalAvgTempEveryHour': 'cumulative_total_avg_temp_every_hour',
            'ChargeOrDischargeCycle': 'charge_or_discharge_cycle',
            'SocLatestValueForEveryCycle': 'soc_latest_value_for_every_cycle',
            'DodLatestValueForEveryCycle': 'dod_latest_value_for_every_cycle',
            'SystemPeakCurrentInDischargeOneCycle': 'system_peak_current_in_discharge_one_cycle',
            'InstantaneousCurrent': 'instantaneous_current',
            'AmbientTemperature': 'ambient_temperature',
            'BatteryRunHours': 'battery_run_hours',
            'BMSBankDischargeCycle': 'bms_bank_discharge_cycle',
            'BMSAmbientTemperatureHN': 'bms_ambient_temperature_hn',
            'BMSSocLN': 'bms_soc_ln',
            'BMSStringVoltageLNH': 'bms_string_voltage_lnh',
            'BMSStringCurrentHN': 'bms_string_current_hn',
            'BMSBmsSedCommunication': 'bms_bms_sed_communication',
            'BMSCellCommunication': 'bms_cell_communication',
            'BMSCellVoltageLN': 'bms_cell_voltage_ln',
            'BMSCellVoltageNH': 'bms_cell_voltage_nh',
            'BMSCellTemperatureHN': 'bms_cell_temperature_hn',
            'BMSBuzzer': 'bms_buzzer',
            'ChargerID': 'charger_id',
            'ChargerDeviceID': 'charger_device_id',
            'ACVoltage': 'ac_voltage',
            'ACCurrent': 'ac_current',
            'Frequency': 'frequency',
            'Energy': 'energy',
            'ChargerInputMains': 'charger_input_mains',
            'ChargerInputPhase': 'charger_input_phase',
            'ChargerDCVoltageOLN': 'charger_dc_voltage_oln',
            'ChargerACVoltageULN': 'charger_ac_voltage_uln',
            'ChargerLoad': 'charger_load',
            'ChargerTrip': 'charger_trip',
            'ChargerOutputMccb': 'charger_output_mccb',
            'ChargerBatteryCondition': 'charger_battery_condition',
            'ChargerTestPushButton': 'charger_test_push_button',
            'ChargerResetPushButton': 'charger_reset_push_button',
            'ChargerAlarmSupplyFuse': 'charger_alarm_supply_fuse',
            'ChargerFilterFuse': 'charger_filter_fuse',
            'ChargerOutputFuse': 'charger_output_fuse',
            'ChargerInputFuse': 'charger_input_fuse',
            'ChargerRectifierFuse': 'charger_rectifier_fuse',
            'ServerTime': 'server_time'
        }
        
        # Rename columns
        df_renamed = df.rename(columns=column_mapping)
        
        # Handle data type conversions
        logger.info("Converting data types...")
        
        # Convert boolean columns
        boolean_columns = [
            'charger_alarm_supply_fuse', 'charger_filter_fuse', 
            'charger_output_fuse', 'charger_input_fuse'
        ]
        
        for col in boolean_columns:
            if col in df_renamed.columns:
                df_renamed[col] = df_renamed[col].astype(bool)
        
        # Convert numeric columns
        numeric_columns = [
            'packet_id', 'device_id', 'cells_connected_count', 'problem_cells',
            'cell_number', 'cell_voltage', 'cell_temperature', 'cell_specific_gravity',
            'string_voltage', 'system_peak_current_in_charge_one_cycle',
            'average_discharging_current', 'average_charging_current',
            'ah_in_for_one_charge_cycle', 'ah_out_for_one_discharge_cycle',
            'cumulative_ah_in', 'cumulative_ah_out', 'charge_time_cycle',
            'discharge_time_cycle', 'total_charging_energy', 'total_discharging_energy',
            'every_hour_avg_temp', 'cumulative_total_avg_temp_every_hour',
            'soc_latest_value_for_every_cycle', 'dod_latest_value_for_every_cycle',
            'system_peak_current_in_discharge_one_cycle', 'instantaneous_current',
            'ambient_temperature', 'battery_run_hours', 'bms_bank_discharge_cycle',
            'bms_ambient_temperature_hn', 'bms_soc_ln', 'bms_string_voltage_lnh',
            'bms_string_current_hn', 'bms_cell_voltage_ln', 'bms_cell_voltage_nh',
            'bms_cell_temperature_hn', 'ac_voltage', 'ac_current', 'frequency',
            'energy', 'charger_dc_voltage_oln', 'charger_ac_voltage_uln', 'charger_load'
        ]
        
        for col in numeric_columns:
            if col in df_renamed.columns:
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
        
        # Insert data into database
        logger.info("Inserting data into database...")
        
        # Get the columns that exist in our renamed dataframe
        db_columns = [col for col in df_renamed.columns if col in column_mapping.values()]
        
        # Prepare insert statement
        placeholders = ', '.join(['?' for _ in db_columns])
        insert_sql = f"INSERT INTO battery_data ({', '.join(db_columns)}) VALUES ({placeholders})"
        
        # Insert data row by row
        for index, row in df_renamed.iterrows():
            values = []
            for col in db_columns:
                value = row[col]
                if pd.notna(value):
                    # Convert datetime objects to string
                    if hasattr(value, 'isoformat'):
                        values.append(value.isoformat())
                    else:
                        values.append(value)
                else:
                    values.append(None)
            
            cursor.execute(insert_sql, values)
            
            if (index + 1) % 100 == 0:
                logger.info(f"Inserted {index + 1} rows...")
        
        # Commit changes
        conn.commit()
        logger.info(f"Successfully inserted {len(df_renamed)} rows into database")
        
        # Verify data
        cursor.execute("SELECT COUNT(*) FROM battery_data")
        count = cursor.fetchone()[0]
        logger.info(f"Database now contains {count} records")
        
        # Show sample data
        cursor.execute("SELECT * FROM battery_data LIMIT 3")
        sample_data = cursor.fetchall()
        logger.info("Sample data from database:")
        for i, row in enumerate(sample_data, 1):
            logger.info(f"Row {i}: {row[:10]}...")  # Show first 10 columns
        
        # Close connection
        conn.close()
        logger.info("Database creation completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

if __name__ == "__main__":
    success = create_database_from_excel()
    if success:
        print("\n✅ Database created successfully!")
        print("📊 You can now use this database with your battery monitoring system.")
        print("🔗 The database matches your actual data structure from the Excel file.")
    else:
        print("\n❌ Failed to create database. Check the logs above for details.") 