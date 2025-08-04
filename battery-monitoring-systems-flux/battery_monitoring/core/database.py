"""
Database management for battery monitoring system.

Provides database connection, models, and operations for storing and retrieving
battery monitoring data, ML model metadata, and system logs.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, Boolean, create_engine,
    MetaData, Table, select, insert, update, delete, and_, or_, func, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .config import get_config
from .exceptions import DatabaseError, DatabaseConnectionError, DatabaseQueryError
from .logger import get_logger

# Create base class for declarative models
Base = declarative_base()


class BatteryData(Base):
    """Model for battery monitoring data with comprehensive fields."""
    
    __tablename__ = "battery_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    packet_id = Column(Integer, nullable=True)
    start_packet = Column(String(10), nullable=True)
    protocol_version = Column(Float, nullable=True)
    data_identifier = Column(String(10), nullable=True)
    site_id = Column(String(50), nullable=True)
    time = Column(String(20), nullable=True)
    date = Column(String(20), nullable=True)
    packet_datetime = Column(String(50), nullable=True)
    device_id = Column(Integer, nullable=True)
    bms_manufacturer_id = Column(String(50), nullable=True)
    serial_number = Column(String(50), nullable=True)
    installation_date = Column(String(50), nullable=True)
    cells_connected_count = Column(Integer, nullable=True)
    problem_cells = Column(Integer, nullable=True)
    cell_number = Column(Integer, nullable=True)
    cell_voltage = Column(Float, nullable=True)
    cell_temperature = Column(Float, nullable=True)
    cell_specific_gravity = Column(Float, nullable=True)
    cell_server_time = Column(String(50), nullable=True)
    string_voltage = Column(Float, nullable=True)
    system_peak_current_in_charge_one_cycle = Column(Float, nullable=True)
    average_discharging_current = Column(Float, nullable=True)
    average_charging_current = Column(Float, nullable=True)
    ah_in_for_one_charge_cycle = Column(Float, nullable=True)
    ah_out_for_one_discharge_cycle = Column(Float, nullable=True)
    cumulative_ah_in = Column(Float, nullable=True)
    cumulative_ah_out = Column(Float, nullable=True)
    charge_time_cycle = Column(Float, nullable=True)
    discharge_time_cycle = Column(Float, nullable=True)
    total_charging_energy = Column(Float, nullable=True)
    total_discharging_energy = Column(Float, nullable=True)
    every_hour_avg_temp = Column(Float, nullable=True)
    cumulative_total_avg_temp_every_hour = Column(Float, nullable=True)
    charge_or_discharge_cycle = Column(String(20), nullable=True)
    soc_latest_value_for_every_cycle = Column(Float, nullable=True)
    dod_latest_value_for_every_cycle = Column(Float, nullable=True)
    system_peak_current_in_discharge_one_cycle = Column(Float, nullable=True)
    instantaneous_current = Column(Float, nullable=True)
    ambient_temperature = Column(Float, nullable=True)
    battery_run_hours = Column(Float, nullable=True)
    bms_bank_discharge_cycle = Column(Integer, nullable=True)
    bms_ambient_temperature_hn = Column(Float, nullable=True)
    bms_soc_ln = Column(Float, nullable=True)
    bms_string_voltage_lnh = Column(Float, nullable=True)
    bms_string_current_hn = Column(Float, nullable=True)
    bms_bms_sed_communication = Column(String(50), nullable=True)
    bms_cell_communication = Column(String(50), nullable=True)
    bms_cell_voltage_ln = Column(Float, nullable=True)
    bms_cell_voltage_nh = Column(Float, nullable=True)
    bms_cell_temperature_hn = Column(Float, nullable=True)
    bms_buzzer = Column(String(20), nullable=True)
    charger_id = Column(String(50), nullable=True)
    charger_device_id = Column(String(50), nullable=True)
    ac_voltage = Column(Float, nullable=True)
    ac_current = Column(Float, nullable=True)
    frequency = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    charger_input_mains = Column(String(50), nullable=True)
    charger_input_phase = Column(String(50), nullable=True)
    charger_dc_voltage_oln = Column(Float, nullable=True)
    charger_ac_voltage_uln = Column(Float, nullable=True)
    charger_load = Column(Float, nullable=True)
    charger_trip = Column(String(50), nullable=True)
    charger_output_mccb = Column(String(50), nullable=True)
    charger_battery_condition = Column(String(50), nullable=True)
    charger_test_push_button = Column(String(50), nullable=True)
    charger_reset_push_button = Column(String(50), nullable=True)
    charger_alarm_supply_fuse = Column(Boolean, nullable=True)
    charger_filter_fuse = Column(Boolean, nullable=True)
    charger_output_fuse = Column(Boolean, nullable=True)
    charger_input_fuse = Column(Boolean, nullable=True)
    charger_rectifier_fuse = Column(String(50), nullable=True)
    server_time = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<BatteryData(id={self.id}, device_id={self.device_id}, cell_number={self.cell_number})>"


class AnomalyDetection(Base):
    """Model for anomaly detection results."""
    
    __tablename__ = "anomaly_detection"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    battery_data_id = Column(Integer, nullable=False)
    device_id = Column(Integer, nullable=False)
    cell_number = Column(Integer, nullable=False)
    voltage_anomaly = Column(Boolean, nullable=False)
    temperature_anomaly = Column(Boolean, nullable=False)
    specific_gravity_anomaly = Column(Boolean, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AnomalyDetection(id={self.id}, device_id={self.device_id}, cell_number={self.cell_number})>"


class CellPrediction(Base):
    """Model for cell health predictions."""
    
    __tablename__ = "cell_prediction"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, nullable=False)
    cell_number = Column(Integer, nullable=False)
    prediction = Column(String(20), nullable=False)  # 'alive' or 'dead'
    confidence = Column(Float, nullable=False)
    features_used = Column(Text, nullable=True)  # JSON string of features
    predicted_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<CellPrediction(id={self.id}, device_id={self.device_id}, cell_number={self.cell_number})>"


class Forecasting(Base):
    """Model for forecasting results."""
    
    __tablename__ = "forecasting"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, nullable=False)
    cell_number = Column(Integer, nullable=False)
    forecast_type = Column(String(20), nullable=False)  # 'voltage', 'temperature', 'specific_gravity'
    forecast_steps = Column(Integer, nullable=False)
    forecast_values = Column(Text, nullable=False)  # JSON string of forecasted values
    forecast_dates = Column(Text, nullable=False)  # JSON string of forecast dates
    confidence_intervals = Column(Text, nullable=True)  # JSON string of confidence intervals
    forecasted_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Forecasting(id={self.id}, device_id={self.device_id}, cell_number={self.cell_number})>"


class ModelMetadata(Base):
    """Model for ML model metadata."""
    
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)  # 'anomaly', 'prediction', 'forecasting'
    model_version = Column(String(20), nullable=False)
    model_path = Column(String(255), nullable=False)
    training_date = Column(DateTime, nullable=False)
    performance_metrics = Column(Text, nullable=True)  # JSON string of metrics
    hyperparameters = Column(Text, nullable=True)  # JSON string of hyperparameters
    feature_columns = Column(Text, nullable=True)  # JSON string of feature columns
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelMetadata(id={self.id}, model_type={self.model_type}, version={self.model_version})>"


class SystemMetrics(Base):
    """Model for system performance metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, metric_name={self.metric_name}, value={self.metric_value})>"


class DatabaseManager:
    """Database manager for battery monitoring system."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger("database")
        self.engine = None
        self.SessionLocal = None
        self._setup_database()
    
    def _setup_database(self) -> None:
        """Setup database connection and create tables."""
        try:
            # Create database URL
            if self.config.database.type == "sqlite":
                database_url = self.config.database.url
                # Use StaticPool for SQLite to avoid threading issues
                self.engine = create_engine(
                    database_url,
                    poolclass=StaticPool,
                    connect_args={"check_same_thread": False}
                )
            else:
                # PostgreSQL or other databases
                database_url = self.config.database.url
                self.engine = create_engine(
                    database_url,
                    pool_size=self.config.database.pool_size,
                    max_overflow=self.config.database.max_overflow,
                    pool_timeout=self.config.database.pool_timeout,
                    pool_recycle=self.config.database.pool_recycle
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Check if tables already exist, if not create them
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            
            if 'battery_data' not in existing_tables:
                self.logger.info("Creating database tables...")
                Base.metadata.create_all(bind=self.engine)
            else:
                self.logger.info("Database tables already exist, skipping creation")
            
            self.logger.info("Database setup completed successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to setup database: {str(e)}")
            raise DatabaseConnectionError(f"Failed to setup database: {str(e)}")
    
    def get_session(self) -> Session:
        """Get a database session."""
        if not self.SessionLocal:
            raise DatabaseConnectionError("Database not initialized")
        return self.SessionLocal()
    
    def close_session(self, session: Session) -> None:
        """Close a database session."""
        try:
            session.close()
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
    
    def insert_battery_data(self, data: pd.DataFrame) -> int:
        """Insert battery data into database."""
        session = self.get_session()
        inserted_count = 0
        
        try:
            for _, row in data.iterrows():
                battery_data = BatteryData(
                    packet_id=row.get('packet_id', row.get('PacketID', 0)),
                    start_packet=row.get('start_packet', row.get('StartPacket', '')),
                    protocol_version=row.get('protocol_version', row.get('ProtocolVersion', None)),
                    data_identifier=row.get('data_identifier', row.get('DataIdentifier', '')),
                    site_id=row.get('site_id', row.get('SiteID', '')),
                    time=row.get('time', row.get('Time', '')),
                    date=row.get('date', row.get('Date', '')),
                    packet_datetime=row.get('packet_datetime', row.get('PacketDateTime', '')),
                    device_id=row.get('device_id', row.get('DeviceID', 0)),
                    bms_manufacturer_id=row.get('bms_manufacturer_id', row.get('BMSManufacturerID', '')),
                    serial_number=row.get('serial_number', row.get('SerialNumber', '')),
                    installation_date=row.get('installation_date', row.get('InstallationDate', '')),
                    cells_connected_count=row.get('cells_connected_count', row.get('CellsConnectedCount', None)),
                    problem_cells=row.get('problem_cells', row.get('ProblemCells', None)),
                    cell_number=row.get('cell_number', row.get('CellNumber', 0)),
                    cell_voltage=row.get('cell_voltage', row.get('CellVoltage', 0.0)),
                    cell_temperature=row.get('cell_temperature', row.get('CellTemperature', 0.0)),
                    cell_specific_gravity=row.get('cell_specific_gravity', row.get('CellSpecificGravity', 0.0)),
                    cell_server_time=row.get('cell_server_time', row.get('CellServerTime', '')),
                    string_voltage=row.get('string_voltage', row.get('StringVoltage', None)),
                    system_peak_current_in_charge_one_cycle=row.get('system_peak_current_in_charge_one_cycle', row.get('SystemPeakCurrentInChargeOneCycle', None)),
                    average_discharging_current=row.get('average_discharging_current', row.get('AverageDischargingCurrent', None)),
                    average_charging_current=row.get('average_charging_current', row.get('AverageChargingCurrent', None)),
                    ah_in_for_one_charge_cycle=row.get('ah_in_for_one_charge_cycle', row.get('AhInForOneChargeCycle', None)),
                    ah_out_for_one_discharge_cycle=row.get('ah_out_for_one_discharge_cycle', row.get('AhOutForOneDischargeCycle', None)),
                    cumulative_ah_in=row.get('cumulative_ah_in', row.get('CumulativeAHIn', None)),
                    cumulative_ah_out=row.get('cumulative_ah_out', row.get('CumulativeAHOut', None)),
                    charge_time_cycle=row.get('charge_time_cycle', row.get('ChargeTimeCycle', None)),
                    discharge_time_cycle=row.get('discharge_time_cycle', row.get('DischargeTimeCycle', None)),
                    total_charging_energy=row.get('total_charging_energy', row.get('TotalChargingEnergy', None)),
                    total_discharging_energy=row.get('total_discharging_energy', row.get('TotalDischargingEnergy', None)),
                    every_hour_avg_temp=row.get('every_hour_avg_temp', row.get('EveryHourAvgTemp', None)),
                    cumulative_total_avg_temp_every_hour=row.get('cumulative_total_avg_temp_every_hour', row.get('CumulativeTotalAvgTempEveryHour', None)),
                    charge_or_discharge_cycle=row.get('charge_or_discharge_cycle', row.get('ChargeOrDischargeCycle', '')),
                    soc_latest_value_for_every_cycle=row.get('soc_latest_value_for_every_cycle', row.get('SocLatestValueForEveryCycle', None)),
                    dod_latest_value_for_every_cycle=row.get('dod_latest_value_for_every_cycle', row.get('DodLatestValueForEveryCycle', None)),
                    system_peak_current_in_discharge_one_cycle=row.get('system_peak_current_in_discharge_one_cycle', row.get('SystemPeakCurrentInDischargeOneCycle', None)),
                    instantaneous_current=row.get('instantaneous_current', row.get('InstantaneousCurrent', None)),
                    ambient_temperature=row.get('ambient_temperature', row.get('AmbientTemperature', None)),
                    battery_run_hours=row.get('battery_run_hours', row.get('BatteryRunHours', None)),
                    bms_bank_discharge_cycle=row.get('bms_bank_discharge_cycle', row.get('BMSBankDischargeCycle', None)),
                    bms_ambient_temperature_hn=row.get('bms_ambient_temperature_hn', row.get('BMSAmbientTemperatureHN', None)),
                    bms_soc_ln=row.get('bms_soc_ln', row.get('BMSSocLN', None)),
                    bms_string_voltage_lnh=row.get('bms_string_voltage_lnh', row.get('BMSStringVoltageLNH', None)),
                    bms_string_current_hn=row.get('bms_string_current_hn', row.get('BMSStringCurrentHN', None)),
                    bms_bms_sed_communication=row.get('bms_bms_sed_communication', row.get('BMSBmsSedCommunication', '')),
                    bms_cell_communication=row.get('bms_cell_communication', row.get('BMSCellCommunication', '')),
                    bms_cell_voltage_ln=row.get('bms_cell_voltage_ln', row.get('BMSCellVoltageLN', None)),
                    bms_cell_voltage_nh=row.get('bms_cell_voltage_nh', row.get('BMSCellVoltageNH', None)),
                    bms_cell_temperature_hn=row.get('bms_cell_temperature_hn', row.get('BMSCellTemperatureHN', None)),
                    bms_buzzer=row.get('bms_buzzer', row.get('BMSBuzzer', '')),
                    charger_id=row.get('charger_id', row.get('ChargerID', '')),
                    charger_device_id=row.get('charger_device_id', row.get('ChargerDeviceID', '')),
                    ac_voltage=row.get('ac_voltage', row.get('ACVoltage', None)),
                    ac_current=row.get('ac_current', row.get('ACCurrent', None)),
                    frequency=row.get('frequency', row.get('Frequency', None)),
                    energy=row.get('energy', row.get('Energy', None)),
                    charger_input_mains=row.get('charger_input_mains', row.get('ChargerInputMains', '')),
                    charger_input_phase=row.get('charger_input_phase', row.get('ChargerInputPhase', '')),
                    charger_dc_voltage_oln=row.get('charger_dc_voltage_oln', row.get('ChargerDCVoltageOLN', None)),
                    charger_ac_voltage_uln=row.get('charger_ac_voltage_uln', row.get('ChargerACVoltageULN', None)),
                    charger_load=row.get('charger_load', row.get('ChargerLoad', None)),
                    charger_trip=row.get('charger_trip', row.get('ChargerTrip', '')),
                    charger_output_mccb=row.get('charger_output_mccb', row.get('ChargerOutputMccb', '')),
                    charger_battery_condition=row.get('charger_battery_condition', row.get('ChargerBatteryCondition', '')),
                    charger_test_push_button=row.get('charger_test_push_button', row.get('ChargerTestPushButton', '')),
                    charger_reset_push_button=row.get('charger_reset_push_button', row.get('ChargerResetPushButton', '')),
                    charger_alarm_supply_fuse=row.get('charger_alarm_supply_fuse', row.get('ChargerAlarmSupplyFuse', False)),
                    charger_filter_fuse=row.get('charger_filter_fuse', row.get('ChargerFilterFuse', False)),
                    charger_output_fuse=row.get('charger_output_fuse', row.get('ChargerOutputFuse', False)),
                    charger_input_fuse=row.get('charger_input_fuse', row.get('ChargerInputFuse', False)),
                    charger_rectifier_fuse=row.get('charger_rectifier_fuse', row.get('ChargerRectifierFuse', '')),
                    server_time=row.get('server_time', row.get('ServerTime', ''))
                )
                session.add(battery_data)
                inserted_count += 1
            
            session.commit()
            self.logger.info(f"Inserted {inserted_count} battery data records")
            return inserted_count
        
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to insert battery data: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert battery data: {str(e)}")
        finally:
            self.close_session(session)
    
    def get_battery_data(
        self,
        device_id: Optional[int] = None,
        cell_number: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get battery data from database."""
        session = self.get_session()
        
        try:
            query = session.query(BatteryData)
            
            # Apply filters
            if device_id is not None:
                query = query.filter(BatteryData.device_id == device_id)
            
            if cell_number is not None:
                query = query.filter(BatteryData.cell_number == cell_number)
            
            if start_date is not None:
                query = query.filter(BatteryData.packet_datetime >= start_date)
            
            if end_date is not None:
                query = query.filter(BatteryData.packet_datetime <= end_date)
            
            # Order by timestamp
            query = query.order_by(BatteryData.packet_datetime)
            
            # Apply limit
            if limit is not None:
                query = query.limit(limit)
            
            # Execute query
            results = query.all()
            
            # Convert to DataFrame with all columns
            data = []
            for result in results:
                data.append({
                    'id': result.id,
                    'packet_id': result.packet_id,
                    'start_packet': result.start_packet,
                    'protocol_version': result.protocol_version,
                    'data_identifier': result.data_identifier,
                    'site_id': result.site_id,
                    'time': result.time,
                    'date': result.date,
                    'packet_datetime': result.packet_datetime,
                    'device_id': result.device_id,
                    'bms_manufacturer_id': result.bms_manufacturer_id,
                    'serial_number': result.serial_number,
                    'installation_date': result.installation_date,
                    'cells_connected_count': result.cells_connected_count,
                    'problem_cells': result.problem_cells,
                    'cell_number': result.cell_number,
                    'cell_voltage': result.cell_voltage,
                    'cell_temperature': result.cell_temperature,
                    'cell_specific_gravity': result.cell_specific_gravity,
                    'cell_server_time': result.cell_server_time,
                    'string_voltage': result.string_voltage,
                    'system_peak_current_in_charge_one_cycle': result.system_peak_current_in_charge_one_cycle,
                    'average_discharging_current': result.average_discharging_current,
                    'average_charging_current': result.average_charging_current,
                    'ah_in_for_one_charge_cycle': result.ah_in_for_one_charge_cycle,
                    'ah_out_for_one_discharge_cycle': result.ah_out_for_one_discharge_cycle,
                    'cumulative_ah_in': result.cumulative_ah_in,
                    'cumulative_ah_out': result.cumulative_ah_out,
                    'charge_time_cycle': result.charge_time_cycle,
                    'discharge_time_cycle': result.discharge_time_cycle,
                    'total_charging_energy': result.total_charging_energy,
                    'total_discharging_energy': result.total_discharging_energy,
                    'every_hour_avg_temp': result.every_hour_avg_temp,
                    'cumulative_total_avg_temp_every_hour': result.cumulative_total_avg_temp_every_hour,
                    'charge_or_discharge_cycle': result.charge_or_discharge_cycle,
                    'soc_latest_value_for_every_cycle': result.soc_latest_value_for_every_cycle,
                    'dod_latest_value_for_every_cycle': result.dod_latest_value_for_every_cycle,
                    'system_peak_current_in_discharge_one_cycle': result.system_peak_current_in_discharge_one_cycle,
                    'instantaneous_current': result.instantaneous_current,
                    'ambient_temperature': result.ambient_temperature,
                    'battery_run_hours': result.battery_run_hours,
                    'bms_bank_discharge_cycle': result.bms_bank_discharge_cycle,
                    'bms_ambient_temperature_hn': result.bms_ambient_temperature_hn,
                    'bms_soc_ln': result.bms_soc_ln,
                    'bms_string_voltage_lnh': result.bms_string_voltage_lnh,
                    'bms_string_current_hn': result.bms_string_current_hn,
                    'bms_bms_sed_communication': result.bms_bms_sed_communication,
                    'bms_cell_communication': result.bms_cell_communication,
                    'bms_cell_voltage_ln': result.bms_cell_voltage_ln,
                    'bms_cell_voltage_nh': result.bms_cell_voltage_nh,
                    'bms_cell_temperature_hn': result.bms_cell_temperature_hn,
                    'bms_buzzer': result.bms_buzzer,
                    'charger_id': result.charger_id,
                    'charger_device_id': result.charger_device_id,
                    'ac_voltage': result.ac_voltage,
                    'ac_current': result.ac_current,
                    'frequency': result.frequency,
                    'energy': result.energy,
                    'charger_input_mains': result.charger_input_mains,
                    'charger_input_phase': result.charger_input_phase,
                    'charger_dc_voltage_oln': result.charger_dc_voltage_oln,
                    'charger_ac_voltage_uln': result.charger_ac_voltage_uln,
                    'charger_load': result.charger_load,
                    'charger_trip': result.charger_trip,
                    'charger_output_mccb': result.charger_output_mccb,
                    'charger_battery_condition': result.charger_battery_condition,
                    'charger_test_push_button': result.charger_test_push_button,
                    'charger_reset_push_button': result.charger_reset_push_button,
                    'charger_alarm_supply_fuse': result.charger_alarm_supply_fuse,
                    'charger_filter_fuse': result.charger_filter_fuse,
                    'charger_output_fuse': result.charger_output_fuse,
                    'charger_input_fuse': result.charger_input_fuse,
                    'charger_rectifier_fuse': result.charger_rectifier_fuse,
                    'server_time': result.server_time,
                    'created_at': result.created_at
                })
            
            df = pd.DataFrame(data)
            self.logger.info(f"Retrieved {len(df)} battery data records")
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to get battery data: {str(e)}")
            raise DatabaseQueryError(f"Failed to get battery data: {str(e)}")
        finally:
            self.close_session(session)
    
    def insert_anomaly_detection(self, anomaly_data: List[Dict]) -> int:
        """Insert anomaly detection results."""
        session = self.get_session()
        inserted_count = 0
        
        try:
            for data in anomaly_data:
                anomaly = AnomalyDetection(
                    battery_data_id=data.get('battery_data_id'),
                    device_id=data.get('device_id'),
                    cell_number=data.get('cell_number'),
                    voltage_anomaly=data.get('voltage_anomaly', False),
                    temperature_anomaly=data.get('temperature_anomaly', False),
                    specific_gravity_anomaly=data.get('specific_gravity_anomaly', False),
                    anomaly_score=data.get('anomaly_score', 0.0)
                )
                session.add(anomaly)
                inserted_count += 1
            
            session.commit()
            self.logger.info(f"Inserted {inserted_count} anomaly detection records")
            return inserted_count
        
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to insert anomaly detection data: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert anomaly detection data: {str(e)}")
        finally:
            self.close_session(session)
    
    def insert_cell_prediction(self, prediction_data: Dict) -> int:
        """Insert cell prediction result."""
        session = self.get_session()
        
        try:
            prediction = CellPrediction(
                device_id=prediction_data.get('device_id'),
                cell_number=prediction_data.get('cell_number'),
                prediction=prediction_data.get('prediction'),
                confidence=prediction_data.get('confidence'),
                features_used=prediction_data.get('features_used')
            )
            session.add(prediction)
            session.commit()
            
            self.logger.info(f"Inserted cell prediction for device {prediction_data.get('device_id')}, cell {prediction_data.get('cell_number')}")
            return prediction.id
        
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to insert cell prediction: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert cell prediction: {str(e)}")
        finally:
            self.close_session(session)
    
    def insert_forecasting(self, forecast_data: Dict) -> int:
        """Insert forecasting result."""
        session = self.get_session()
        
        try:
            import json
            
            forecasting = Forecasting(
                device_id=forecast_data.get('device_id'),
                cell_number=forecast_data.get('cell_number'),
                forecast_type=forecast_data.get('forecast_type'),
                forecast_steps=forecast_data.get('forecast_steps'),
                forecast_values=json.dumps(forecast_data.get('forecast_values', [])),
                forecast_dates=json.dumps(forecast_data.get('forecast_dates', [])),
                confidence_intervals=json.dumps(forecast_data.get('confidence_intervals', []))
            )
            session.add(forecasting)
            session.commit()
            
            self.logger.info(f"Inserted forecasting for device {forecast_data.get('device_id')}, cell {forecast_data.get('cell_number')}")
            return forecasting.id
        
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to insert forecasting: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert forecasting: {str(e)}")
        finally:
            self.close_session(session)
    
    def save_model_metadata(self, metadata: Dict) -> int:
        """Save model metadata."""
        session = self.get_session()
        
        try:
            import json
            
            model_metadata = ModelMetadata(
                model_type=metadata.get('model_type'),
                model_version=metadata.get('model_version'),
                model_path=metadata.get('model_path'),
                training_date=metadata.get('training_date'),
                performance_metrics=json.dumps(metadata.get('performance_metrics', {})),
                hyperparameters=json.dumps(metadata.get('hyperparameters', {})),
                feature_columns=json.dumps(metadata.get('feature_columns', []))
            )
            session.add(model_metadata)
            session.commit()
            
            self.logger.info(f"Saved model metadata for {metadata.get('model_type')} v{metadata.get('model_version')}")
            return model_metadata.id
        
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save model metadata: {str(e)}")
            raise DatabaseQueryError(f"Failed to save model metadata: {str(e)}")
        finally:
            self.close_session(session)
    
    def get_latest_model(self, model_type: str) -> Optional[Dict]:
        """Get the latest model metadata for a given type."""
        session = self.get_session()
        
        try:
            import json
            
            model = session.query(ModelMetadata).filter(
                and_(
                    ModelMetadata.model_type == model_type,
                    ModelMetadata.is_active == True
                )
            ).order_by(ModelMetadata.training_date.desc()).first()
            
            if model:
                return {
                    'id': model.id,
                    'model_type': model.model_type,
                    'model_version': model.model_version,
                    'model_path': model.model_path,
                    'training_date': model.training_date,
                    'performance_metrics': json.loads(model.performance_metrics) if model.performance_metrics else {},
                    'hyperparameters': json.loads(model.hyperparameters) if model.hyperparameters else {},
                    'feature_columns': json.loads(model.feature_columns) if model.feature_columns else []
                }
            
            return None
        
        except Exception as e:
            self.logger.error(f"Failed to get latest model: {str(e)}")
            raise DatabaseQueryError(f"Failed to get latest model: {str(e)}")
        finally:
            self.close_session(session)
    
    def save_system_metric(self, metric_name: str, metric_value: float, metric_unit: str = "") -> int:
        """Save system metric."""
        session = self.get_session()
        
        try:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit
            )
            session.add(metric)
            session.commit()
            
            return metric.id
        
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save system metric: {str(e)}")
            raise DatabaseQueryError(f"Failed to save system metric: {str(e)}")
        finally:
            self.close_session(session)
    
    def get_system_metrics(
        self,
        metric_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get system metrics."""
        session = self.get_session()
        
        try:
            query = session.query(SystemMetrics)
            
            if metric_name:
                query = query.filter(SystemMetrics.metric_name == metric_name)
            
            if start_date:
                query = query.filter(SystemMetrics.timestamp >= start_date)
            
            if end_date:
                query = query.filter(SystemMetrics.timestamp <= end_date)
            
            query = query.order_by(SystemMetrics.timestamp.desc())
            
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            
            data = []
            for result in results:
                data.append({
                    'id': result.id,
                    'metric_name': result.metric_name,
                    'metric_value': result.metric_value,
                    'metric_unit': result.metric_unit,
                    'timestamp': result.timestamp
                })
            
            df = pd.DataFrame(data)
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
            raise DatabaseQueryError(f"Failed to get system metrics: {str(e)}")
        finally:
            self.close_session(session)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        session = self.get_session()
        
        try:
            stats = {}
            
            # Count records in each table
            stats['battery_data_count'] = session.query(func.count(BatteryData.id)).scalar()
            stats['anomaly_detection_count'] = session.query(func.count(AnomalyDetection.id)).scalar()
            stats['cell_prediction_count'] = session.query(func.count(CellPrediction.id)).scalar()
            stats['forecasting_count'] = session.query(func.count(Forecasting.id)).scalar()
            stats['model_metadata_count'] = session.query(func.count(ModelMetadata.id)).scalar()
            stats['system_metrics_count'] = session.query(func.count(SystemMetrics.id)).scalar()
            
            # Get latest timestamps
            latest_battery = session.query(func.max(BatteryData.packet_datetime)).scalar()
            stats['latest_battery_data'] = str(latest_battery) if latest_battery else None
            
            return stats
        
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            raise DatabaseQueryError(f"Failed to get database stats: {str(e)}")
        finally:
            self.close_session(session)
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old data from database."""
        session = self.get_session()
        deleted_count = 0
        
        try:
            cutoff_date = datetime.utcnow() - pd.Timedelta(days=days_to_keep)
            
            # Delete old battery data
            deleted_battery = session.query(BatteryData).filter(
                BatteryData.packet_datetime < cutoff_date
            ).delete()
            deleted_count += deleted_battery
            
            # Delete old system metrics
            deleted_metrics = session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_date
            ).delete()
            deleted_count += deleted_metrics
            
            session.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} old records")
            return deleted_count
        
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
            raise DatabaseQueryError(f"Failed to cleanup old data: {str(e)}")
        finally:
            self.close_session(session)


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
    
    return _db_manager


def close_database_manager() -> None:
    """Close global database manager."""
    global _db_manager
    
    if _db_manager and _db_manager.engine:
        _db_manager.engine.dispose()
        _db_manager = None 