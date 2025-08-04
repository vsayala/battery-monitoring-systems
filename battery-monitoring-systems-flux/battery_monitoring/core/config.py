"""
Configuration management for battery monitoring system.

Handles loading and validation of configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    type: str = Field(default="sqlite", description="Database type")
    url: str = Field(default="sqlite:///./battery_monitoring.db", description="Database URL")
    host: Optional[str] = Field(default=None, description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: Optional[str] = Field(default=None, description="Database name")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[str] = Field(default=None, description="Database password")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")
    
    class Config:
        extra = "ignore"


class MLConfig(BaseSettings):
    """Machine learning configuration settings."""
    
    models_dir: str = Field(default="./models", description="Models directory")
    anomaly_model_path: str = Field(default="./models/anomaly_detection.pkl", description="Anomaly detection model path")
    prediction_model_path: str = Field(default="./models/cell_prediction.pkl", description="Cell prediction model path")
    forecasting_model_path: str = Field(default="./models/forecasting.pkl", description="Forecasting model path")
    
    test_size: float = Field(default=0.2, description="Test set size")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    
    # Anomaly detection settings
    contamination: float = Field(default=0.1, description="Anomaly contamination factor")
    n_estimators: int = Field(default=100, description="Number of estimators for ensemble methods")
    max_samples: Union[str, int] = Field(default="auto", description="Max samples for isolation forest")
    
    # Prediction thresholds
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold for predictions")
    dead_cell_threshold: float = Field(default=0.3, description="Threshold for dead cell classification")
    alive_cell_threshold: float = Field(default=0.7, description="Threshold for alive cell classification")
    
    # Forecasting settings
    forecast_steps: int = Field(default=25, description="Number of steps to forecast")
    lookback_window: int = Field(default=50, description="Lookback window for time series")
    min_samples: int = Field(default=100, description="Minimum samples required for training")
    
    class Config:
        extra = "ignore"


class LLMConfig(BaseSettings):
    """LLM configuration settings."""
    
    provider: str = Field(default="ollama", description="LLM provider")
    model: str = Field(default="llama2:7b", description="LLM model name")
    base_url: str = Field(default="http://localhost:11434", description="LLM base URL")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens for generation")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Prompts
    system_prompt: str = Field(
        default="You are an AI assistant for battery monitoring systems. You help analyze battery data and provide insights.",
        description="System prompt for LLM"
    )
    anomaly_analysis_prompt: str = Field(
        default="Analyze the following battery cell data for anomalies in voltage, temperature, and specific gravity:",
        description="Prompt for anomaly analysis"
    )
    prediction_analysis_prompt: str = Field(
        default="Based on the battery cell patterns, predict if the cell is likely to be dead or alive:",
        description="Prompt for prediction analysis"
    )
    forecasting_analysis_prompt: str = Field(
        default="Analyze the battery cell trends and predict future values:",
        description="Prompt for forecasting analysis"
    )
    
    class Config:
        extra = "ignore"


class MLOpsConfig(BaseSettings):
    """MLOps configuration settings."""
    
    # MLflow settings
    tracking_uri: str = Field(default="sqlite:///mlflow.db", description="MLflow tracking URI")
    experiment_name: str = Field(default="battery_monitoring", description="MLflow experiment name")
    artifact_location: str = Field(default="./mlruns", description="MLflow artifact location")
    registry_uri: Optional[str] = Field(default=None, description="MLflow registry URI")
    
    # Monitoring settings
    drift_threshold: float = Field(default=0.1, description="Data drift threshold")
    performance_threshold: float = Field(default=0.8, description="Performance threshold")
    alert_interval: int = Field(default=300, description="Alert interval in seconds")
    retention_days: int = Field(default=30, description="Data retention in days")
    
    # Metrics to track
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1_score", "mae", "rmse", "mape"],
        description="Metrics to track"
    )
    
    # Alerting settings
    email_enabled: bool = Field(default=False, description="Enable email alerts")
    slack_enabled: bool = Field(default=False, description="Enable Slack alerts")
    webhook_enabled: bool = Field(default=False, description="Enable webhook alerts")
    
    class Config:
        extra = "ignore"


class LoggingConfig(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file: str = Field(default="./logs/project_master.log", description="Main log file")
    max_size: int = Field(default=10485760, description="Max log file size in bytes")
    backup_count: int = Field(default=5, description="Number of backup files")
    
    # Module-specific log files
    data_processing_log: str = Field(default="./logs/data_processing.log", description="Data processing log")
    ml_training_log: str = Field(default="./logs/ml_training.log", description="ML training log")
    anomaly_detection_log: str = Field(default="./logs/anomaly_detection.log", description="Anomaly detection log")
    prediction_log: str = Field(default="./logs/prediction.log", description="Prediction log")
    forecasting_log: str = Field(default="./logs/forecasting.log", description="Forecasting log")
    web_app_log: str = Field(default="./logs/web_app.log", description="Web app log")
    llm_log: str = Field(default="./logs/llm.log", description="LLM log")
    mlops_log: str = Field(default="./logs/mlops.log", description="MLOps log")
    
    class Config:
        extra = "ignore"


class DataConfig(BaseSettings):
    """Data configuration settings."""
    
    input_file: str = Field(default="./data/data.xlsx", description="Input data file")
    processed_dir: str = Field(default="./data/processed", description="Processed data directory")
    features_dir: str = Field(default="./data/features", description="Features directory")
    
    # Key columns for analysis
    key_columns: List[str] = Field(
        default=["CellNumber", "CellVoltage", "CellTemperature", "CellSpecificGravity", "DeviceID", "SerialNumber", "PacketDateTime"],
        description="Key columns for analysis"
    )
    
    # Data validation ranges
    voltage_range: List[float] = Field(default=[2.0, 4.2], description="Valid voltage range")
    temperature_range: List[float] = Field(default=[0, 60], description="Valid temperature range")
    specific_gravity_range: List[float] = Field(default=[1.0, 1.3], description="Valid specific gravity range")
    max_missing_percentage: float = Field(default=0.1, description="Maximum missing data percentage")
    
    class Config:
        extra = "ignore"


class WebAppConfig(BaseSettings):
    """Web application configuration settings."""
    
    # Frontend settings
    frontend_port: int = Field(default=3000, description="Frontend port")
    api_url: str = Field(default="http://localhost:8000", description="API URL")
    websocket_url: str = Field(default="ws://localhost:8001", description="WebSocket URL")
    
    # Backend API settings
    api_title: str = Field(default="Battery Monitoring API", description="API title")
    api_description: str = Field(default="API for battery monitoring system with ML/LLM capabilities", description="API description")
    api_version: str = Field(default="1.0.0", description="API version")
    docs_url: str = Field(default="/docs", description="API docs URL")
    redoc_url: str = Field(default="/redoc", description="API redoc URL")
    
    # WebSocket settings
    ping_interval: int = Field(default=20, description="WebSocket ping interval")
    ping_timeout: int = Field(default=20, description="WebSocket ping timeout")
    max_connections: int = Field(default=100, description="Maximum WebSocket connections")
    
    class Config:
        extra = "ignore"


class SecurityConfig(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(default="your-secret-key-change-in-production", description="Secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry in minutes")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="CORS origins"
    )
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="CORS methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS headers")
    
    # Rate limiting
    requests_per_minute: int = Field(default=100, description="Requests per minute")
    burst_size: int = Field(default=20, description="Burst size for rate limiting")
    
    class Config:
        extra = "ignore"


class PerformanceConfig(BaseSettings):
    """Performance configuration settings."""
    
    # Timeouts
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    model_inference_timeout: int = Field(default=60, description="Model inference timeout in seconds")
    data_processing_timeout: int = Field(default=300, description="Data processing timeout in seconds")
    
    # Batch processing
    batch_size: int = Field(default=1000, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum workers for parallel processing")
    
    # Caching
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")
    
    class Config:
        extra = "ignore"


class Config(BaseSettings):
    """Main configuration class for the battery monitoring system."""
    
    # Application settings
    app_name: str = Field(default="Battery Monitoring System", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=True, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Host address")
    port: int = Field(default=8000, description="Port number")
    websocket_port: int = Field(default=8001, description="WebSocket port")
    reload: bool = Field(default=True, description="Auto-reload")
    
    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    ml: MLConfig = Field(default_factory=MLConfig, description="Machine learning configuration")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig, description="MLOps configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    data: DataConfig = Field(default_factory=DataConfig, description="Data configuration")
    web_app: WebAppConfig = Field(default_factory=WebAppConfig, description="Web app configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields
    
    @validator("database", "ml", "llm", "mlops", "logging", "data", "web_app", "security", "performance", pre=True)
    def validate_config_sections(cls, v, values, **kwargs):
        """Validate configuration sections."""
        if isinstance(v, dict):
            return v
        return v
    
    @classmethod
    def load_from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            # Create config instance with default values
            config_instance = cls()
            
            # Update with YAML data
            if config_data:
                # Handle nested configuration structure
                for section, section_data in config_data.items():
                    if hasattr(config_instance, section) and isinstance(section_data, dict):
                        # Update the section with YAML data
                        section_config = getattr(config_instance, section)
                        for key, value in section_data.items():
                            if hasattr(section_config, key):
                                setattr(section_config, key, value)
                    elif hasattr(config_instance, section):
                        # Direct field assignment
                        setattr(config_instance, section, section_data)
            
            return config_instance
        
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def save_to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        
        try:
            # Convert to dictionary
            config_dict = self.dict()
            
            # Restructure to nested format
            nested_config = {}
            for key, value in config_dict.items():
                if "_" in key:
                    section, subkey = key.split("_", 1)
                    if section not in nested_config:
                        nested_config[section] = {}
                    nested_config[section][subkey] = value
                else:
                    nested_config[key] = value
            
            # Ensure directories exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(nested_config, f, default_flow_style=False, indent=2)
        
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")
    
    def get_model_path(self, model_type: str) -> str:
        """Get model path for specific model type."""
        model_paths = {
            "anomaly": self.ml.anomaly_model_path,
            "prediction": self.ml.prediction_model_path,
            "forecasting": self.ml.forecasting_model_path,
        }
        
        if model_type not in model_paths:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_paths[model_type]
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            Path(self.ml.models_dir),
            Path(self.logging.file).parent,
            Path(self.data.processed_dir),
            Path(self.data.features_dir),
            Path(self.mlops.artifact_location),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    
    if _config is None:
        # Try to load from environment variable first
        config_path = os.getenv("BATTERY_CONFIG_PATH", "config_local.yaml")
        _config = Config.load_from_yaml(config_path)
        _config.ensure_directories()
    
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config
    _config.ensure_directories() 