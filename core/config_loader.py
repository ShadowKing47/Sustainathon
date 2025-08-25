""
Configuration loader for the Energy Monitoring System.
"""
import os
import yaml
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigLoader:
    """Handles loading and accessing configuration settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_path: Optional path to the config directory
        """
        # Set default paths
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = Path(config_path) if config_path else self.base_dir / 'config'
        self.config_file = self.config_dir / 'config.yaml'
        self.schema_file = self.config_dir / 'schema.yaml'
        
        # Load configurations
        self.config = self._load_config()
        self.schema = self._load_schema()
        
        # Update with environment variables
        self._update_from_env()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the main configuration file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            logger.warning(f"Config file not found: {self.config_file}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load the data schema file."""
        try:
            if self.schema_file.exists():
                with open(self.schema_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            logger.warning(f"Schema file not found: {self.schema_file}")
            return {}
        except Exception as e:
            logger.error(f"Error loading schema file: {e}")
            return {}
    
    def _update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # Database settings
        if 'DB_HOST' in os.environ:
            self.config['database'] = self.config.get('database', {})
            self.config['database']['host'] = os.environ['DB_HOST']
        
        if 'DB_PORT' in os.environ:
            self.config['database'] = self.config.get('database', {})
            self.config['database']['port'] = int(os.environ['DB_PORT'])
        
        if 'DB_NAME' in os.environ:
            self.config['database'] = self.config.get('database', {})
            self.config['database']['name'] = os.environ['DB_NAME']
        
        if 'DB_USER' in os.environ:
            self.config['database'] = self.config.get('database', {})
            self.config['database']['user'] = os.environ['DB_USER']
        
        if 'DB_PASSWORD' in os.environ:
            self.config['database'] = self.config.get('database', {})
            self.config['database']['password'] = os.environ['DB_PASSWORD']
        
        # API keys
        if 'OPENWEATHER_API_KEY' in os.environ:
            self.config['api_keys'] = self.config.get('api_keys', {})
            self.config['api_keys']['openweather'] = os.environ['OPENWEATHER_API_KEY']
        
        # Paths
        if 'DATA_DIR' in os.environ:
            self.config['paths'] = self.config.get('paths', {})
            self.config['paths']['data'] = os.environ['DATA_DIR']
        
        if 'MODEL_DIR' in os.environ:
            self.config['paths'] = self.config.get('paths', {})
            self.config['paths']['models'] = os.environ['MODEL_DIR']
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot notation key.
        
        Args:
            key: Dot notation key (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            The configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_schema(self, dataset: str) -> Dict[str, Any]:
        """
        Get the schema for a specific dataset.
        
        Args:
            dataset: Name of the dataset
            
        Returns:
            Dictionary containing the dataset schema
        """
        return self.schema.get(dataset, {})
    
    def get_database_uri(self) -> str:
        """
        Get the database connection URI.
        
        Returns:
            Database connection string
        """
        db_config = self.get('database', {})
        
        if 'uri' in db_config:
            return db_config['uri']
        
        # Construct URI from components
        user = db_config.get('user', '')
        password = f":{db_config.get('password', '')}" if db_config.get('password') else ''
        host = db_config.get('host', 'localhost')
        port = f":{db_config.get('port', '')}" if db_config.get('port') else ''
        name = db_config.get('name', 'energy_monitor')
        
        return f"postgresql://{user}{password}@{host}{port}/{name}"
    
    def get_data_path(self, *subpaths: str) -> str:
        """
        Get a path relative to the data directory.
        
        Args:
            *subpaths: Path components to append
            
        Returns:
            Absolute path
        """
        base_path = self.get('paths.data', str(self.base_dir / 'data'))
        return str(Path(base_path).joinpath(*subpaths).resolve())
    
    def get_model_path(self, *subpaths: str) -> str:
        """
        Get a path relative to the models directory.
        
        Args:
            *subpaths: Path components to append
            
        Returns:
            Absolute path
        """
        base_path = self.get('paths.models', str(self.base_dir / 'models'))
        return str(Path(base_path).joinpath(*subpaths).resolve())
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get an API key for a service.
        
        Args:
            service: Name of the service
            
        Returns:
            API key or None if not found
        """
        return self.get(f'api_keys.{service.lower()}')

# Global configuration instance
config = ConfigLoader()
