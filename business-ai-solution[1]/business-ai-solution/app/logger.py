import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

def setup_logger(name: str, 
                log_level: str = "INFO",
                log_file: Optional[str] = None,
                log_dir: str = "../logs",
                use_json: bool = False,
                use_console: bool = True) -> logging.Logger:
    """
    Setup a logger with specified configuration
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (optional)
        log_dir: Directory for log files
        use_json: Whether to use JSON formatting
        use_console: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create log directory if it doesn't exist
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
    
    # Console handler
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if use_json:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_path = Path(log_dir) / log_file
        
        # Use rotating file handler for better log management
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if use_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default configuration
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    
    # Check if logger already exists
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    
    # Determine log file based on module name
    module_name = name.split('.')[-1] if '.' in name else name
    log_file = f"{module_name}.log"
    
    # Setup logger with default configuration
    return setup_logger(
        name=name,
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        log_file=log_file,
        log_dir=os.getenv('LOG_DIR', '../logs'),
        use_json=os.getenv('LOG_JSON', 'false').lower() == 'true',
        use_console=True
    )

class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_with_context(self, level: str, message: str, **context):
        """Log message with additional context"""
        extra_fields = {'context': context}
        self.logger.log(getattr(logging, level.upper()), message, extra={'extra_fields': extra_fields})

def log_function_call(func):
    """Decorator to log function calls with parameters and return values"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised exception: {e}")
            raise
    
    return wrapper

def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper

class LogContext:
    """Context manager for logging with additional context"""
    
    def __init__(self, logger: logging.Logger, context: dict):
        self.logger = logger
        self.context = context
        self.original_extra = getattr(self.logger, '_extra_fields', {})
    
    def __enter__(self):
        self.logger._extra_fields = {**self.original_extra, **self.context}
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger._extra_fields = self.original_extra

def setup_application_logging(app_name: str = "business-ai-solution",
                            log_level: str = "INFO",
                            log_dir: str = "../logs") -> logging.Logger:
    """
    Setup application-wide logging configuration
    
    Args:
        app_name: Application name
        log_level: Logging level
        log_dir: Log directory
    
    Returns:
        Root logger instance
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Setup root logger
    root_logger = setup_logger(
        name="",
        log_level=log_level,
        log_file="app.log",
        log_dir=log_dir,
        use_json=False,
        use_console=True
    )
    
    # Setup specific loggers for different components
    component_loggers = [
        "app.main",
        "app.ingest_data", 
        "app.train_model",
        "app.monitor",
        "app.utils"
    ]
    
    for component in component_loggers:
        setup_logger(
            name=component,
            log_level=log_level,
            log_file=f"{component.replace('.', '_')}.log",
            log_dir=log_dir,
            use_json=False,
            use_console=False
        )
    
    # Log application startup
    root_logger.info(f"Application {app_name} logging initialized")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info(f"Log directory: {log_path.absolute()}")
    
    return root_logger

# Example usage functions
def log_prediction(logger: logging.Logger, country: str, prediction: float, 
                  confidence: float, processing_time: float):
    """Log a prediction event with structured data"""
    logger.info("Prediction made", extra={
        'extra_fields': {
            'event_type': 'prediction',
            'country': country,
            'prediction_value': prediction,
            'confidence': confidence,
            'processing_time': processing_time
        }
    })

def log_model_training(logger: logging.Logger, model_name: str, 
                      training_time: float, accuracy: float, dataset_size: int):
    """Log model training event"""
    logger.info("Model training completed", extra={
        'extra_fields': {
            'event_type': 'model_training',
            'model_name': model_name,
            'training_time': training_time,
            'accuracy': accuracy,
            'dataset_size': dataset_size
        }
    })

def log_data_ingestion(logger: logging.Logger, source: str, 
                      records_processed: int, processing_time: float):
    """Log data ingestion event"""
    logger.info("Data ingestion completed", extra={
        'extra_fields': {
            'event_type': 'data_ingestion',
            'source': source,
            'records_processed': records_processed,
            'processing_time': processing_time
        }
    })

# Initialize default logger
_default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default application logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_application_logging()
    return _default_logger
