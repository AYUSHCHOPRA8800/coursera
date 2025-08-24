import pytest
import tempfile
import os
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch, Mock

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from logger import (
    setup_logger, get_logger, LoggerMixin, log_function_call, 
    log_execution_time, LogContext, setup_application_logging,
    log_prediction, log_model_training, log_data_ingestion
)

class TestLoggerSetup:
    """Test cases for logger setup and configuration"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_setup_logger_basic(self, temp_log_dir):
        """Test basic logger setup"""
        logger = setup_logger(
            name="test_logger",
            log_level="INFO",
            log_file="test.log",
            log_dir=temp_log_dir,
            use_json=False,
            use_console=True
        )
        
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logger_json_format(self, temp_log_dir):
        """Test logger setup with JSON formatting"""
        logger = setup_logger(
            name="test_json_logger",
            log_level="DEBUG",
            log_file="test_json.log",
            log_dir=temp_log_dir,
            use_json=True,
            use_console=True
        )
        
        assert logger.name == "test_json_logger"
        assert logger.level == logging.DEBUG
        
        # Test JSON logging
        with patch('sys.stdout') as mock_stdout:
            logger.info("Test message")
            # Check that JSON was written to stdout
            mock_stdout.write.assert_called()
    
    def test_setup_logger_file_only(self, temp_log_dir):
        """Test logger setup with file output only"""
        logger = setup_logger(
            name="test_file_logger",
            log_level="WARNING",
            log_file="test_file.log",
            log_dir=temp_log_dir,
            use_json=False,
            use_console=False
        )
        
        assert logger.name == "test_file_logger"
        assert logger.level == logging.WARNING
        
        # Test that log file is created
        log_file = Path(temp_log_dir) / "test_file.log"
        logger.info("Test message")
        assert log_file.exists()
    
    def test_setup_logger_no_file(self, temp_log_dir):
        """Test logger setup without file output"""
        logger = setup_logger(
            name="test_console_logger",
            log_level="ERROR",
            log_file=None,
            log_dir=temp_log_dir,
            use_json=False,
            use_console=True
        )
        
        assert logger.name == "test_console_logger"
        assert logger.level == logging.ERROR
        assert len(logger.handlers) == 1  # Only console handler
    
    def test_setup_logger_invalid_level(self, temp_log_dir):
        """Test logger setup with invalid log level"""
        with pytest.raises(AttributeError):
            setup_logger(
                name="test_invalid_logger",
                log_level="INVALID_LEVEL",
                log_dir=temp_log_dir
            )
    
    def test_get_logger_existing(self, temp_log_dir):
        """Test getting an existing logger"""
        # Create a logger first
        logger1 = setup_logger(
            name="existing_logger",
            log_level="INFO",
            log_dir=temp_log_dir
        )
        
        # Get the same logger
        logger2 = get_logger("existing_logger")
        
        assert logger1 is logger2
        assert logger1.name == "existing_logger"
    
    def test_get_logger_new(self, temp_log_dir):
        """Test getting a new logger"""
        logger = get_logger("new_logger")
        
        assert logger.name == "new_logger"
        assert len(logger.handlers) > 0

class TestLoggerMixin:
    """Test cases for LoggerMixin class"""
    
    def test_logger_mixin_initialization(self, temp_log_dir):
        """Test LoggerMixin initialization"""
        class TestClass(LoggerMixin):
            def __init__(self):
                super().__init__()
        
        obj = TestClass()
        assert hasattr(obj, 'logger')
        assert obj.logger.name == "TestClass"
    
    def test_log_with_context(self, temp_log_dir):
        """Test logging with context"""
        class TestClass(LoggerMixin):
            def __init__(self):
                super().__init__()
        
        obj = TestClass()
        
        with patch.object(obj.logger, 'log') as mock_log:
            obj.log_with_context("INFO", "Test message", user_id=123, action="test")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.INFO
            assert call_args[0][1] == "Test message"
            assert 'extra_fields' in call_args[1]['extra']
            assert call_args[1]['extra']['extra_fields']['context']['user_id'] == 123

class TestLogDecorators:
    """Test cases for logging decorators"""
    
    def test_log_function_call(self, temp_log_dir):
        """Test log_function_call decorator"""
        @log_function_call
        def test_function(a, b, c=None):
            return a + b + (c or 0)
        
        with patch('app.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = test_function(1, 2, c=3)
            
            assert result == 6
            assert mock_logger.debug.call_count == 2  # Entry and exit logs
    
    def test_log_function_call_with_exception(self, temp_log_dir):
        """Test log_function_call decorator with exception"""
        @log_function_call
        def test_function_with_error():
            raise ValueError("Test error")
        
        with patch('app.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError):
                test_function_with_error()
            
            assert mock_logger.debug.call_count == 1  # Entry log
            assert mock_logger.error.call_count == 1  # Error log
    
    def test_log_execution_time(self, temp_log_dir):
        """Test log_execution_time decorator"""
        @log_execution_time
        def test_function():
            import time
            time.sleep(0.1)  # Simulate some work
            return "success"
        
        with patch('app.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = test_function()
            
            assert result == "success"
            assert mock_logger.info.call_count == 1
            call_args = mock_logger.info.call_args[0][0]
            assert "executed in" in call_args
            assert "seconds" in call_args
    
    def test_log_execution_time_with_exception(self, temp_log_dir):
        """Test log_execution_time decorator with exception"""
        @log_execution_time
        def test_function_with_error():
            import time
            time.sleep(0.1)
            raise RuntimeError("Test error")
        
        with patch('app.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(RuntimeError):
                test_function_with_error()
            
            assert mock_logger.error.call_count == 1
            call_args = mock_logger.error.call_args[0][0]
            assert "failed after" in call_args
            assert "seconds" in call_args

class TestLogContext:
    """Test cases for LogContext class"""
    
    def test_log_context(self, temp_log_dir):
        """Test LogContext context manager"""
        logger = setup_logger(
            name="test_context_logger",
            log_level="INFO",
            log_dir=temp_log_dir
        )
        
        context = {'user_id': 123, 'session_id': 'abc123'}
        
        with LogContext(logger, context):
            # The logger should have the context set
            assert hasattr(logger, '_extra_fields')
            assert logger._extra_fields == context
        
        # Context should be restored after exit
        assert not hasattr(logger, '_extra_fields')

class TestApplicationLogging:
    """Test cases for application-wide logging setup"""
    
    def test_setup_application_logging(self, temp_log_dir):
        """Test application-wide logging setup"""
        root_logger = setup_application_logging(
            app_name="test_app",
            log_level="DEBUG",
            log_dir=temp_log_dir
        )
        
        assert root_logger.name == ""
        assert root_logger.level == logging.DEBUG
        
        # Check that log files were created
        log_files = [
            "app.log",
            "app_main.log",
            "app_ingest_data.log",
            "app_train_model.log",
            "app_monitor.log",
            "app_utils.log"
        ]
        
        for log_file in log_files:
            assert (Path(temp_log_dir) / log_file).exists()
    
    def test_setup_application_logging_custom_level(self, temp_log_dir):
        """Test application logging with custom log level"""
        root_logger = setup_application_logging(
            app_name="test_app",
            log_level="WARNING",
            log_dir=temp_log_dir
        )
        
        assert root_logger.level == logging.WARNING

class TestLoggingFunctions:
    """Test cases for specific logging functions"""
    
    def test_log_prediction(self, temp_log_dir):
        """Test log_prediction function"""
        logger = setup_logger(
            name="test_prediction_logger",
            log_level="INFO",
            log_dir=temp_log_dir
        )
        
        with patch.object(logger, 'info') as mock_info:
            log_prediction(logger, "USA", 95.5, 0.85, 0.1)
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0][0] == "Prediction made"
            assert 'extra_fields' in call_args[1]['extra']
            extra_fields = call_args[1]['extra']['extra_fields']
            assert extra_fields['event_type'] == 'prediction'
            assert extra_fields['country'] == 'USA'
            assert extra_fields['prediction_value'] == 95.5
            assert extra_fields['confidence'] == 0.85
            assert extra_fields['processing_time'] == 0.1
    
    def test_log_model_training(self, temp_log_dir):
        """Test log_model_training function"""
        logger = setup_logger(
            name="test_training_logger",
            log_level="INFO",
            log_dir=temp_log_dir
        )
        
        with patch.object(logger, 'info') as mock_info:
            log_model_training(logger, "RandomForest", 120.5, 0.85, 1000)
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0][0] == "Model training completed"
            assert 'extra_fields' in call_args[1]['extra']
            extra_fields = call_args[1]['extra']['extra_fields']
            assert extra_fields['event_type'] == 'model_training'
            assert extra_fields['model_name'] == 'RandomForest'
            assert extra_fields['training_time'] == 120.5
            assert extra_fields['accuracy'] == 0.85
            assert extra_fields['dataset_size'] == 1000
    
    def test_log_data_ingestion(self, temp_log_dir):
        """Test log_data_ingestion function"""
        logger = setup_logger(
            name="test_ingestion_logger",
            log_level="INFO",
            log_dir=temp_log_dir
        )
        
        with patch.object(logger, 'info') as mock_info:
            log_data_ingestion(logger, "World Bank API", 500, 30.2)
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0][0] == "Data ingestion completed"
            assert 'extra_fields' in call_args[1]['extra']
            extra_fields = call_args[1]['extra']['extra_fields']
            assert extra_fields['event_type'] == 'data_ingestion'
            assert extra_fields['source'] == 'World Bank API'
            assert extra_fields['records_processed'] == 500
            assert extra_fields['processing_time'] == 30.2

class TestLogFileOutput:
    """Test cases for log file output"""
    
    def test_log_file_creation(self, temp_log_dir):
        """Test that log files are created and written to"""
        logger = setup_logger(
            name="test_file_output",
            log_level="INFO",
            log_file="test_output.log",
            log_dir=temp_log_dir,
            use_console=False
        )
        
        log_file = Path(temp_log_dir) / "test_output.log"
        
        # Write some log messages
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Check that file exists and contains log messages
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test info message" in content
            assert "Test warning message" in content
            assert "Test error message" in content
            assert "INFO" in content
            assert "WARNING" in content
            assert "ERROR" in content
    
    def test_log_file_rotation(self, temp_log_dir):
        """Test log file rotation"""
        logger = setup_logger(
            name="test_rotation",
            log_level="INFO",
            log_file="test_rotation.log",
            log_dir=temp_log_dir,
            use_console=False
        )
        
        log_file = Path(temp_log_dir) / "test_rotation.log"
        
        # Write enough data to trigger rotation (10MB limit)
        large_message = "x" * 1000  # 1KB per message
        
        for i in range(11000):  # Should trigger rotation
            logger.info(f"Message {i}: {large_message}")
        
        # Check that backup files were created
        backup_files = list(Path(temp_log_dir).glob("test_rotation.log.*"))
        assert len(backup_files) > 0
    
    def test_json_log_format(self, temp_log_dir):
        """Test JSON log format output"""
        logger = setup_logger(
            name="test_json_format",
            log_level="INFO",
            log_file="test_json.log",
            log_dir=temp_log_dir,
            use_json=True,
            use_console=False
        )
        
        log_file = Path(temp_log_dir) / "test_json.log"
        
        # Write a log message
        logger.info("Test JSON message")
        
        # Check that file contains valid JSON
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read().strip()
            log_entry = json.loads(content)
            
            assert log_entry['level'] == 'INFO'
            assert log_entry['message'] == 'Test JSON message'
            assert log_entry['logger'] == 'test_json_format'
            assert 'timestamp' in log_entry
            assert 'module' in log_entry
            assert 'function' in log_entry
            assert 'line' in log_entry

class TestLogLevels:
    """Test cases for different log levels"""
    
    def test_debug_level(self, temp_log_dir):
        """Test DEBUG level logging"""
        logger = setup_logger(
            name="test_debug",
            log_level="DEBUG",
            log_file="test_debug.log",
            log_dir=temp_log_dir,
            use_console=False
        )
        
        log_file = Path(temp_log_dir) / "test_debug.log"
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
    
    def test_info_level(self, temp_log_dir):
        """Test INFO level logging"""
        logger = setup_logger(
            name="test_info",
            log_level="INFO",
            log_file="test_info.log",
            log_dir=temp_log_dir,
            use_console=False
        )
        
        log_file = Path(temp_log_dir) / "test_info.log"
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Debug message" not in content  # Should be filtered out
            assert "Info message" in content
            assert "Warning message" in content
    
    def test_warning_level(self, temp_log_dir):
        """Test WARNING level logging"""
        logger = setup_logger(
            name="test_warning",
            log_level="WARNING",
            log_file="test_warning.log",
            log_dir=temp_log_dir,
            use_console=False
        )
        
        log_file = Path(temp_log_dir) / "test_warning.log"
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Debug message" not in content
            assert "Info message" not in content
            assert "Warning message" in content
            assert "Error message" in content

if __name__ == "__main__":
    pytest.main([__file__])
