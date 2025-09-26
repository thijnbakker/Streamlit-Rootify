import logging
from src.npecage.utils.logger import setup_logger


def test_setup_logger_adds_stream_handler(caplog):
    # Arrange
    logger_name = ""  # root logger
    logger = logging.getLogger(logger_name)

    # Clear previous handlers (important for repeatable tests)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Act
    logger = setup_logger(logging.DEBUG)

    # Assert
    assert logger.level == logging.DEBUG

    # Check at least one StreamHandler is attached
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) > 0, "No StreamHandler attached."

    # Check formatter is correctly set
    for handler in stream_handlers:
        formatter = handler.formatter
        assert formatter is not None, "Formatter is not set on handler."
        assert '%(asctime)s' in formatter._fmt
        assert '%(levelname)s' in formatter._fmt
        assert '%(message)s' in formatter._fmt
