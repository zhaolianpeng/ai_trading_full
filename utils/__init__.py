# utils/__init__.py
from .logger import logger, setup_logger, get_logger

# 确保 logger 可用（延迟初始化）
if logger is None:
    logger = get_logger()

__all__ = ['logger', 'setup_logger', 'get_logger']
