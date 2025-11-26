# utils/logger.py
import logging
import sys
from pathlib import Path
from config import LOG_LEVEL, LOG_FILE

def setup_logger(name: str = 'ai_trading', log_level: str = None, log_file: str = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR）
        log_file: 日志文件路径，None 表示不写入文件
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    level = getattr(logging, (log_level or LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file or LOG_FILE:
        try:
            file_path = Path(log_file or LOG_FILE)
            # 尝试创建父目录（如果不存在）
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot create log directory {file_path.parent}: {e}. Using console only.")
                return logger
            
            # 尝试创建文件处理器
            try:
                file_handler = logging.FileHandler(file_path, encoding='utf-8')
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot create log file {file_path}: {e}. Using console only.")
        except Exception as e:
            # 捕获所有其他异常（如 os.getcwd() 失败）
            # 注意：此时 logger 可能还没有完全初始化，使用 stderr 输出
            try:
                logger.warning(f"Cannot setup file logging: {e}. Using console only.")
            except:
                sys.stderr.write(f"Warning: Cannot setup file logging: {e}. Using console only.\n")
    
    return logger

# 创建默认日志记录器（延迟初始化，避免导入时的权限问题）
logger = None

def get_logger():
    """获取日志记录器（延迟初始化）"""
    global logger
    if logger is None:
        try:
            logger = setup_logger()
        except Exception as e:
            # 如果初始化失败，创建一个基本的控制台日志记录器
            import logging
            logger = logging.getLogger('ai_trading')
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.warning(f"Failed to setup full logger: {e}. Using basic console logger.")
    return logger

# 为了向后兼容，在导入时尝试初始化（但允许失败）
try:
    logger = setup_logger()
except Exception:
    # 如果失败，logger 保持为 None，将在首次使用时初始化
    pass
