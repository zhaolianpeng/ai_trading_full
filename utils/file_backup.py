# utils/file_backup.py
"""
文件备份工具
在写入新文件之前，自动备份已存在的文件
"""
import sys
from pathlib import Path
from datetime import datetime
from utils.logger import logger

def backup_file_if_exists(file_path: Path) -> bool:
    """
    如果文件存在且不为空，则备份它
    
    Args:
        file_path: 要备份的文件路径
    
    Returns:
        bool: 是否成功备份（如果文件不存在或为空，返回 True）
    """
    try:
        # 检查文件是否存在
        if not file_path.exists():
            return True
        
        # 检查文件是否为空
        if file_path.stat().st_size == 0:
            return True
        
        # 生成备份文件名：原文件名 + 时间戳 + .bak
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = file_path.parent / f"{file_path.stem}_{timestamp}{file_path.suffix}.bak"
        
        # 如果备份文件已存在，尝试添加序号
        counter = 1
        original_backup_path = backup_path
        while backup_path.exists():
            backup_path = file_path.parent / f"{file_path.stem}_{timestamp}_{counter}{file_path.suffix}.bak"
            counter += 1
            if counter > 1000:  # 防止无限循环
                logger.warning(f"备份文件序号超过1000，使用原始备份路径: {original_backup_path}")
                backup_path = original_backup_path
                break
        
        # 重命名旧文件为备份文件
        file_path.rename(backup_path)
        logger.info(f"已备份文件: {file_path.name} -> {backup_path.name}")
        return True
        
    except (PermissionError, OSError) as e:
        # 如果备份失败，记录警告但继续（允许覆盖旧文件）
        logger.warning(f"无法备份文件 {file_path}: {e}，将覆盖旧文件")
        return False
    except Exception as e:
        # 捕获所有其他异常
        logger.warning(f"备份文件时发生错误 {file_path}: {e}，将覆盖旧文件")
        return False

