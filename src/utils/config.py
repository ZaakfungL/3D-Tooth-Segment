import yaml
import os
import argparse
from typing import Dict, Any

def load_config(config_path: str = None, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    加载配置：默认配置 <- YAML配置

    Args:
        config_path: YAML 配置文件路径
        default_config: 默认配置字典

    Returns:
        合并后的配置字典
    """
    config = default_config.copy() if default_config else {}

    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # 递归更新配置，这里简单做浅层合并，如果需要深层合并可以改进
                config.update(yaml_config)
    elif config_path:
        print(f"Warning: Config file {config_path} not found. Using defaults.")

    return config

def get_config_argument_parser(description: str = "Script Configuration"):
    """
    获取包含 --config 参数的 ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    # 可以添加其他通用参数
    return parser
