# Membership-Inference-Attack-and-Defense-Tools/core/attack.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
import numpy as np

# --- 1.注册机制实现 ---
# 全局注册表，用于存储所有注册的攻击类
ATTACK_REGISTRY = {}

def register_attack(name: str):
    """
    注册攻击类的装饰器
    这是一个装饰器，用于将具体的攻击类自动注册到全局的ATTACK_REGISTRY中。
    使用方法如下:
        @register_attack('MyAttack')
        class MyAttack(AttackBase):
            ...
    这样，MyAttack类就会被注册为' MyAttack'，可以通过ATTACK_REGISTRY['MyAttack']访问。
    该机制方便了攻击类的动态加载和管理。
    Args:
        name (str): 攻击类的名称，用于注册和查找。
    Returns:
        Callable: 装饰器函数，将类注册到ATTACK_REGISTRY中。
    """
    def decorator(cls):
        ATTACK_REGISTRY[name] = cls
        return cls
    return decorator

# --- 2.配置基类 ---
class AttackConfig:
    """
    所有攻击配置的基类
    该类用于定义攻击的配置参数。具体的攻击实现可以继承该类，并添加特定的配置参数。
    建议使用dataclass在具体实现中定义配置参数，以便于管理和使用。
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- 3.攻击算法抽象基类 ---
class AttackBase(ABC):
    """
    AttackBase 的 Docstring
    MIA攻击算法的抽象基类，定义了所有具体攻击类必须实现的方法。
    强制所有子类实现fit和predict方法，以确保一致的接口。
    """

    def __init__(self, target_model: Any, attack_config: AttackConfig):
        """
        初始化攻击模型
        Args:
            target_model (Any): 目标模型，被攻击的对象，建议经过wrapper处理
            attack_config (AttackConfig): 攻击配置参数
        """
        self.target_model = target_model
        self.attack_config = attack_config

    @abstractmethod
    def fit(self, target_data_loader: Any, shadow_data_loader: Optional[Any] = None) -> None:
        """
        训练攻击模型
        训练阶段
        对于Shadow Model-based MIA: 这里负责训练影子模型和攻击模型
        对于Metric-based MIA: 这里可以什么都不做，或者计算阈值
        Args:
            target_data_loader (Any): 目标模型的数据加载器
            shadow_data_loader (Optional[Any]): 影子模型的数据加载器（可选）
        
        """
        pass

    @abstractmethod
    def predict(self, data_loader: Any) -> np.ndarray:
        """
        推断阶段
        使用攻击模型进行预测
        Args:
            data_loader (Any): 数据加载器
        Returns:
            np.ndarray: 预测结果
            返回成员概率(scores)或者成员标签(0/1)
        """
        pass

    def evaluate(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        评估攻击模型的性能
        计算并返回各种评估指标，如准确率、精确率、召回率等
        Args:
            predictions (np.ndarray): 攻击模型的预测结果
            true_labels (np.ndarray): 真实标签
        Returns:
            Dict[str, Union[float, Dict[str, float]]]: 评估指标的字典
        """
        # # 示例实现，可以根据需要添加更多指标
        # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # accuracy = accuracy_score(true_labels, predictions)
        # precision = precision_score(true_labels, predictions)
        # recall = recall_score(true_labels, predictions)
        # f1 = f1_score(true_labels, predictions)

        # return {
        #     "accuracy": accuracy,
        #     "precision": precision,
        #     "recall": recall,
        #     "f1_score": f1
        # }
        pass