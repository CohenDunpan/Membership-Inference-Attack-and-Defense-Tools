# Membership_Inference_Attack_and_Defense_Tools/utils/model_wrapper.py

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional

class ModelWrapper(ABC):
    """
    模型包装器的抽象基类
    该类定义了一个统一的接口，用于包装不同深度学习框架（如PyTorch、TensorFlow等）的模型。
    通过继承该类，可以实现对不同框架模型的统一调用和管理。
    """
    @abstractmethod
    def get_outputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        获取模型的输出
        该方法接受输入数据，并返回模型的预测输出。
        输入numpy数据，输出该模型针对MIA需要的后验概率（Posterior Probability）。
        形状为 (batch_size, num_classes)
        Args:
            inputs (np.ndarray): 输入数据，通常为numpy数组格式
        Returns:
            np.ndarray: 模型的预测输出，通常为numpy数组格式
        """
        pass

class PyTorchModelWrapper(ModelWrapper):
    """
    PyTorch模型包装器
    该类实现了对PyTorch模型的包装，继承自ModelWrapper基类。
    """
    def __init__(self, model: torch.nn.Module, device: torch.device = torch.device('cpu'), input_size: Optional[tuple] = None):
        """
        初始化PyTorch模型包装器
        Args:
            model (torch.nn.Module): 需要包装的PyTorch模型
            device (torch.device): 模型运行的设备，默认为CPU
            input_size (Optional[tuple]): 输入数据的形状，默认为None
        """
        
        self.model = model.to(device)
        self.device = device
        self.input_size = input_size
        self.model.eval()  # 设置模型为评估模式

    def get_outputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        获取PyTorch模型的输出
        Args:
            inputs (np.ndarray): 输入数据，通常为numpy数组格式
        Returns:
            np.ndarray: 模型的预测输出，通常为numpy数组格式
        """
        with torch.no_grad():
            inputs_tensor = torch.from_numpy(inputs).to(self.device)
            outputs = self.model(inputs_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
        
# 未来再添加其他框架的包装器，如TensorFlowModelWrapper等
