# Membership_Inference_Attack_and_Defense_Tools/Attack/shadow/shokri.py

import numpy as np
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
import torch

from Membership_Inference_Attack_and_Defense_Tools.Attack.components.attack_model import ClassWiseAttackClassifier
from Membership_Inference_Attack_and_Defense_Tools.utils.model_wrapper import ModelWrapper, PyTorchModelWrapper
from Membership_Inference_Attack_and_Defense_Tools.core.attack import AttackBase, AttackConfig, register_attack

# 1. 定义配置类（显示配置）
@dataclass
class ShokriAttackConfig(AttackConfig):
    """
    Shokri攻击配置类
    该类定义了Shokri影子模型攻击的配置参数。
    """
    num_shadow_models: int = 5  # 影子模型的数量
    hidden_layer_sizes: Any = (64, 32)  # 攻击模型的隐藏层大小
    random_state: int = 42  # 随机种子
    batch_size: int = 128  # 批处理大小
    shadow_data_split_ratio: float = 0.5  # 影子数据集划分比例
    epochs_per_shadow: int = 10  # 每个影子模型的训练轮数

# 2. 注册并实现类
@register_attack("shokri")
class ShokriAttack(AttackBase):
    """
    ShokriAttack 的 Docstring
    实现Shokri等人在论文"Membership Inference Attacks Against Machine Learning Models"中提出的影子模型攻击方法。

    """
    def __init__(self, 
                 target_model:ModelWrapper, 
                 config: ShokriAttackConfig, 
                 shadow_model_factory: Optional[Callable[[], Any]] = None, 
                 shadow_model_trainer: Optional[Callable[[Any, Any, int], None]] = None):
        """
        初始化Shokri攻击模型
        Args:
            target_model (ModelWrapper): 目标模型，被攻击的对象，建议经过wrapper处理
            config (ShokriAttackConfig): Shokri攻击配置参数
            shadow_model_factory (Optional[Callable[[], Any]]): 用于创建影子模型的工厂函数
            shadow_model_trainer (Optional[Callable[[Any, Any, int], None]]): 用于训练影子模型的函数，func(model, loader, epochs)
        """
        super().__init__(target_model, config)
        self.config: ShokriAttackConfig = config
        self.shadow_model_factory = shadow_model_factory
        self.shadow_model_trainer = shadow_model_trainer

        # 初始化攻击分类器
        # 为了暂时简化，假设 target_model.num_classes 可用
        # 实际开发中应该在fit时获取
        self.attack_classifier: Optional[ClassWiseAttackClassifier] = None

    def fit(self, shadow_data_loader=None): # target_data_loader=None -> None:
        """
        fit 的 Docstring
        
        :param self: 说明
        :param target_data_loader: 说明
        :param shadow_data_loader: 说明

        shokri攻击的训练过程包含两个主要阶段：
        1. 训练k个影子模型，模拟目标模型的行为。
        2. 构建攻击数据集。
        3. 训练攻击分类器。
        """
        if shadow_data_loader is None:
            raise ValueError("shadow_data_loader cannot be None for ShokriAttack.")
        if self.shadow_model_factory is None or self.shadow_model_trainer is None:
            raise ValueError("shadow_model_factory and shadow_model_trainer must be provided.")
        
        print(f"[*] Starting Shokri Attack Training with {self.config.num_shadow_models} shadow models.")

        # --- 阶段1：准备供货及数据容器 ---
        all_shadow_preds = []
        all_shadow_labels = [] # 样本的真实分类
        all_shadow_membership_labels = [] # 样本的成员标签 IN(1) / OUT(0)

        # 获取全部影子数据
        # full_shadow_data = []
        # for data, label in shadow_data_loader:
        #     full_shadow_data.append((data.numpy(), label.numpy()))
        # full_shadow_data = np.concatenate([np.stack([d for d, l in full_shadow_data]), 
        #                                    np.stack([l for d, l in full_shadow_data])], axis=1)
        # num_shadow_samples = full_shadow_data.shape[0]
        # full_shadow_data = self._unpack_loader(shadow_data_loader)
        X_shadow, y_shadow = self._unpack_loader(shadow_data_loader)

        # 动态确定类别数
        num_classes = len(np.unique(y_shadow))
        self.attack_classifier = ClassWiseAttackClassifier(num_classes=num_classes) # hidden_layer_sizes=self.config.hidden_layer_sizes,random_state=self.config.random_state

        # --- 阶段2：循环训练影子模型 ---
        dataset_len = len(X_shadow)

        for i in tqdm(range(self.config.num_shadow_models), desc="Training Shadow Models"):
            # 1. 随机划分影子数据集
            perm = np.random.permutation(dataset_len)
            
            split_point = int(self.config.shadow_data_split_ratio * dataset_len)
            shadow_train_data = (X_shadow[perm[:split_point]], y_shadow[perm[:split_point]])
            shadow_test_data = (X_shadow[perm[split_point:]], y_shadow[perm[split_point:]])

            # 2. 创建影子模型
            shadow_model = self.shadow_model_factory()

            # 3. 训练影子模型
            shadow_train_loader = self._create_data_loader(shadow_train_data, self.config.batch_size)
            self.shadow_model_trainer(shadow_model, shadow_train_loader, self.config.epochs_per_shadow)

            # 4. 获取影子模型在训练集和测试集上的预测
            # shadow_model_wrapper = ModelWrapper(shadow_model)  # 假设有一个通用的ModelWrapper
            shadow_model_wrapper = PyTorchModelWrapper(shadow_model)  # 假设影子模型是PyTorch模型
            train_preds = shadow_model_wrapper.get_outputs(shadow_train_data[0])
            test_preds = shadow_model_wrapper.get_outputs(shadow_test_data[0])

            # 5. 构建攻击数据集
            all_shadow_preds.append(np.concatenate([train_preds, test_preds], axis=0))
            all_shadow_labels.append(np.concatenate([shadow_train_data[1], shadow_test_data[1]], axis=0))
            all_shadow_membership_labels.append(np.concatenate([np.ones(len(shadow_train_data[0])), np.zeros(len(shadow_test_data[0]))], axis=0))

        # --- 阶段3：汇总所有影子模型的数据并训练攻击分类器 ---
        all_shadow_preds = np.concatenate(all_shadow_preds, axis=0)
        all_shadow_labels = np.concatenate(all_shadow_labels, axis=0)
        all_shadow_membership_labels = np.concatenate(all_shadow_membership_labels, axis=0)

        print(f"[*] Training Attack Classifier on {all_shadow_preds.shape[0]} samples.")
        self.attack_classifier.fit(all_shadow_preds, all_shadow_labels, all_shadow_membership_labels)
        print(f"[*] Shokri Attack Training Completed.")

    def predict(self, target_data_loader) -> np.ndarray:
        """
        攻击阶段：输入目标样本，输出成员概率
        """
        # 1. 获取目标模型的预测
        # 解包 loader
        X_target, y_target = self._unpack_loader(target_data_loader)

        # 2. TargetModel预测
        target_preds = self.target_model.get_outputs(X_target)
        # 3. 使用攻击分类器进行预测
        if self.attack_classifier is None:
            raise ValueError("Attack classifier has not been trained. Please call fit() before predict().")
        membership_scores = self.attack_classifier.predict(target_preds, y_target)
        return membership_scores
    
    def _unpack_loader(self, data_loader) -> np.ndarray:
        """
        _unpack_loader 的 Docstring
        
        :param self: 说明
        :param data_loader: 说明
        :return: 说明
        :rtype: Any
        辅助函数：将Dataloader解包为numpy数组（X，y）。
        在小数据集上可行，大数据集需要流式处理。
        """
        # 如果是list或tuple，直接处理
        if isinstance(data_loader, tuple):
            return data_loader[0], data_loader[1]
        
        # 如果是Pytorch DataLoader
        xs = []
        ys = []
        for data, label in data_loader:
            xs.append(data.numpy())
            ys.append(label.numpy())
        X = np.concatenate(xs)
        y = np.concatenate(ys)
        return X, y
    

    def _create_data_loader(self, data: np.ndarray, batch_size: int):
        """
        _create_data_loader 的 Docstring
        
        :param self: 说明
        :param data: 说明
        :param batch_size: 说明
        :return: 说明
        :rtype: Any
        辅助函数：将numpy数组转换为简单的DataLoader。
        这里只是一个简化示例，实际使用中应根据具体框架实现。
        """
        X, y = data

        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader
                                                          

