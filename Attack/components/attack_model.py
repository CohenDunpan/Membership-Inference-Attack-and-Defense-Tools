# Membership_Inference_Attack_and_Defense_Tools/Attack/components/attack_model.py

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, List

class ClassWiseAttackClassifier:
    """
    ClassWiseAttackClassifier 的 Docstring
    实现Shokri论文中的核心策略：
    针对目标模型的每一个类别$y$，单独训练一个攻击二分类器。
    输入为目标模型在该类别上的后验概率，输出为样本是否属于训练集的标签。
    """
    def __init__(self, num_classes: int, hidden_layer_sizes: List[int] = [64, 32], random_state: int = 42):
        """
        初始化ClassWiseAttackClassifier
        Args:
            num_classes (int): 目标模型的类别数量
            hidden_layer_sizes (List[int]): MLP隐藏层的大小列表
            random_state (int): 随机种子，默认为42
        """
        self.num_classes = num_classes
        self.attack_classifiers: Dict[int, MLPClassifier] = {}
        for class_idx in range(num_classes):
            self.attack_classifiers[class_idx] = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                                random_state=random_state,
                                                                max_iter=200)
    def fit(self, shadow_preds: np.ndarray, shadow_labels: np.ndarray, shadow_membership: np.ndarray):
        """
        训练每个类别的攻击分类器
        Args:
            shadow_preds (np.ndarray): 影子模型的预测后验概率，形状为 (num_samples, num_classes)
            shadow_labels (np.ndarray): 影子模型的真实标签，形状为 (num_samples,)
            shadow_membership (np.ndarray): 影子模型样本的成员标签，形状为 (num_samples,)
        """
        for class_idx in range(self.num_classes):
            # 1. 筛选出属于当前类别的样本
            class_indices = np.where(shadow_labels == class_idx)[0]
            if len(class_indices) == 0:
                continue

            # 2. 准备训练数据X(预测向量)和标签y(成员标签)
            X_class = shadow_preds[class_indices, class_idx].reshape(-1, 1)
            y_class = shadow_membership[class_indices]

            # 3. 只有当既有正样本又有负样本时，才训练分类器，避免报错
            if len(np.unique(y_class)) < 2:
                print(f"[Warning] Class {class_idx} has only one class in membership labels. Skipping training for this class.")
                continue

            clf = MLPClassifier(hidden_layer_sizes=self.attack_classifiers[class_idx].hidden_layer_sizes,
                                random_state=self.attack_classifiers[class_idx].random_state,
                                max_iter=self.attack_classifiers[class_idx].max_iter)
            clf.fit(X_class, y_class)
            self.attack_classifiers[class_idx] = clf

    def predict(self, target_preds: np.ndarray, target_labels: np.ndarray) -> np.ndarray:
        """
        对目标样本进行攻击推断。
        Args:
            target_preds (np.ndarray): 目标模型的预测后验概率，形状为 (num_samples, num_classes)
            target_labels (np.ndarray): 目标模型的真实标签，形状为 (num_samples,)
        Returns:
            np.ndarray: 预测的成员标签，形状为 (num_samples,)
        """
        final_scores = np.zeros(target_preds.shape[0])

        for class_idx in range(self.num_classes):
            class_indices = np.where(target_labels == class_idx)[0]
            if len(class_indices) == 0 or class_idx not in self.attack_classifiers:
                # 如果该类别没有对应的攻击模型，默认返回0.5（表示不确定）
                final_scores[class_indices] = 0.5
                continue

            X_class = target_preds[class_indices, class_idx].reshape(-1, 1)
            clf = self.attack_classifiers[class_idx]

            # 获取属于该类别样本的预测概率
            probs = clf.predict_proba(X_class)[:, 1]
            final_scores[class_indices] = probs

            return final_scores

            # if hasattr(clf, "predict_proba"):
            #     class_scores = clf.predict_proba(X_class)[:, 1]
            # else:
            #     class_scores = clf.decision_function(X_class)

            # final_scores[class_indices] = class_scores

            # return (final_scores >= 0.5).astype(int)