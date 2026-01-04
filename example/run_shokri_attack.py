# Membership_Inference_attack_and_Defence_Tools/example/run_shokri_attack.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from Membership_Inference_Attack_and_Defense_Tools.utils.model_wrapper import PyTorchModelWrapper
from Membership_Inference_Attack_and_Defense_Tools.Attack import ATTACK_REGISTRY
from Membership_Inference_Attack_and_Defense_Tools.Attack.shadow.shokri import ShokriAttackConfig

# --- 1. 自定义的模型 --- 
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.fc(x)
    
# --- 2. 准备数据 ---
# 这里我们使用随机数据作为示例
X_shadow = torch.randn(1000, 10)
y_shadow = torch.randint(0, 2, (1000,))
shadow_loader = TensorDataset(X_shadow, y_shadow) # 简单封装

target_model = SimpleNN()
target_wrapper = PyTorchModelWrapper(target_model) # 假设已经训练好了

# --- 3. 配置并运行Shokri攻击 ---
# 定义shadow model 工厂和训练器
# 需要告诉库，如何创建一个新的影子模型
def shadow_factory():
    return SimpleNN()

# 需要告诉库：如何训练这个模型（库不关心optimization细节）
def shadow_trainer(model, loader, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(loader, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
# 配置攻击
config = ShokriAttackConfig(
    num_shadow_models=3,
    hidden_layer_sizes=(32, 16),
    random_state=123,
    batch_size=64,
    shadow_data_split_ratio=0.5,
    epochs_per_shadow=5
)

attack_class = ATTACK_REGISTRY["shokri"]
shokri_attack = attack_class(
    target_model=target_wrapper,
    config=config,
    shadow_model_factory=shadow_factory,
    shadow_model_trainer=shadow_trainer
)   

# 训练攻击模型
shokri_attack.fit(shadow_data_loader=shadow_loader)

# 进行攻击推断
attack_results = shokri_attack.predict(shadow_loader)
print("Attack Results:", attack_results)
