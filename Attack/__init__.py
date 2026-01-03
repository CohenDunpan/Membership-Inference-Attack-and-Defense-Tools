from .shadow import ShokriAttack

from Membership_Inference_Attack_and_Defense_Tools.core.attack import AttackBase, AttackConfig, register_attack, ATTACK_REGISTRY
from typing import Any

def get_attack(name: str, target_model: Any, attack_config: AttackConfig) -> AttackBase:
    """
    根据名称获取攻击实例
    该函数用于根据攻击名称从注册表中查找对应的攻击类，并实例化该类。
    Args:
        name (str): 攻击类的名称
        target_model (Any): 目标模型，被攻击的对象
        attack_config (AttackConfig): 攻击配置参数
    Returns:
        AttackBase: 实例化的攻击类对象
    Raises:
        ValueError: 如果指定的攻击名称未注册，则抛出异常
    """
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Attack '{name}' is not registered. Available attacks: {list(ATTACK_REGISTRY.keys())}")
    attack_class = ATTACK_REGISTRY[name]
    return attack_class(target_model, attack_config)