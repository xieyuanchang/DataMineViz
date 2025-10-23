from abc import ABC, abstractmethod

class ModelStrategy(ABC):
    """
    模型策略接口
    定义所有模型策略必须实现的方法
    """
    
    @abstractmethod
    def initialize(self, config):
        """
        初始化策略
        
        参数:
        - config: 配置字典
        """
        pass
    
    @abstractmethod
    def create_model(self):
        """
        创建模型
        
        返回:
        - 模型实例
        """
        pass
    
    @abstractmethod
    def train_model(self, X_train, y_train):
        """
        训练模型
        
        参数:
        - X_train: 训练特征
        - y_train: 训练标签
        """
        pass
    
    @abstractmethod
    def create_visualizer(self):
        """
        创建可视化器
        
        返回:
        - 可视化器实例
        """
        pass
    
    @abstractmethod
    def configure_visualizer(self, visualizer, X, y, X_train, X_test, y_train, y_test, feature_names, target_names):
        """
        配置可视化器
        
        参数:
        - visualizer: 可视化器实例
        - X: 完整特征集
        - y: 完整标签集
        - X_train: 训练特征
        - X_test: 测试特征
        - y_train: 训练标签
        - y_test: 测试标签
        - feature_names: 特征名称
        - target_names: 目标名称
        """
        pass
    
    @abstractmethod
    def get_model_name(self):
        """
        获取模型名称
        
        返回:
        - 模型名称字符串
        """
        pass
    
    @abstractmethod
    def print_model_info(self, X, y, feature_names):
        """
        打印模型信息
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - feature_names: 特征名称
        """
        pass