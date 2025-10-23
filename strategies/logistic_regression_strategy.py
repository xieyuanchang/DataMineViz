from sklearn.linear_model import LogisticRegression
from logistic_regression_visualizer import LogisticRegressionVisualizer
from .model_strategy import ModelStrategy

class LogisticRegressionStrategy(ModelStrategy):
    """
    逻辑回归策略实现
    包含逻辑回归模型的特定逻辑和配置
    """
    
    def __init__(self):
        self.model = None
        self.config = {}
    
    def initialize(self, config):
        """
        初始化逻辑回归策略
        
        参数:
        - config: 配置字典，包含逻辑回归特定配置
        """
        # 逻辑回归默认配置
        default_config = {
            'max_steps': 15,
            'learning_rate': 0.01,
            'random_state': 42,
            'max_iter': 1000,
            'auto_run': True,
            'auto_close': True
        }
        
        # 更新默认配置
        default_config.update(config)
        self.config = default_config
    
    def create_model(self):
        """
        创建逻辑回归模型
        
        返回:
        - LogisticRegression 实例
        """
        self.model = LogisticRegression(
            random_state=self.config['random_state'],
            max_iter=self.config['max_iter']
        )
        return self.model
    
    def train_model(self, X_train, y_train):
        """
        训练逻辑回归模型
        
        参数:
        - X_train: 训练特征
        - y_train: 训练标签
        """
        self.model.fit(X_train, y_train)
    
    def create_visualizer(self):
        """
        创建逻辑回归可视化器
        
        返回:
        - LogisticRegressionVisualizer 实例
        """
        return LogisticRegressionVisualizer()
    
    def configure_visualizer(self, visualizer, X, y, X_train, X_test, y_train, y_test, feature_names, target_names):
        """
        配置逻辑回归可视化器
        
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
        # 设置数据集和模型
        visualizer.set_data(X, y, X_train, X_test, y_train, y_test)
        visualizer.set_model(self.model)
        visualizer.set_feature_names(feature_names)
        visualizer.set_target_names(target_names)
        visualizer.max_steps = self.config['max_steps']
        visualizer.set_learning_rate(self.config['learning_rate'])
        visualizer.auto_run = self.config['auto_run']
        visualizer.auto_close = self.config['auto_close']
    
    def get_model_name(self):
        """
        获取模型名称
        
        返回:
        - 模型名称字符串
        """
        return "逻辑回归"
    
    def print_model_info(self, X, y, feature_names):
        """
        打印逻辑回归模型信息
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - feature_names: 特征名称
        """
        print(f"逻辑回归配置:")
        print(f"  最大训练步数: {self.config['max_steps']}")
        print(f"  学习率: {self.config['learning_rate']}")
        
        # 计算模型参数
        print("模型参数:")
        print(f"  权重 ({', '.join(feature_names)}): {self.model.coef_[0]}")
        print(f"  偏置: {self.model.intercept_[0]:.4f}")
        
        # 计算模型性能（这里假设已经训练过模型）
        train_score = self.model.score(X[:int(len(X)*0.8)], y[:int(len(y)*0.8)]) * 100
        test_score = self.model.score(X[int(len(X)*0.8):], y[int(len(y)*0.8):]) * 100
        print(f"  训练集准确率: {train_score:.1f}%")
        print(f"  测试集准确率: {test_score:.1f}%")