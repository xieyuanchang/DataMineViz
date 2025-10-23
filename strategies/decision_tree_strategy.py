from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from decision_tree_visualizer import DecisionTreeVisualizer
from .model_strategy import ModelStrategy

class DecisionTreeStrategy(ModelStrategy):
    """
    决策树策略实现
    包含决策树模型的特定逻辑和配置
    """
    
    def __init__(self):
        self.model = None
        self.config = {}
    
    def initialize(self, config):
        """
        初始化决策树策略
        
        参数:
        - config: 配置字典，包含决策树特定配置
        """
        # 决策树默认配置
        default_config = {
            'max_depth': 3,
            'max_steps': 5,
            'random_state': 42,
            'auto_run': True,
            'auto_close': True
        }
        
        # 更新默认配置
        default_config.update(config)
        self.config = default_config
    
    def create_model(self):
        """
        创建决策树模型
        
        返回:
        - DecisionTreeClassifier 实例
        """
        self.model = DecisionTreeClassifier(
            max_depth=self.config['max_depth'],
            random_state=self.config['random_state']
        )
        return self.model
    
    def train_model(self, X_train, y_train):
        """
        训练决策树模型
        
        参数:
        - X_train: 训练特征
        - y_train: 训练标签
        """
        self.model.fit(X_train, y_train)
    
    def create_visualizer(self):
        """
        创建决策树可视化器
        
        返回:
        - DecisionTreeVisualizer 实例
        """
        return DecisionTreeVisualizer()
    
    def configure_visualizer(self, visualizer, X, y, X_train, X_test, y_train, y_test, feature_names, target_names):
        """
        配置决策树可视化器
        
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
        visualizer.set_max_depth(self.config['max_depth'])
        visualizer.set_feature_names(feature_names)
        visualizer.set_target_names(target_names)
        visualizer.max_steps = self.config['max_steps']
        visualizer.auto_run = self.config['auto_run']
        visualizer.auto_close = self.config['auto_close']
        
        # 计算并设置特征重要性和信息增益
        feature_importance = self.calculate_feature_importance(feature_names)
        info_gain = self.calculate_information_gain(X, y, feature_names)
        visualizer.feature_importance = feature_importance
        visualizer.info_gain = info_gain
    
    def get_model_name(self):
        """
        获取模型名称
        
        返回:
        - 模型名称字符串
        """
        return "决策树"
    
    def calculate_feature_importance(self, feature_names):
        """
        计算并返回决策树的特征重要性
        
        参数:
        - feature_names: 特征名称列表
        
        返回:
        - 特征重要性字典
        """
        importances = self.model.feature_importances_
        return {name: imp for name, imp in zip(feature_names, importances)}
    
    def calculate_information_gain(self, X, y, feature_names):
        """
        计算每个特征的信息增益
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - feature_names: 特征名称列表
        
        返回:
        - 信息增益字典
        """
        # 使用mutual_info_classif计算互信息（作为信息增益的估计）
        mutual_info = mutual_info_classif(X, y, discrete_features=False, random_state=self.config['random_state'])
        return {name: gain for name, gain in zip(feature_names, mutual_info)}
    
    def print_model_info(self, X, y, feature_names):
        """
        打印决策树模型信息
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - feature_names: 特征名称
        """
        print(f"决策树配置:")
        print(f"  最大深度: {self.config['max_depth']}")
        print(f"  最大训练步数: {self.config['max_steps']}")
        
        # 计算信息增益
        print("计算特征信息增益...")
        info_gain = self.calculate_information_gain(X, y, feature_names)
        print("特征信息增益:")
        for name, gain in sorted(info_gain.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {gain:.4f}")
        
        # 计算特征重要性
        print("计算特征重要性...")
        feature_importance = self.calculate_feature_importance(feature_names)
        print("特征重要性 (Gini重要性):")
        for name, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {importance:.4f}")