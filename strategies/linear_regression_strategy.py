from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from linear_regression_visualizer import LinearRegressionVisualizer
from .model_strategy import ModelStrategy

class LinearRegressionStrategy(ModelStrategy):
    """
    线性回归策略实现
    包含线性回归模型的特定逻辑和配置
    """
    
    def __init__(self):
        self.model = None
        self.config = {}
        self.scaler = None
    
    def initialize(self, config):
        """
        初始化线性回归策略
        
        参数:
        - config: 配置字典，包含线性回归特定配置
        """
        # 线性回归默认配置
        default_config = {
            'max_steps': 15,
            'learning_rate': 0.01,
            'random_state': 42,
            'normalize_features': True,
            'auto_run': True,
            'auto_close': True
        }
        
        # 更新默认配置
        default_config.update(config)
        self.config = default_config
    
    def create_model(self):
        """
        创建线性回归模型
        
        返回:
        - LinearRegression 实例
        """
        self.model = LinearRegression()
        if self.config['normalize_features']:
            self.scaler = StandardScaler()
        return self.model
    
    def train_model(self, X_train, y_train):
        """
        训练线性回归模型
        
        参数:
        - X_train: 训练特征
        - y_train: 训练标签
        """
        # 如果配置了特征标准化，则进行标准化
        if self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train, y_train)
    
    def create_visualizer(self):
        """
        创建线性回归可视化器
        
        返回:
        - LinearRegressionVisualizer 实例
        """
        return LinearRegressionVisualizer()
    
    def configure_visualizer(self, visualizer, X, y, X_train, X_test, y_train, y_test, feature_names, target_name):
        """
        配置线性回归可视化器
        
        参数:
        - visualizer: 可视化器实例
        - X: 完整特征集
        - y: 完整标签集
        - X_train: 训练特征
        - X_test: 测试特征
        - y_train: 训练标签
        - y_test: 测试标签
        - feature_names: 特征名称
        - target_name: 目标名称（对于回归任务是单个名称）
        """
        # 设置数据集和模型
        visualizer.set_data(X, y, X_train, X_test, y_train, y_test)
        visualizer.set_model(self.model)
        visualizer.feature_names = feature_names
        visualizer.target_name = target_name  # 注意这里用的是target_name而不是target_names
        visualizer.max_steps = self.config['max_steps']
        visualizer.learning_rate = self.config['learning_rate']
        visualizer.auto_run = self.config['auto_run']
        visualizer.auto_close = self.config['auto_close']
        
        # 传递标准化器（如果有）
        visualizer.scaler = self.scaler
    
    def get_model_name(self):
        """
        获取模型名称
        
        返回:
        - 模型名称字符串
        """
        return "线性回归"
    
    def print_model_info(self, X, y, feature_names):
        """
        打印模型信息
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - feature_names: 特征名称
        """
        print(f"\n{self.get_model_name()}模型信息:")
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        
        if feature_names:
            print("特征名称:", feature_names)
        
        if self.model is not None:
            print("\n模型参数:")
            print(f"截距: {self.model.intercept_:.4f}")
            print("系数:")
            if feature_names and len(feature_names) == len(self.model.coef_):
                for i, (name, coef) in enumerate(zip(feature_names, self.model.coef_)):
                    print(f"  {name}: {coef:.4f}")
            else:
                for i, coef in enumerate(self.model.coef_):
                    print(f"  特征{i+1}: {coef:.4f}")
            
            # 计算并显示R²值（决定系数）
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
                r2_score = self.model.score(X_scaled, y)
            else:
                r2_score = self.model.score(X, y)
            print(f"\n决定系数 (R²): {r2_score:.4f}")
