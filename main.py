import numpy as np
from sklearn.model_selection import train_test_split
from data_generation import load_loan_approval_data
from strategies.decision_tree_strategy import DecisionTreeStrategy
from strategies.logistic_regression_strategy import LogisticRegressionStrategy

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ============== 系统全局配置 ==============
# 1. 模型选择配置
# MODEL_TYPE 可选值："decision_tree" 或 "logistic_regression"
MODEL_TYPE = "decision_tree"  # 默认演示决策树模型
# 2. 执行模式配置
# AUTO_RUN 控制是否自动运行动画，无需手动点击"开始"按钮
# True: 自动开始运行动画
# False: 需要手动点击"开始"按钮启动
AUTO_RUN = True

# 3. 窗口控制配置
# AUTO_CLOSE 控制预测完成后是否自动关闭窗口
# True: 预测完成后自动关闭窗口
# False: 预测完成后保持窗口打开，需要手动关闭
AUTO_CLOSE = True

# 4. 决策树特定配置
# DECISION_TREE_MAX_DEPTH 决策树的最大深度
DECISION_TREE_MAX_DEPTH = 3
# DECISION_TREE_MAX_STEPS 决策树训练的最大步数
DECISION_TREE_MAX_STEPS = 5

# 5. 逻辑回归特定配置
# LOGISTIC_REGRESSION_MAX_STEPS 逻辑回归训练的最大步数
LOGISTIC_REGRESSION_MAX_STEPS = 15
# LOGISTIC_REGRESSION_LEARNING_RATE 逻辑回归的学习率
LOGISTIC_REGRESSION_LEARNING_RATE = 0.01

# 6. 数据集配置
# DATA_FILE_PATH 数据集文件路径
DATA_FILE_PATH = 'data/loan_approval_data.csv'
# TEST_SIZE 测试集比例
TEST_SIZE = 0.2
# RANDOM_STATE 随机种子，保证结果可复现
RANDOM_STATE = 42

class ModelStrategyFactory:
    """
    模型策略工厂
    负责创建不同类型的模型策略
    """
    
    @staticmethod
    def create_strategy(model_type):
        """
        创建指定类型的模型策略
        
        参数:
        - model_type: 模型类型，"decision_tree"或"logistic_regression"
        
        返回:
        - ModelStrategy 实例
        """
        if model_type.lower() == "decision_tree":
            return DecisionTreeStrategy()
        elif model_type.lower() == "logistic_regression":
            return LogisticRegressionStrategy()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

def main():
    """
    主函数：演示决策树或逻辑回归模型执行过程可视化
    根据全局配置变量控制程序行为
    """
    print("===== 机器学习模型可视化系统 =====")
    print(f"全局配置:")
    print(f"  模型类型: {MODEL_TYPE}")
    print(f"  自动运行: {'是' if AUTO_RUN else '否'}")
    print(f"  自动关闭: {'是' if AUTO_CLOSE else '否'}")
    print(f"  数据集: {DATA_FILE_PATH}")
    print(f"  测试集比例: {TEST_SIZE * 100}%")
    
    # 创建模型策略
    strategy = ModelStrategyFactory.create_strategy(MODEL_TYPE)
    model_name = strategy.get_model_name()
    
    print(f"\n{model_name}可视化开始...")
    print(f"第一阶段: 展示{model_name}逐步构建过程")
    print(f"第二阶段: 展示{model_name}对测试样本的预测过程")
    print("请稍等，正在初始化...")
    
    # 准备策略配置
    strategy_config = {
        'random_state': RANDOM_STATE,
        'auto_run': AUTO_RUN,
        'auto_close': AUTO_CLOSE
    }
    
    # 根据模型类型添加特定配置
    if MODEL_TYPE.lower() == "decision_tree":
        strategy_config.update({
            'max_depth': DECISION_TREE_MAX_DEPTH,
            'max_steps': DECISION_TREE_MAX_STEPS
        })
    else:  # logistic_regression
        strategy_config.update({
            'max_steps': LOGISTIC_REGRESSION_MAX_STEPS,
            'learning_rate': LOGISTIC_REGRESSION_LEARNING_RATE
        })
    
    # 初始化策略
    strategy.initialize(strategy_config)
    
    # 从CSV文件加载数据
    print("从CSV文件加载数据...")
    X, y, feature_names, target_names = load_loan_approval_data(file_path=DATA_FILE_PATH)
    
    # 打印数据集统计信息
    print(f"数据集包含 {len(X)} 个申请样本")
    approval_rate = np.mean(y) * 100
    print(f"批准率: {approval_rate:.1f}%")
    avg_income = np.mean(X[:, 0])
    avg_credit = np.mean(X[:, 1])
    print(f"平均收入: {int(avg_income):,} 元")
    print(f"平均信用评分: {avg_credit:.1f}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 创建模型
    print(f"创建{model_name}模型...")
    model = strategy.create_model()
    
    # 训练模型
    strategy.train_model(X_train, y_train)
    
    # 打印模型信息（决策树会在这里计算信息增益）
    strategy.print_model_info(X, y, feature_names)
    
    # 创建和配置可视化器
    visualizer = strategy.create_visualizer()
    strategy.configure_visualizer(visualizer, X, y, X_train, X_test, y_train, y_test, feature_names, target_names)
    
    # 初始化可视化
    visualizer.initialize_visualization()
    
    # 运行动画
    print("启动可视化动画...")
    visualizer.run_animation()
    
    print("可视化完成")

if __name__ == "__main__":
    main()