import numpy as np
import pandas as pd
import os

def create_loan_approval_data(n_samples=200):
    """
    创建房屋贷款审批数据集
    
    参数:
    - n_samples: 样本数量
    
    返回:
    - X: 特征矩阵
    - y: 目标变量
    - feature_names: 特征名称列表
    - target_names: 目标变量名称列表
    """
    np.random.seed(42)
    
    # 生成基础数据
    incomes = np.maximum(30000, np.minimum(300000, np.random.normal(loc=120000, scale=30000, size=n_samples)))
    credit_scores = np.maximum(300, np.minimum(850, np.random.normal(loc=680, scale=70, size=n_samples)))
    
    # 计算批准概率并生成结果
    income_factor = np.clip((incomes - 80000) / 100000, 0, 1) * 0.6
    credit_factor = np.clip((credit_scores - 650) / 150, 0, 1) * 0.3
    random_factor = np.random.random(n_samples) * 0.1
    approval_prob = income_factor + credit_factor + random_factor
    approvals = (approval_prob > 0.5).astype(int)
    
    # 构建并保存数据集
    feature_names = ['收入', '信用评分']
    target_names = ['拒绝', '批准']
    X = np.column_stack((incomes, credit_scores))
    y = approvals
    
    # 确保data目录存在并保存数据
    if not os.path.exists('data'):
        os.makedirs('data')
    
    df = pd.DataFrame(X, columns=feature_names)
    df['贷款结果'] = df.index.map(lambda i: target_names[y[i]])
    df.to_csv('data/loan_approval_data.csv', index=False, encoding='utf-8-sig')
    
    return X, y, feature_names, target_names

def load_loan_approval_data(file_path='data/loan_approval_data.csv'):
    """
    从CSV文件加载房屋贷款审批数据集
    
    参数:
    - file_path: CSV文件路径
    
    返回:
    - X: 特征矩阵
    - y: 目标变量
    - feature_names: 特征名称列表
    - target_names: 目标变量名称列表
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取特征和目标变量
    feature_names = ['收入', '信用评分']
    target_names = ['拒绝', '批准']
    
    X = df[feature_names].values
    # 将目标变量名称映射回数字
    y = df['贷款结果'].map({target_names[0]: 0, target_names[1]: 1}).values
    
    return X, y, feature_names, target_names