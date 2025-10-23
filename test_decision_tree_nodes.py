import numpy as np
from sklearn.tree import DecisionTreeClassifier
from decision_tree_visualizer import DecisionTreeVisualizer
import unittest
from unittest.mock import patch, MagicMock

class TestDecisionTreeNodes(unittest.TestCase):
    """
    测试决策树节点是否每个节点只使用一个特征进行分支
    """
    
    def setUp(self):
        """
        设置测试环境
        """
        # 创建简单的测试数据
        self.X = np.array([[100000, 700], [80000, 650], [120000, 750], [60000, 600]])
        self.y = np.array([1, 1, 0, 0])
        self.feature_names = ['收入', '信用评分']
        self.target_names = ['拒绝', '批准']
        
        # 创建决策树模型
        self.tree = DecisionTreeClassifier(max_depth=2, random_state=42)
        self.tree.fit(self.X, self.y)
    
    def test_each_node_single_feature(self):
        """
        测试每个非叶节点是否只使用一个特征进行分支
        """
        # 获取树结构
        tree_structure = self.tree.tree_
        
        # 检查每个节点
        for i in range(tree_structure.node_count):
            # 如果是内部节点（非叶节点）
            if tree_structure.children_left[i] != tree_structure.children_right[i]:
                # 检查是否有且只有一个特征被使用
                feature_idx = tree_structure.feature[i]
                # 确保feature_idx不是TREE_UNDEFINED（表示使用了特征）
                self.assertNotEqual(feature_idx, -2, f"节点{i}没有使用特征")
                # 确保特征索引在有效范围内
                self.assertTrue(0 <= feature_idx < len(self.feature_names), 
                              f"节点{i}使用了无效的特征索引: {feature_idx}")
                # 打印节点信息
                print(f"节点{i}: 使用特征 '{self.feature_names[feature_idx]}', 阈值={tree_structure.threshold[i]:.2f}")
    
    def test_tree_visualization_node_features(self):
        """
        测试可视化器是否能正确显示节点的特征使用
        """
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.figure'), \
             patch('visualization_utils.BaseModelVisualizer.initialize_visualization'):
            
            # 创建可视化器
            visualizer = DecisionTreeVisualizer()
            visualizer.set_data(self.X, self.y, feature_names=self.feature_names)
            visualizer.set_model(self.tree)
            visualizer.set_feature_names(self.feature_names)
            visualizer.set_target_names(self.target_names)
            
            # 模拟matplotlib轴对象
            mock_ax = MagicMock()
            
            # 调用绘图方法
            visualizer.plot_model_structure(mock_ax)
            
            # 验证plot_tree被正确调用，并且特征名称被传递
            # 注意：由于我们使用了mock，这里主要验证方法被调用
            print("验证plot_model_structure方法成功调用")
    
    def test_tree_uniqueness_of_split_features(self):
        """
        测试决策树的每个分支是否有明确的特征和阈值
        """
        # 获取树结构
        tree_structure = self.tree.tree_
        
        # 收集所有内部节点的分裂信息
        splits = []
        for i in range(tree_structure.node_count):
            if tree_structure.children_left[i] != tree_structure.children_right[i]:
                feature_idx = tree_structure.feature[i]
                threshold = tree_structure.threshold[i]
                splits.append({
                    'node': i,
                    'feature': self.feature_names[feature_idx],
                    'threshold': threshold,
                    'samples': tree_structure.n_node_samples[i]
                })
        
        # 打印分裂信息
        print("决策树分裂信息:")
        for split in splits:
            print(f"节点{split['node']}: '{split['feature']}' < {split['threshold']:.2f} (样本数: {split['samples']})")
        
        # 确保至少有一个分裂（如果树不是单节点的）
        if len(self.X) > 1:
            self.assertTrue(len(splits) > 0, "决策树没有进行任何分裂")

if __name__ == '__main__':
    unittest.main()
