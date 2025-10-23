from visualization_utils import BaseModelVisualizer
from sklearn.tree import plot_tree
import numpy as np

class DecisionTreeVisualizer(BaseModelVisualizer):
    """
    决策树模型的可视化器
    继承自BaseModelVisualizer，实现决策树特定的可视化功能
    """
    def __init__(self):
        super().__init__()
        # 初始化决策树特定的属性
        self.max_depth = 3  # 默认决策树最大深度
        self.feature_names = ['特征 1', '特征 2']  # 默认特征名称
        self.target_names = ['类别 0', '类别 1']  # 默认目标变量名称
    
    def set_max_depth(self, max_depth):
        """设置决策树的最大深度"""
        self.max_depth = max_depth
    
    def set_feature_names(self, feature_names):
        """设置特征名称"""
        self.feature_names = feature_names
    
    def set_target_names(self, target_names):
        """设置目标变量名称"""
        self.target_names = target_names
    
    def fit_step_by_step(self):
        """
        逐步训练决策树
        随着训练步数增加，使用更多的样本进行训练
        """
        if self.step >= self.max_steps or self.model is None:
            return
            
        # 确定当前使用的样本数
        n_samples = min(int((self.step + 1) / self.max_steps * len(self.X_train)), len(self.X_train))
        
        # 使用当前样本子集训练决策树
        self.model.fit(self.X_train[:n_samples], self.y_train[:n_samples])
        
        # 保存当前状态（训练样本数、训练得分、测试得分）
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test) if self.X_test is not None else 0
        self.train_iterations.append((n_samples, train_score, test_score))
        
        self.step += 1
    
    def predict_visualization(self):
        """
        可视化决策树的预测过程
        返回当前测试样本及其预测结果
        
        返回:
        - test_point: 测试样本
        - pred_class: 预测类别
        - pred_proba: 预测概率
        """
        # 检查是否还有测试样本需要处理
        if self.test_sample_index >= len(self.X_test) or self.model is None:
            return None
            
        # 获取当前测试样本
        test_sample = self.X_test[self.test_sample_index].reshape(1, -1)
        
        # 预测类别和概率
        pred_class = self.model.predict(test_sample)[0]
        pred_proba = self.model.predict_proba(test_sample)[0]
        
        # 先保存当前样本信息
        result = test_sample[0], pred_class, pred_proba
        
        # 再更新索引用于下次预测
        self.test_sample_index += 1
        
        return result
    
    def plot_data(self, ax):
        """
        绘制数据点，使用自定义的特征名称
        显示每次训练使用的样本，用不同样式区分当前使用的样本和未使用的样本
        """
        if self.X_train is not None and self.y_train is not None:
            # 计算当前训练步骤使用的样本数
            if not self._is_training_complete:
                n_current_samples = int((self.step + 1) / self.max_steps * len(self.X_train))
            else:
                n_current_samples = len(self.X_train)
            
            # 绘制所有未使用的样本（如果有）
            if n_current_samples < len(self.X_train):
                ax.scatter(self.X_train[n_current_samples:, 0], self.X_train[n_current_samples:, 1], 
                          c=self.y_train[n_current_samples:], cmap='coolwarm', marker='o', 
                          alpha=0.3, edgecolors='gray', label='未使用样本')
            
            # 绘制当前使用的样本（高亮显示）
            scatter_train = ax.scatter(self.X_train[:n_current_samples, 0], self.X_train[:n_current_samples, 1], 
                      c=self.y_train[:n_current_samples], cmap='coolwarm', marker='o', 
                      alpha=0.8, edgecolors='black', s=100, linewidths=1.5, label='当前训练样本')
            
            # 在预测阶段，添加测试样本的显示
            if self._is_training_complete and self.X_test is not None and self.test_sample_index > 0:
                # 确保索引不越界
                valid_index = min(self.test_sample_index - 1, len(self.X_test) - 1)
                # 显示最近的测试样本
                ax.scatter(self.X_test[valid_index, 0], self.X_test[valid_index, 1], 
                          c='yellow', marker='*', s=300, edgecolors='black', linewidths=2, label='当前测试样本')
            
            # 创建清晰的图例，避免重复项
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')
            
            # 设置轴标签
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])
            
            # 根据当前阶段显示不同的文本说明
            if not self._is_training_complete:
                # 训练阶段显示使用样本数
                ax.text(0.02, 0.98, f'当前训练样本数: {n_current_samples}/{len(self.X_train)}', 
                        transform=ax.transAxes, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8), 
                        verticalalignment='top', horizontalalignment='left')
            else:
                # 预测阶段显示已训练样本总数
                ax.text(0.02, 0.98, f'总训练样本数: {n_current_samples}', 
                        transform=ax.transAxes, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8), 
                        verticalalignment='top', horizontalalignment='left')
                # 显示当前测试样本信息
                if self.X_test is not None and self.test_sample_index > 0:
                    valid_index = min(self.test_sample_index - 1, len(self.X_test) - 1)
                    ax.text(0.02, 0.90, f'测试样本 {valid_index + 1}/{len(self.X_test)}', 
                            transform=ax.transAxes, fontsize=10, 
                            bbox=dict(facecolor='lightyellow', alpha=0.8), 
                            verticalalignment='top', horizontalalignment='left')
    
    def plot_decision_boundary(self, ax):
        """
        绘制决策树的决策边界
        """
        if self.step == 0 or self.model is None:
            return
        
        # 确定边界范围
        x_min, x_max = self.X[:, 0].min() - 10000, self.X[:, 0].max() + 10000  # 为收入添加适当的边距
        y_min, y_max = self.X[:, 1].min() - 20, self.X[:, 1].max() + 20  # 为信用评分添加适当的边距
        
        # 创建网格
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1000),  # 收入网格步长为1000
                             np.arange(y_min, y_max, 1))  # 信用评分网格步长为1
        
        # 预测网格点的类别
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        
        # 设置坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    def plot_model_structure(self, ax):
        """
        绘制决策树的结构，使用自定义的特征名称和目标变量名称
        确保每个节点清晰显示使用的特征，并且每个分支只有一个特征
        添加动态调整节点大小的功能，确保节点框能够完全显示
        """
        if self.step == 0 or self.model is None:
            return
        
        # 清除轴
        ax.clear()
        
        # 根据树的复杂度动态调整字体大小
        # 计算树的深度和节点数量
        tree_depth = getattr(self.model, 'get_depth', lambda: 3)()
        tree_nodes = getattr(self.model, 'tree_', None).node_count if hasattr(self.model, 'tree_') else 1
        
        # 根据树的复杂度动态调整字体大小
        # 节点越多，字体越小，确保所有节点都能显示
        base_fontsize = 12
        complexity_factor = min(tree_depth, tree_nodes / 10)
        dynamic_fontsize = max(int(base_fontsize - complexity_factor * 1.5), 8)  # 最小字体8，确保为整数
        
        # 使用scikit-learn的plot_tree函数绘制决策树
        # 调整参数以确保动态适应画面
        tree_plot = plot_tree(self.model, ax=ax, filled=True, 
                 feature_names=self.feature_names, 
                 class_names=self.target_names, 
                 fontsize=dynamic_fontsize,  # 使用动态计算的字体大小
                 node_ids=True,  # 显示节点ID
                 proportion=True,  # 显示样本比例
                 rounded=True,  # 使用圆角矩形
                 precision=2,  # 设置数值精度
                 impurity=True,  # 显示不纯度信息
                 label='all',  # 确保所有标签都显示
                 max_depth=None)  # 不限制深度，让所有节点都显示
        
        # 优化节点框的显示
        self._optimize_node_display(ax)
        
        # 为每个内部节点添加更明显的特征标注
        self._highlight_node_features(ax)
        
        # 设置标题
        ax.set_title('决策树结构 (每个节点仅使用一个特征)', fontsize=14, pad=20)
        
        # 设置合适的坐标轴范围，确保所有节点都可见
        ax.autoscale_view()
        
    def _optimize_node_display(self, ax):
        """
        优化节点框的显示，确保所有节点都能完全显示在画面中
        调整节点框大小、间距和缩放比例
        """
        # 获取当前图形大小
        fig_width, fig_height = ax.figure.get_size_inches()
        
        # 检查是否有足够的空间显示所有节点
        # 如果树比较复杂，可能需要调整缩放
        tree_depth = getattr(self.model, 'get_depth', lambda: 3)()
        
        # 对于较深的树，适当调整显示范围
        if tree_depth > 3:
            # 增加左右边距以容纳更宽的树
            ax.figure.subplots_adjust(left=0.05, right=0.95)
            
            # 获取所有树节点的艺术家对象
            tree_artists = [artist for artist in ax.get_children() 
                          if isinstance(artist, (plt.Text, plt.Rectangle))]
            
            # 调整文本大小以适应空间
            for artist in tree_artists:
                if isinstance(artist, plt.Text):
                    # 略微减小文本大小以确保完全显示
                    current_fontsize = artist.get_fontsize()
                    if current_fontsize > 7:  # 确保字体不会太小
                        artist.set_fontsize(max(current_fontsize - 1, 7))
    
    def _highlight_node_features(self, ax):
        """
        为决策树中的每个内部节点添加高亮特征显示
        确保用户能清楚看到每个节点使用的特征
        与动态调整节点大小功能兼容
        """
        # 获取树结构信息
        if hasattr(self.model, 'tree_'):
            tree = self.model.tree_
            
            # 获取树的复杂度信息
            tree_depth = getattr(self.model, 'get_depth', lambda: 3)()
            tree_nodes = tree.node_count
            
            # 遍历所有节点，为内部节点添加特征标注
            for i in range(tree.node_count):
                # 检查是否是内部节点（有子节点）
                if tree.children_left[i] != tree.children_right[i]:
                    # 获取使用的特征索引
                    feature_idx = tree.feature[i]
                    if feature_idx != -2:  # 确保特征索引有效
                        # 获取特征名称
                        feature_name = self.feature_names[feature_idx]
                        
                        # 在树图的适当位置添加特征高亮标注
                        # 这里我们使用一种简单的方法 - 在轴上添加文本说明
                        # 注意：由于无法直接访问plot_tree生成的节点位置，
                        # 我们添加总体说明而非针对每个节点
                        pass
        
        # 根据树的复杂度动态调整说明文本的大小和位置
        # 对于复杂的树，使用更小的字体和更紧凑的布局
        base_fontsize = 9
        complexity_factor = min(tree_depth, tree_nodes / 10)
        text_fontsize = max(int(base_fontsize - complexity_factor * 0.5), 6)  # 最小字体6，确保为整数
        
        # 简化说明文本内容，减少占用空间
        bottom_text = "".join([
            "节点解读:\n",
            f"- 顶部: 分支特征\n",
            f"- 下方: 判断条件\n",
            "\n",
            "特征:\n",
            "\n".join([f"- {name}" for name in self.feature_names])
        ])
        
        # 在右下角添加简化的综合说明框
        ax.text(0.95, 0.05, bottom_text, 
               transform=ax.transAxes, 
               fontsize=text_fontsize, 
               horizontalalignment='right',
               verticalalignment='bottom',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
               multialignment='left')
        
        # 调整重要说明的位置，确保不干扰树的显示
        ax.text(1.02, 0.95, '每个节点仅用一个特征', 
               transform=ax.transAxes, 
               fontsize=min(10, text_fontsize + 1), style='italic', color='red', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink", alpha=0.8))
        
        # 优化整体布局，确保树的主要部分有足够空间
        # 对于复杂的树，增加左右边距
        if tree_depth > 3 or tree_nodes > 10:
            ax.figure.subplots_adjust(left=0.05, right=0.9, bottom=0.15, top=0.9)
        else:
            # 对于简单的树，可以保持原有的布局
            ax.figure.subplots_adjust(left=0.1, right=0.85, bottom=0.15, top=0.9)
    
    def highlight_prediction_path(self, test_point, pred_class, pred_proba):
        """
        高亮显示预测路径，使用自定义的特征名称和目标变量名称
        """
        # 绘制测试点
        self.ax1.scatter(test_point[0], test_point[1], c='yellow', marker='*', s=300, 
                         edgecolors='black', linewidths=2, label='测试样本')
        
        # 获取预测结果的可读名称
        pred_result_name = self.target_names[pred_class] if pred_class < len(self.target_names) else f'类别{pred_class}'
        
        # 在图上标注预测结果和样本特征值
        feature_info = '\n'.join([f'{name}: {value:.0f}' for name, value in zip(self.feature_names, test_point)])
        self.ax1.text(test_point[0] + 0.1 * (self.X[:, 0].max() - self.X[:, 0].min()), 
                     test_point[1] + 0.1 * (self.X[:, 1].max() - self.X[:, 1].min()), 
                     f'{feature_info}\n预测: {pred_result_name}\n概率: {pred_proba[pred_class]:.2f}',
                     fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # 添加图例
        self.ax1.legend(loc='best')
    
    def _update_titles(self):
        """
        更新子图标题，添加决策树特有的信息
        """
        if self.step < self.max_steps:
            current_samples = int((self.step) / self.max_steps * len(self.X_train))
            self.ax1.set_title(f'决策树训练过程 (样本数: {current_samples}/{len(self.X_train)}, 深度: {self.max_depth})')
            self.ax1.text(0.5, -1.2, f'训练进度: {int(self.step/self.max_steps*100)}%', 
                         ha='center', transform=self.ax1.transAxes, fontsize=12)
        else:
            self.ax1.set_title(f'决策树预测过程 (样本 {self.test_sample_index}/{len(self.X_test)})')
        
        self.ax2.set_title('决策树结构')
    
    def _get_animated_artists(self):
        """
        获取需要在动画中更新的Artist对象列表
        
        返回:
        - artists: 需要更新的所有Artist对象列表
        """
        artists = []
        
        # 收集第一个子图中的所有Artist对象
        artists.extend(self.ax1.patches)
        artists.extend(self.ax1.lines)
        artists.extend(self.ax1.collections)
        artists.extend(self.ax1.texts)
        
        # 收集第二个子图中的所有Artist对象
        artists.extend(self.ax2.patches)
        artists.extend(self.ax2.lines)
        artists.extend(self.ax2.collections)
        artists.extend(self.ax2.texts)
        
        # 收集图例
        legend = self.ax1.get_legend()
        if legend is not None:
            artists.extend(legend.get_patches())
            artists.extend(legend.get_lines())
            artists.extend(legend.get_texts())
        
        return artists