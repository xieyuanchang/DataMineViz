from visualization_utils import BaseModelVisualizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class LogisticRegressionVisualizer(BaseModelVisualizer):
    """
    逻辑回归模型的可视化器
    继承自BaseModelVisualizer，实现逻辑回归特定的可视化功能
    """
    def __init__(self):
        super().__init__()
        # 初始化逻辑回归特定的属性
        self.learning_rate = 0.01  # 学习率
        self.feature_names = ['特征 1', '特征 2']  # 默认特征名称
        self.target_names = ['类别 0', '类别 1']  # 默认目标变量名称
        self.loss_history = []  # 记录损失值历史
        self.coef_history = []  # 记录系数历史
        self.intercept_history = []  # 记录截距历史
    
    def set_learning_rate(self, learning_rate):
        """设置学习率"""
        self.learning_rate = learning_rate
    
    def set_feature_names(self, feature_names):
        """设置特征名称"""
        self.feature_names = feature_names
    
    def set_target_names(self, target_names):
        """设置目标变量名称"""
        self.target_names = target_names
    
    def fit_step_by_step(self):
        """
        逐步训练逻辑回归模型
        模拟梯度下降过程，随着训练步数增加，逐步优化模型参数
        """
        if self.step >= self.max_steps or self.model is None:
            return
        
        # 初始化模型参数（如果是第一步）
        if self.step == 0:
            # 确保模型已初始化
            if not hasattr(self.model, 'coef_'):
                # 强制模型使用初始参数
                self.model = LogisticRegression(penalty=None, solver='lbfgs', random_state=42)
                self.model.coef_ = np.zeros((1, 2))
                self.model.intercept_ = np.zeros(1)
                self.model.classes_ = np.array([0, 1])
        
        # 计算当前使用的样本比例
        sample_ratio = min((self.step + 1) / self.max_steps, 1.0)
        n_samples = int(sample_ratio * len(self.X_train))
        
        # 获取当前样本子集
        X_subset = self.X_train[:n_samples]
        y_subset = self.y_train[:n_samples]
        
        # 计算梯度并更新参数
        # 1. 计算预测概率
        z = np.dot(X_subset, self.model.coef_.T) + self.model.intercept_
        y_pred = 1 / (1 + np.exp(-z))
        
        # 2. 计算梯度
        gradient_coef = np.dot((y_pred.flatten() - y_subset).T, X_subset) / n_samples
        gradient_intercept = np.mean(y_pred.flatten() - y_subset)
        
        # 3. 更新参数
        self.model.coef_ -= self.learning_rate * gradient_coef.reshape(1, -1)
        self.model.intercept_ -= self.learning_rate * gradient_intercept
        
        # 4. 计算并记录损失值（交叉熵损失）
        loss = -np.mean(y_subset * np.log(y_pred.flatten() + 1e-15) + 
                      (1 - y_subset) * np.log(1 - y_pred.flatten() + 1e-15))
        
        # 记录训练历史
        self.loss_history.append(loss)
        self.coef_history.append(self.model.coef_.copy())
        self.intercept_history.append(self.model.intercept_.copy())
        
        # 保存当前状态（训练样本比例、损失值、训练得分）
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test) if self.X_test is not None else 0
        self.train_iterations.append((sample_ratio, loss, train_score, test_score))
        
        self.step += 1
    
    def predict_visualization(self):
        """
        可视化逻辑回归的预测过程
        返回当前测试样本及其预测结果
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
        显示每次训练使用的样本比例
        """
        if self.X_train is not None and self.y_train is not None:
            # 计算当前训练步骤使用的样本比例
            if not self._is_training_complete:
                sample_ratio = min((self.step + 1) / self.max_steps, 1.0)
                n_current_samples = int(sample_ratio * len(self.X_train))
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
                # 训练阶段显示使用样本比例
                ax.text(0.02, 0.98, f'当前训练样本比例: {int(sample_ratio * 100)}%', 
                        transform=ax.transAxes, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8), 
                        verticalalignment='top', horizontalalignment='left')
                # 显示当前损失值
                if len(self.loss_history) > 0:
                    ax.text(0.02, 0.92, f'当前损失值: {self.loss_history[-1]:.4f}', 
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
        绘制逻辑回归的决策边界（直线）
        """
        if self.step == 0 or self.model is None:
            return
        
        # 获取模型参数
        w1, w2 = self.model.coef_[0]
        b = self.model.intercept_[0]
        
        # 确定边界范围
        x_min, x_max = self.X[:, 0].min() - 10000, self.X[:, 0].max() + 10000  # 为收入添加适当的边距
        y_min, y_max = self.X[:, 1].min() - 20, self.X[:, 1].max() + 20  # 为信用评分添加适当的边距
        
        # 创建网格
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1000),  # 收入网格步长为1000
                             np.arange(y_min, y_max, 1))  # 信用评分网格步长为1
        
        # 预测网格点的类别
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界（概率为0.5的等值线）
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        
        # 绘制决策线（仅当特征系数不为0时）
        if w2 != 0:
            # 计算决策线的y值
            x_values = np.linspace(x_min, x_max, 100)
            y_values = (-w1 * x_values - b) / w2
            
            # 只显示在数据范围内的部分
            valid_indices = (y_values >= y_min) & (y_values <= y_max)
            if np.any(valid_indices):
                ax.plot(x_values[valid_indices], y_values[valid_indices], 'k-', linewidth=2, label='决策边界')
        
        # 设置坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    def plot_model_structure(self, ax):
        """
        绘制逻辑回归的模型参数和训练进度
        显示权重、偏置、损失函数曲线和模型公式
        """
        if self.step == 0:
            # 初始状态，显示模型介绍
            ax.text(0.5, 0.5, '逻辑回归模型\n点击"下一步"开始训练', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round,pad=1'))
            ax.axis('off')
            return
        
        # 清除轴
        ax.clear()
        
        # 判断是否是训练完成或预测完成状态
        is_final_state = self._is_training_complete and (self.X_test is None or self.test_sample_index >= len(self.X_test))
        
        if is_final_state:
            # 最终状态：重点显示模型公式和参数
            self._plot_final_model_info(ax)
        else:
            # 训练或预测过程中：显示常规信息
            self._plot_training_info(ax)
        
        # 隐藏主轴
        ax.axis('off')
    
    def _plot_training_info(self, ax):
        """
        绘制训练过程中的信息
        """
        # 清除轴
        ax.clear()
        
        # 在父级ax内直接绘制信息，避免创建覆盖整个figure的子图
        if self.model is not None:
            weights = self.model.coef_[0]
            intercept = self.model.intercept_[0]
            
            # 1. 显示权重和偏置（左上角区域）
            weights_ax = ax.inset_axes([0.05, 0.55, 0.45, 0.4])
            bars = weights_ax.bar([0, 1], weights, color=['blue', 'green'])
            weights_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                weights_ax.text(bar.get_x() + bar.get_width()/2., height, 
                              f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top')
            
            # 添加截距信息
            weights_ax.text(0.5, 0.95, f'截距: {intercept:.4f}', 
                          transform=weights_ax.transAxes, ha='center', 
                          bbox=dict(facecolor='white', alpha=0.8))
            
            # 设置刻度标签
            weights_ax.set_xticks([0, 1])
            weights_ax.set_xticklabels([f'{self.feature_names[0]}权重', 
                                      f'{self.feature_names[1]}权重'], fontsize=8)
            weights_ax.set_title('模型参数', fontsize=10)
            weights_ax.grid(True, alpha=0.3)
        
        # 2. 显示损失函数曲线（右上角区域）
        if len(self.loss_history) > 0:
            loss_ax = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
            loss_ax.plot(range(len(self.loss_history)), self.loss_history, 'r-')
            loss_ax.set_xlabel('训练步数', fontsize=8)
            loss_ax.set_ylabel('损失值', fontsize=8)
            loss_ax.set_title('损失函数曲线', fontsize=10)
            loss_ax.grid(True, alpha=0.3)
            loss_ax.tick_params(axis='both', labelsize=8)
        
        # 3. 显示训练和测试准确率（左下角区域）
        if len(self.train_iterations) > 0:
            accuracy_ax = ax.inset_axes([0.05, 0.05, 0.45, 0.4])
            iterations = range(len(self.train_iterations))
            train_scores = [item[2] * 100 for item in self.train_iterations]  # 转换为百分比
            test_scores = [item[3] * 100 for item in self.train_iterations]
            
            accuracy_ax.plot(iterations, train_scores, 'b-', label='训练准确率')
            accuracy_ax.plot(iterations, test_scores, 'g-', label='测试准确率')
            accuracy_ax.set_xlabel('训练步数', fontsize=8)
            accuracy_ax.set_ylabel('准确率 (%)', fontsize=8)
            accuracy_ax.set_title('准确率变化', fontsize=10)
            accuracy_ax.legend(fontsize=8)
            accuracy_ax.grid(True, alpha=0.3)
            accuracy_ax.set_ylim(0, 100)
            accuracy_ax.tick_params(axis='both', labelsize=8)
        
        # 4. 显示权重变化趋势（右下角区域）
        if len(self.coef_history) > 0:
            coef_trend_ax = ax.inset_axes([0.55, 0.05, 0.4, 0.4])
            coef1_values = [coef[0, 0] for coef in self.coef_history]
            coef2_values = [coef[0, 1] for coef in self.coef_history]
            
            coef_trend_ax.plot(range(len(coef1_values)), coef1_values, 'b-', label=f'{self.feature_names[0]}权重')
            coef_trend_ax.plot(range(len(coef2_values)), coef2_values, 'g-', label=f'{self.feature_names[1]}权重')
            coef_trend_ax.set_xlabel('训练步数', fontsize=8)
            coef_trend_ax.set_ylabel('权重值', fontsize=8)
            coef_trend_ax.set_title('权重变化趋势', fontsize=10)
            coef_trend_ax.legend(fontsize=8)
            coef_trend_ax.grid(True, alpha=0.3)
            coef_trend_ax.tick_params(axis='both', labelsize=8)
    
    def _plot_final_model_info(self, ax):
        """
        绘制最终的模型信息，重点显示模型公式和参数
        """
        if self.model is None:
            return
        
        # 清除轴
        ax.clear()
        
        weights = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        
        # 1. 在父级ax内显示模型公式（上部区域）
        formula_ax = ax.inset_axes([0.1, 0.55, 0.8, 0.4])
        formula_ax.axis('off')
        
        # 构建逻辑回归公式
        linear_combination = f"z = {weights[0]:.4f} × {self.feature_names[0]} + {weights[1]:.4f} × {self.feature_names[1]} + {intercept:.4f}"
        sigmoid = "y = 1 / (1 + e^(-z))"
        decision_rule = "如果 y ≥ 0.5，则预测为正类；否则为负类"
        
        formula_text = (f"\n逻辑回归模型公式\n\n"  
                      f"线性组合: {linear_combination}\n"  
                      f"Sigmoid函数: {sigmoid}\n"  
                      f"决策规则: {decision_rule}\n")
        
        formula_ax.text(0.5, 0.5, formula_text, ha='center', va='center', fontsize=11, 
                       bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=1'))
        
        # 2. 在父级ax内显示参数详情（下部区域）
        params_ax = ax.inset_axes([0.1, 0.05, 0.8, 0.4])
        params_ax.axis('off')
        
        # 创建参数表格
        param_data = [
            ['参数名', '值'],
            [f'{self.feature_names[0]}权重', f'{weights[0]:.6f}'],
            [f'{self.feature_names[1]}权重', f'{weights[1]:.6f}'],
            ['偏置 (截距)', f'{intercept:.6f}'],
            ['最终损失值', f'{self.loss_history[-1]:.6f}' if len(self.loss_history) > 0 else '-']
        ]
        
        # 绘制表格
        table = params_ax.table(cellText=param_data, loc='center', cellLoc='center', 
                               bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.3)
        
        # 设置表头样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
            else:
                cell.set_edgecolor('#CCCCCC')
        
        # 添加标题
        ax.text(0.5, 0.98, '最终模型参数', ha='center', va='top', fontsize=12, transform=ax.transAxes)
    
    def highlight_prediction_path(self, test_point, pred_class, pred_proba):
        """
        高亮显示逻辑回归的预测过程
        显示测试点和其在特征空间中的位置
        """
        # 绘制测试点
        self.ax1.scatter(test_point[0], test_point[1], c='yellow', marker='*', s=300, 
                         edgecolors='black', linewidths=2, label='测试样本')
        
        # 获取当前样本的索引
        current_index = self.test_sample_index - 1
        
        # 检查预测是否正确
        is_correct = False
        if self.y_test is not None and current_index >= 0 and current_index < len(self.y_test):
            true_class = self.y_test[current_index]
            is_correct = (str(pred_class) == str(true_class))
        
        # 在图上标注预测结果
        result_text = f'预测: {self.target_names[pred_class]}\n'
        if is_correct and self.y_test is not None:
            result_text += f'真实: {self.target_names[true_class]}\n'
            result_text += f'[正确]\n'
        else:
            result_text += f'[错误]\n' if self.y_test is not None else ''
        result_text += f'概率: {pred_proba[pred_class]:.2f}'
        
        # 根据预测是否正确设置不同的背景色
        bbox_color = 'lightgreen' if is_correct else 'lightcoral'
        
        self.ax1.text(test_point[0] + 0.1, test_point[1] + 0.1, 
                     result_text, fontsize=10, bbox=dict(facecolor=bbox_color, alpha=0.8))
        
        # 添加图例
        self.ax1.legend(loc='best')
    
    def _setup_axes_limits(self, ax):
        """
        设置坐标轴范围，确保决策边界清晰可见
        """
        if self.X is not None:
            x_min, x_max = self.X[:, 0].min() - 10000, self.X[:, 0].max() + 10000  # 为收入添加适当的边距
            y_min, y_max = self.X[:, 1].min() - 20, self.X[:, 1].max() + 20  # 为信用评分添加适当的边距
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 设置坐标轴标签，优先使用特征名称
            if self.feature_names and len(self.feature_names) >= 2:
                ax.set_xlabel(self.feature_names[0])
                ax.set_ylabel(self.feature_names[1])
            else:
                ax.set_xlabel('特征 1')
                ax.set_ylabel('特征 2')