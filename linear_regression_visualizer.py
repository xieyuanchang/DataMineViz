import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from visualization_utils import BaseModelVisualizer

class LinearRegressionVisualizer(BaseModelVisualizer):
    """
    线性回归模型可视化器
    展示线性回归模型的学习过程和预测结果
    """
    
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.01
        self.scaler = None
        self.target_name = "目标值"  # 对于回归任务，使用单个目标名称
        
        # 用于记录训练过程
        self.train_losses = []
        self.weights_history = []
        self.bias_history = []
        
        # 当前权重和偏置（用于可视化梯度下降过程）
        self.current_weights = None
        self.current_bias = None
        
        # 评估指标
        self.r2_scores = []
        
        # 预测历史记录，用于最终结果统计
        self._prediction_history = []
        
        # 标记是否已显示预测结果摘要
        self._summary_shown = False
        
        # 标记是否需要步进
        self._step_forward_required = False
    
    def set_target_name(self, target_name):
        """
        设置目标变量名称
        
        参数:
        - target_name: 目标变量名称
        """
        self.target_name = target_name
    
    def initialize_visualization(self):
        """
        初始化线性回归特定的可视化环境
        """
        # 调用基类的初始化方法，它会创建菜单栏和滚动条
        self.fig, self.ax1, self.ax2 = super().initialize_visualization()
        
        # 更新标题为线性回归相关
        self.fig.suptitle(f'线性回归模型执行过程可视化 - {self.target_name}', fontsize=16)
        
        return self.fig, self.ax1, self.ax2
    
    def fit_step_by_step(self):
        """
        逐步训练线性回归模型，用于动画演示
        """
        if self.step == 0:
            # 初始化权重和偏置
            n_features = self.X_train.shape[1]
            self.current_weights = np.zeros(n_features)
            self.current_bias = 0
            
            # 记录初始状态
            self.weights_history.append(self.current_weights.copy())
            self.bias_history.append(self.current_bias)
            
            # 计算初始损失
            loss = self._compute_mse_loss(self.X_train, self.y_train)
            self.train_losses.append(loss)
            
            # 计算初始R²
            r2 = self._compute_r2_score(self.X_train, self.y_train)
            self.r2_scores.append(r2)
        
        # 执行梯度下降更新
        if self.step < self.max_steps:
            # 前向传播
            y_pred = self._forward(self.X_train)
            
            # 计算梯度
            gradients = self._compute_gradients(self.X_train, self.y_train, y_pred)
            
            # 更新权重和偏置
            self.current_weights -= self.learning_rate * gradients['dw']
            self.current_bias -= self.learning_rate * gradients['db']
            
            # 记录历史
            self.weights_history.append(self.current_weights.copy())
            self.bias_history.append(self.current_bias)
            
            # 计算并记录损失
            loss = self._compute_mse_loss(self.X_train, self.y_train)
            self.train_losses.append(loss)
            
            # 计算并记录R²
            r2 = self._compute_r2_score(self.X_train, self.y_train)
            self.r2_scores.append(r2)
            
            # 绘制当前状态
            self._plot_regression_state()
            
            self.step += 1
        else:
            # 训练完成，使用最终权重更新模型
            self._is_training_complete = True
            if hasattr(self, 'btn_play_pause'):
                self.btn_play_pause.label.set_text('训练完成')
                self.btn_play_pause.color = 'lightgreen'
            
            # 将训练好的参数赋值给sklearn模型
            if self.model is not None:
                self.model.coef_ = self.current_weights
                self.model.intercept_ = self.current_bias
            
            # 最终绘制
            self._plot_regression_state()
    
    def predict_visualization(self):
        """
        可视化预测过程
        """
        if not self._is_training_complete or self.X_test is None or self.test_sample_index >= len(self.X_test):
            return
        
        # 选择一个测试样本
        test_sample = self.X_test[self.test_sample_index]
        true_value = self.y_test[self.test_sample_index]
        
        # 使用当前权重进行预测
        pred_value = self._forward(test_sample.reshape(1, -1))[0]
        
        # 绘制预测结果
        self._plot_prediction_result(test_sample, true_value, pred_value)
        
        # 记录预测历史，用于最终结果统计
        self._prediction_history.append((pred_value, true_value))
        
        self.test_sample_index += 1
        
        # 检查是否所有测试样本都已处理
        if self.test_sample_index >= len(self.X_test):
            self._is_prediction_complete = True
    
    def _forward(self, X):
        """
        前向传播计算预测值
        
        参数:
        - X: 输入特征
        
        返回:
        - y_pred: 预测值
        """
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            return np.dot(X_scaled, self.current_weights) + self.current_bias
        else:
            return np.dot(X, self.current_weights) + self.current_bias
    
    def _compute_mse_loss(self, X, y):
        """
        计算均方误差损失
        """
        y_pred = self._forward(X)
        return np.mean((y_pred - y) ** 2)
    
    def _compute_r2_score(self, X, y):
        """
        计算决定系数R²
        """
        y_pred = self._forward(X)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def _compute_gradients(self, X, y, y_pred):
        """
        计算梯度
        """
        m = len(y)
        
        # 如果使用了标准化器
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            dw = (1/m) * np.dot(X_scaled.T, (y_pred - y))
        else:
            dw = (1/m) * np.dot(X.T, (y_pred - y))
        
        db = (1/m) * np.sum(y_pred - y)
        
        return {'dw': dw, 'db': db}
    
    def _plot_regression_state(self):
        """
        绘制当前回归状态
        """
        # 左侧：数据散点图和拟合线
        self.ax1.clear()
        
        # 选择前两个特征进行可视化（如果有多个特征）
        if self.X_train.shape[1] >= 2:
            # 2D散点图 + 3D平面的投影
            scatter = self.ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, 
                                      cmap='viridis', alpha=0.6, s=50)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=self.ax1)
            cbar.set_label(self.target_name)
            
            # 设置轴标签
            if self.feature_names and len(self.feature_names) >= 2:
                self.ax1.set_xlabel(self.feature_names[0])
                self.ax1.set_ylabel(self.feature_names[1])
            else:
                self.ax1.set_xlabel('特征1')
                self.ax1.set_ylabel('特征2')
            
            self.ax1.set_title(f'训练数据分布 - 步骤 {self.step}/{self.max_steps}')
        else:
            # 1D散点图 + 拟合线
            X_line = np.linspace(self.X_train[:, 0].min(), self.X_train[:, 0].max(), 100).reshape(-1, 1)
            y_line = self._forward(X_line)
            
            self.ax1.scatter(self.X_train[:, 0], self.y_train, alpha=0.6, s=50)
            self.ax1.plot(X_line, y_line, color='red', linewidth=2)
            
            # 设置轴标签
            if self.feature_names:
                self.ax1.set_xlabel(self.feature_names[0])
            else:
                self.ax1.set_xlabel('特征')
            self.ax1.set_ylabel(self.target_name)
            
            self.ax1.set_title(f'回归拟合 - 步骤 {self.step}/{self.max_steps}')
        
        # 右侧：训练信息
        self.ax2.clear()
        
        # 创建子图区域
        gs = self.ax2.inset_axes([0.05, 0.55, 0.45, 0.4])  # 损失函数
        gs2 = self.ax2.inset_axes([0.55, 0.55, 0.4, 0.4])  # R²曲线
        gs3 = self.ax2.inset_axes([0.05, 0.05, 0.9, 0.45])  # 权重变化
        
        # 绘制损失函数
        gs.plot(self.train_losses, 'b-', linewidth=2)
        gs.set_xlabel('步骤')
        gs.set_ylabel('MSE损失')
        gs.set_title('损失函数')
        gs.grid(True, alpha=0.3)
        
        # 绘制R²曲线
        gs2.plot(self.r2_scores, 'g-', linewidth=2)
        gs2.set_xlabel('步骤')
        gs2.set_ylabel('R²')
        gs2.set_title('决定系数')
        gs2.grid(True, alpha=0.3)
        gs2.set_ylim(max(0, min(self.r2_scores) - 0.1), min(1, max(self.r2_scores) + 0.1))
        
        # 绘制权重变化
        weights_array = np.array(self.weights_history)
        for i in range(weights_array.shape[1]):
            if self.feature_names and i < len(self.feature_names):
                label = self.feature_names[i]
            else:
                label = f'特征{i+1}'
            gs3.plot(weights_array[:, i], label=label)
        
        # 绘制偏置
        gs3.plot(self.bias_history, 'k--', label='偏置')
        
        gs3.set_xlabel('步骤')
        gs3.set_ylabel('参数值')
        gs3.set_title('模型参数变化')
        gs3.legend(fontsize=8, loc='best')
        gs3.grid(True, alpha=0.3)
        
        # 设置主标题
        self.ax2.set_title('训练过程监控')
        self.ax2.axis('off')  # 关闭主轴
    
    def _plot_prediction_result(self, test_sample, true_value, pred_value):
        """
        绘制预测结果
        """
        # 在左侧轴上高亮显示测试样本
        if self.X_train.shape[1] >= 2:
            self.ax1.scatter(test_sample[0], test_sample[1], color='red', s=100, 
                            marker='*', edgecolors='black', linewidths=2, 
                            label=f'预测: {pred_value:.2f}\n真实: {true_value:.2f}')
            self.ax1.legend(fontsize=10)
        else:
            self.ax1.scatter(test_sample[0], true_value, color='red', s=100, 
                            marker='*', edgecolors='black', linewidths=2)
            self.ax1.scatter(test_sample[0], pred_value, color='blue', s=100, 
                            marker='x', edgecolors='black', linewidths=2)
            
            # 添加预测标签
            self.ax1.annotate(f'预测: {pred_value:.2f}', 
                            xy=(test_sample[0], pred_value),
                            xytext=(10, 10), textcoords='offset points')
            self.ax1.annotate(f'真实: {true_value:.2f}', 
                            xy=(test_sample[0], true_value),
                            xytext=(10, -15), textcoords='offset points')
        
        # 在右侧轴上显示预测信息
        error = abs(pred_value - true_value)
        error_percent = (error / true_value) * 100 if true_value != 0 else 0
        
        info_text = f"预测样本 #{self.test_sample_index}:\n"
        info_text += f"真实值: {true_value:.2f}\n"
        info_text += f"预测值: {pred_value:.2f}\n"
        info_text += f"绝对误差: {error:.2f}\n"
        info_text += f"相对误差: {error_percent:.2f}%"
        
        # 创建信息显示区域
        info_ax = self.ax2.inset_axes([0.1, 0.1, 0.8, 0.3])
        info_ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10)
        info_ax.axis('off')
    
    def run_animation(self):
        """
        运行动画，可视化训练过程
        调用基类的run_animation方法以避免创建多个窗口
        """
        # 设置动画参数
        interval = 500  # 500ms per frame
        
        # 计算需要的动画帧数
        n_frames = self.max_steps + (len(self.X_test) if self.X_test is not None else 0) + 2
        
        # 确保在调用基类方法前，所有必要的属性都已正确初始化
        # 注意：这里不再输出调试信息，因为我们已经确认问题在于多个实例
        
        # 调用基类的run_animation方法，避免重复创建窗口
        # 基类的run_animation方法会负责创建动画和调用plt.show()
        super().run_animation(frames=n_frames, interval=interval, blit=False)
    
    def plot_decision_boundary(self, ax):
        """
        实现基类要求的plot_decision_boundary方法
        对于线性回归，这里绘制回归线或回归平面
        """
        # 线性回归的可视化在_plot_regression_state方法中实现
        # 这里可以保留为空，因为我们在update_plot中会调用_plot_regression_state
        pass
    
    def plot_data(self, ax):
        """
        绘制数据点，重写基类方法以适应线性回归的需求
        """
        # 这个方法在基类的update_plot中被调用，但我们在线性回归中
        # 会在_plot_regression_state方法中完成所有绘图，所以这里可以留空
        pass
    
    def plot_model_structure(self, ax):
        """
        绘制模型结构，重写基类方法以适应线性回归的需求
        """
        # 这个方法在基类的update_plot中被调用，但我们在线性回归中
        # 会在_plot_regression_state方法中完成所有绘图，所以这里可以留空
        pass
    
    def update_plot(self, frame):
        """
        更新动画帧，重写基类方法
        这是动画的核心方法，每帧都会调用
        """
        # 确保按钮状态正确更新
        if hasattr(self, 'btn_play_pause'):
            self.btn_play_pause.label.set_text('自动播放' if self._manual_step_mode else '暂停')
        
        # 处理更新逻辑
        do_update = False
        
        # 自动模式下，只要没有暂停就执行更新
        if not self._manual_step_mode:
            do_update = not self._is_paused
        elif self._step_forward_required:
            # 手动模式下，如果需要步进，则执行更新
            do_update = True
        
        # 如果不需要更新，直接返回
        if not do_update:
            return self._get_animated_artists()
        
        # 预测完成后不清除图形，保持结果显示
        if not self._is_prediction_complete:
            # 清除当前绘图
            self.ax1.clear()
            self.ax2.clear()
        
        # 训练过程
        if not self._is_training_complete:
            # 执行一步训练
            self.fit_step_by_step()
            
            # 检查是否训练完成
            if self.step >= self.max_steps:
                self._is_training_complete = True
        
        # 在手动模式下，执行完更新后重置步进标志
        if self._manual_step_mode and do_update:
            self._step_forward_required = False
        
        # 1. 绘制数据点和回归线（线性回归特有的绘图）
        self._plot_regression_state()
        
        # 2. 当模型训练完成后，开始可视化预测过程
        if self._is_training_complete and self.X_test is not None and not self._is_prediction_complete:
            # 执行一步预测
            self.predict_visualization()
            
            # 检查是否预测完成
            if self.test_sample_index >= len(self.X_test):
                self._is_prediction_complete = True
        
        # 确保坐标轴设置正确
        self._setup_axes_limits(self.ax1)
        
        # 返回需要更新的Artist对象列表
        return self._get_animated_artists()
    
    def _get_animated_artists(self):
        """
        获取需要在动画中更新的Artist对象列表
        这是从基类继承或重写的方法
        """
        artists = []
        
        # 收集两个子图中的所有Artist对象
        if hasattr(self, 'ax1'):
            artists.extend(self.ax1.patches)
            artists.extend(self.ax1.lines)
            artists.extend(self.ax1.collections)
            artists.extend(self.ax1.texts)
        
        if hasattr(self, 'ax2'):
            artists.extend(self.ax2.patches)
            artists.extend(self.ax2.lines)
            artists.extend(self.ax2.collections)
            artists.extend(self.ax2.texts)
        
        return artists
    
    def _show_prediction_summary_direct(self):
        """
        直接显示预测结果摘要
        无论是否有预测历史记录，都显示模型性能信息
        """
        if not self._summary_shown:
            # 确保预测历史不为空，即使没有显式预测过程
            if not self._prediction_history and self.X_test is not None and len(self.X_test) > 0:
                # 如果预测历史为空但有测试数据，则进行预测
                for i in range(len(self.X_test)):
                    X_sample = self.X_test[i:i+1]
                    y_sample = self.y_test[i]
                    pred = self._forward(X_sample)[0]
                    self._prediction_history.append((pred, y_sample))
            
            # 收集预测结果数据
            predictions = []
            true_values = []
            absolute_errors = []
            
            # 从预测历史中提取数据
            for pred, true in self._prediction_history:
                predictions.append(pred)
                true_values.append(true)
                absolute_errors.append(abs(pred - true))
            
            # 显示预测结果摘要
            self._show_prediction_summary(predictions, true_values, absolute_errors)
            self._summary_shown = True
    
    def _show_prediction_summary(self, predictions=None, true_values=None, absolute_errors=None):
        """
        在控制台输出线性回归预测结果摘要
        
        参数:
        - predictions: 预测值列表
        - true_values: 真实值列表
        - absolute_errors: 绝对误差列表
        """
        print("\n===== 线性回归模型性能指标 =====")
        
        if hasattr(self, 'r2_scores') and len(self.r2_scores) > 0:
            # 计算样本数量
            train_count = len(self.X_train) if hasattr(self, 'X_train') and self.X_train is not None else 0
            test_count = len(self.X_test) if hasattr(self, 'X_test') and self.X_test is not None else 0
            
            # 打印模型性能指标
            print(f"R² Score:       {self.r2_scores[-1]:.6f}")
            print(f"最终损失:        {self.train_losses[-1]:.6f}")
            print(f"训练样本数:      {train_count}")
            print(f"测试样本数:      {test_count}")
            print(f"学习率:          {self.learning_rate}")
            print(f"训练步数:        {self.step}")
        
        # 如果有预测结果，计算并显示预测误差统计
        if predictions and true_values and absolute_errors:
            print("\n===== 预测误差统计 =====")
            # 计算平均绝对误差
            mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0
            # 计算最大绝对误差
            max_ae = max(absolute_errors) if absolute_errors else 0
            # 计算最小绝对误差
            min_ae = min(absolute_errors) if absolute_errors else 0
            
            print(f"平均绝对误差 (MAE): {mae:.6f}")
            print(f"最大绝对误差:       {max_ae:.6f}")
            print(f"最小绝对误差:       {min_ae:.6f}")
            
            # 打印前几个预测样本的详细信息
            print("\n===== 预测样本详细信息 =====")
            num_samples_to_show = min(5, len(predictions))  # 最多显示5个样本
            print(f"{'样本索引':<8} {'真实值':<10} {'预测值':<10} {'绝对误差':<10} {'相对误差(%)':<12}")
            print("-" * 55)
            
            for i in range(num_samples_to_show):
                pred = predictions[i]
                true = true_values[i]
                abs_error = absolute_errors[i]
                rel_error = (abs_error / true * 100) if true != 0 else float('inf')
                print(f"{i:<8} {true:<10.4f} {pred:<10.4f} {abs_error:<10.4f} {rel_error:<12.2f}")
                
            if len(predictions) > num_samples_to_show:
                print(f"... 还有 {len(predictions) - num_samples_to_show} 个样本未显示")
