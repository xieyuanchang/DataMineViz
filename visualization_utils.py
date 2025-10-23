import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class BaseModelVisualizer:
    """
    机器学习模型可视化的基类
    提供通用的可视化功能，可被不同的算法模型继承和扩展
    """
    def __init__(self):
        # 数据相关属性
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 模型相关属性
        self.model = None
        
        # 可视化相关属性
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.step = 0
        self.max_steps = 30  # 默认最大训练步数
        self.test_sample_index = 0
        self.train_iterations = []
        self.correct_predictions = 0  # 记录正确预测的数量
        self._is_prediction_complete = False  # 标记预测是否完成
        
        # 动画相关属性
        self.animation = None
        
        # 动画状态控制
        self._is_training_complete = False
        self._is_paused = False
        self._manual_step_mode = True  # 默认启用手动步进模式
        self._step_forward_required = False  # 标记是否需要前进一步
        
        # 特征名称
        self.feature_names = None
    
    def initialize_visualization(self):
        """初始化可视化环境"""
        # 创建图形，为按钮留出空间
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))
        self.fig.subplots_adjust(bottom=0.2)  # 增加底部边距，为按钮留出空间
        self.fig.suptitle('决策树模型执行过程可视化', fontsize=16)
        
        # 绑定按键事件处理函数
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # 添加控制按钮
        self._add_control_buttons()
        
        return self.fig, self.ax1, self.ax2
    
    def _add_control_buttons(self):
        """添加控制按钮"""
        # 创建前进一步按钮（移到左下角）
        self.ax_next_step = self.fig.add_axes([0.05, 0.05, 0.15, 0.075])
        self.btn_next_step = Button(self.ax_next_step, '下一步')
        self.btn_next_step.on_clicked(self._on_next_step_clicked)
        
        # 创建自动播放/暂停按钮（移到左下角，在下一步按钮旁边）
        self.ax_play_pause = self.fig.add_axes([0.25, 0.05, 0.15, 0.075])
        self.btn_play_pause = Button(self.ax_play_pause, '自动播放')
        self.btn_play_pause.on_clicked(self._on_play_pause_clicked)
    
    def _on_next_step_clicked(self, event):
        """处理下一步按钮点击事件"""
        # 确保进入手动步进模式
        self._manual_step_mode = True
        self._is_paused = True
        # 强制设置步进标志
        self._step_forward_required = True
        # 确保按钮状态更新
        if hasattr(self, 'btn_play_pause'):
            self.btn_play_pause.label.set_text('自动播放')
        
        # 检查是否预测已完成，如果完成则不进行更新
        if self._is_prediction_complete:
            return
        
        # 直接执行一步更新
        if not self._is_training_complete or (self.X_test is not None and self.test_sample_index < len(self.X_test)):
            # 清除当前绘图
            self.ax1.clear()
            self.ax2.clear()
            
            # 训练过程
            if not self._is_training_complete:
                self.fit_step_by_step()
                if self.step >= self.max_steps:
                    self._is_training_complete = True
            
            # 绘制数据点和决策边界
            self.plot_data(self.ax1)
            self.plot_decision_boundary(self.ax1)
            
            # 绘制模型结构
            self.plot_model_structure(self.ax2)
            
            # 预测过程
            if self._is_training_complete and self.X_test is not None and self.test_sample_index < len(self.X_test):
                result = self.predict_visualization()
                if result:
                    test_point, pred_class, pred_proba = result
                    self.highlight_prediction_path(test_point, pred_class, pred_proba)
                    
                    # 检查是否所有测试样本都已处理完
                    if self.test_sample_index >= len(self.X_test):
                        self._is_prediction_complete = True
                        # 显示预测成功率
                        self._show_prediction_summary()
            
            # 更新标题
            self._update_titles()
            
            # 设置坐标轴
            self._setup_axes_limits(self.ax1)
            
            # 立即触发图形重绘
            self.fig.canvas.draw_idle()
        
        # 重置步进标志
        self._step_forward_required = False
    
    def _on_play_pause_clicked(self, event):
        """处理播放/暂停按钮点击事件"""
        # 切换模式
        self._manual_step_mode = not self._manual_step_mode
        self._is_paused = self._manual_step_mode
        # 更新按钮文本
        self.btn_play_pause.label.set_text('自动播放' if self._manual_step_mode else '暂停')
        # 如果进入自动模式，清除步进标志
        if not self._manual_step_mode:
            self._step_forward_required = False
        # 立即触发图形更新
        self.fig.canvas.draw_idle()
        
    def _on_key_press(self, event):
        """
        处理键盘事件
        空格键暂停/继续动画
        q键退出动画
        n键前进一步
        """
        if event.key == ' ':
            # 空格键暂停/继续动画
            self._manual_step_mode = not self._manual_step_mode
            self._is_paused = self._manual_step_mode
            if hasattr(self, 'btn_play_pause'):
                self.btn_play_pause.label.set_text('自动播放' if self._manual_step_mode else '暂停')
        elif event.key == 'q':
            # q键退出动画
            plt.close(self.fig)
        elif event.key == 'n':
            # n键前进一步
            self._manual_step_mode = True
            self._is_paused = True
            self._step_forward_required = True
            # 确保按钮状态更新
            if hasattr(self, 'btn_play_pause'):
                self.btn_play_pause.label.set_text('自动播放')
    
    def set_data(self, X, y, X_train=None, X_test=None, y_train=None, y_test=None, feature_names=None):
        """设置数据"""
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
        # 重置训练和预测状态
        self.step = 0
        self.test_sample_index = 0
        self.train_iterations = []
        self.correct_predictions = 0
        self._is_training_complete = False
        self._is_prediction_complete = False
    
    def set_model(self, model):
        """设置模型"""
        self.model = model
    
    def fit_step_by_step(self):
        """
        逐步训练模型
        子类需要重写此方法
        """
        raise NotImplementedError("子类必须实现fit_step_by_step方法")
    
    def predict_visualization(self):
        """
        可视化模型的预测过程
        子类需要重写此方法
        
        返回:
        - test_point: 测试样本
        - pred_class: 预测类别
        - pred_proba: 预测概率
        """
        raise NotImplementedError("子类必须实现predict_visualization方法")
    
    def plot_data(self, ax):
        """
        绘制数据点
        子类可以重写此方法以自定义数据点的绘制
        """
        if self.X_train is not None and self.y_train is not None:
            ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, 
                      cmap='coolwarm', marker='o', alpha=0.6, edgecolors='black')
    
    def plot_decision_boundary(self, ax):
        """
        绘制决策边界
        子类需要重写此方法
        """
        raise NotImplementedError("子类必须实现plot_decision_boundary方法")
    
    def plot_model_structure(self, ax):
        """
        绘制模型结构
        子类需要重写此方法
        """
        raise NotImplementedError("子类必须实现plot_model_structure方法")
    
    def __init__(self):
        # 数据相关属性
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 模型相关属性
        self.model = None
        
        # 可视化相关属性
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.step = 0
        self.max_steps = 30  # 默认最大训练步数
        self.test_sample_index = 0
        self.train_iterations = []
        self.correct_predictions = 0  # 记录正确预测的数量
        self._is_prediction_complete = False  # 标记预测是否完成
        
        # 动画相关属性
        self.animation = None
        
        # 动画状态控制
        self._is_training_complete = False
        self._is_paused = False
        self._manual_step_mode = True  # 默认启用手动步进模式
        self._step_forward_required = False  # 标记是否需要前进一步
        
        # 特征名称
        self.feature_names = None
        
        # 新增属性：预测分组大小和自动关闭设置
        self.prediction_batch_size = 10  # 每批预测的样本数量
        self._auto_close = True  # 是否自动关闭窗口
        self._prediction_batch_processed = 0  # 记录当前批次已处理的样本数
        self._batch_correct_predictions = 0  # 当前批次正确预测数
        self._total_processed_samples = 0  # 总共处理的样本数
    
    def update_plot(self, frame):
        """
        更新动画帧
        这是动画的核心方法，每帧都会调用
        
        参数:
        frame: 当前帧索引
        
        返回:
        artists: 需要更新的Artist对象列表，用于blitting优化
        """
        # 添加调试信息
        print(f"[调试] 更新帧: {frame}")
        print(f"[调试] 当前状态 - 训练完成: {self._is_training_complete}, 预测完成: {self._is_prediction_complete}")
        print(f"[调试] 当前步骤: {self.step}, 测试样本索引: {self.test_sample_index}")
        
        # 确保按钮状态正确更新
        if hasattr(self, 'btn_play_pause'):
            self.btn_play_pause.label.set_text('自动播放' if self._manual_step_mode else '暂停')
        
        # 处理更新逻辑
        do_update = False
        
        # 手动模式下，不执行更新（已在按钮点击事件中直接处理）
        if not self._manual_step_mode:
            # 自动模式下，只要没有暂停就执行更新
            do_update = not self._is_paused
        
        # 如果不需要更新，直接返回
        if not do_update:
            print("[调试] 不执行更新，返回当前artists")
            return self._get_animated_artists()
            
        # 预测完成后不清除图形，保持结果显示
        if not self._is_prediction_complete:
            # 对于预测阶段，只有在开始新批次时才清除图形
            if self._is_training_complete and self._prediction_batch_processed == 0:
                print("[调试] 开始新批次，清除当前绘图")
                self.ax1.clear()
                self.ax2.clear()
            elif not self._is_training_complete:
                print("[调试] 清除当前绘图")
                self.ax1.clear()
                self.ax2.clear()
        
        # 训练过程
        if not self._is_training_complete:
            print("[调试] 执行训练步骤")
            self.fit_step_by_step()
            
            # 检查是否训练完成
            if self.step >= self.max_steps:
                self._is_training_complete = True
                print("[调试] 训练完成")
        
        # 在手动模式下，执行完更新后重置步进标志
        if self._manual_step_mode and do_update:
            self._step_forward_required = False
        
        # 1. 在ax1上绘制数据点和决策边界
        print("[调试] 绘制数据点和决策边界")
        self.plot_data(self.ax1)
        self.plot_decision_boundary(self.ax1)
        
        # 2. 在ax2上绘制模型结构
        print("[调试] 绘制模型结构")
        self.plot_model_structure(self.ax2)
        
        # 绘制训练进度（如果有）
        if hasattr(self, 'plot_training_progress') and len(self.train_iterations) > 0:
            self.plot_training_progress(self.ax2)
        
        # 3. 当模型训练完成后，开始可视化预测过程
        if self._is_training_complete and self.X_test is not None and not self._is_prediction_complete:
            print("[调试] 开始预测可视化过程（批量处理）")
            
            # 批量处理预测
            batch_start_index = self.test_sample_index
            batch_end_index = min(batch_start_index + self.prediction_batch_size, len(self.X_test))
            
            print(f"[调试] 处理批次: {batch_start_index}-{batch_end_index-1}/{len(self.X_test)}")
            
            # 重置批次计数器
            self._batch_correct_predictions = 0
            
            # 处理当前批次的所有样本
            while self.test_sample_index < batch_end_index:
                test_sample = self.X_test[self.test_sample_index].reshape(1, -1)
                
                # 预测类别和概率
                if hasattr(self, 'model') and self.model is not None:
                    pred_class = self.model.predict(test_sample)[0]
                    pred_proba = self.model.predict_proba(test_sample)[0]
                    
                    # 高亮显示当前样本的预测路径（只显示最新的一个）
                    if self.test_sample_index == batch_end_index - 1:  # 只高亮最后一个样本
                        print(f"[调试] 高亮显示样本: {self.test_sample_index}, 类别: {pred_class}")
                        self.highlight_prediction_path(test_sample[0], pred_class, pred_proba)
                    
                    # 计算当前样本的预测是否正确
                    if self.y_test is not None and self.test_sample_index < len(self.y_test):
                        true_label = self.y_test[self.test_sample_index]
                        # 使用字符串比较避免类型问题
                        is_correct = (str(pred_class) == str(true_label))
                        if is_correct:
                            self._batch_correct_predictions += 1
                            self.correct_predictions += 1
                            print(f"[调试] 样本 {self.test_sample_index} 预测正确!")
                        else:
                            print(f"[调试] 样本 {self.test_sample_index} 预测错误!")
                
                # 更新索引
                self.test_sample_index += 1
                self._prediction_batch_processed += 1
                self._total_processed_samples += 1
            
            # 在当前批次完成后显示批次信息
            print(f"[调试] 批次完成，正确预测数: {self._batch_correct_predictions}/{batch_end_index - batch_start_index}")
            
            # 添加批次信息到图表
            batch_info_text = (
                f"批次: {batch_start_index//self.prediction_batch_size + 1}\n"
                f"样本范围: {batch_start_index+1}-{batch_end_index}/{len(self.X_test)}\n"
                f"本批正确预测: {self._batch_correct_predictions}/{batch_end_index - batch_start_index}\n"
                f"累计正确预测: {self.correct_predictions}/{self.test_sample_index}"
            )
            self.ax1.text(0.02, 0.02, batch_info_text, 
                         transform=self.ax1.transAxes,
                         fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.8),
                         verticalalignment='bottom', horizontalalignment='left')
            
            # 重置批次处理计数
            self._prediction_batch_processed = 0
            
            # 检查是否所有测试样本都已处理完
            if self.test_sample_index >= len(self.X_test):
                self._is_prediction_complete = True
                print("[调试] 预测完成，显示预测总结")
                # 显示预测成功率
                self._show_prediction_summary()
                
                # 如果设置了自动关闭，在显示总结后延迟关闭
                # 使用try-except忽略可能的异常，因为关闭后程序已经完成
                if hasattr(self, '_auto_close') and self._auto_close:
                    print("[调试] 预测完成，准备自动关闭")
                    try:
                        # 给用户3秒时间查看结果
                        plt.pause(3)
                        plt.close(self.fig)
                        print("[调试] 窗口已关闭")
                    except Exception as e:
                        print(f"[调试] 关闭窗口时出现异常（可忽略）: {e}")
        
        # 更新标题
        print("[调试] 更新标题")
        self._update_titles()
        
        # 确保坐标轴设置正确
        self._setup_axes_limits(self.ax1)
        
        # 返回需要更新的Artist对象列表
        print("[调试] 返回更新后的artists")
        return self._get_animated_artists()
    
    def _get_animated_artists(self):
        """
        获取需要在动画中更新的Artist对象列表
        
        返回:
        artists: 需要更新的所有Artist对象列表
        """
        artists = []
        
        # 收集两个子图中的所有Artist对象
        artists.extend(self.ax1.patches)
        artists.extend(self.ax1.lines)
        artists.extend(self.ax1.collections)
        artists.extend(self.ax1.texts)
        
        artists.extend(self.ax2.patches)
        artists.extend(self.ax2.lines)
        artists.extend(self.ax2.collections)
        artists.extend(self.ax2.texts)
        
        return artists
    
    def _update_titles(self):
        """更新子图标题"""
        if not self._is_training_complete:
            self.ax1.set_title(f'训练进行中 (步骤 {self.step}/{self.max_steps})')
            self.ax1.text(0.5, -1.2, f'训练进度: {int(self.step/self.max_steps*100)}%', 
                         ha='center', transform=self.ax1.transAxes, fontsize=12)
        elif self._is_prediction_complete:
            # 预测已完成，显示总结标题
            self.ax1.set_title('预测完成 - 显示总结信息')
            self.ax2.set_title('预测完成 - 显示总结信息')
        else:
            if self.X_test is not None:
                self.ax1.set_title(f'预测过程 (样本 {self.test_sample_index}/{len(self.X_test)})')
            else:
                self.ax1.set_title('训练完成')
        
        if not self._is_prediction_complete:
            self.ax2.set_title('模型结构')
    
    def highlight_prediction_path(self, test_point, pred_class, pred_proba):
        """
        高亮显示预测路径
        子类可以重写此方法以自定义预测路径的显示
        """
        # 明确标记该方法被调用
        print("\n=====[高亮预测路径方法开始执行]=====")
        
        # 绘制测试点
        print("[高亮] 绘制测试点")
        self.ax1.scatter(test_point[0], test_point[1], c='yellow', marker='*', s=300, 
                         edgecolors='black', linewidths=2, label='测试样本')
        
        # 检查预测是否正确
        is_correct = False
        # 获取当前样本的索引
        current_index = self.test_sample_index - 1
        
        # 在控制台输出预测信息，帮助调试
        print(f"[高亮调试] 样本索引: {current_index}")
        print(f"[高亮调试] 预测类别: {pred_class}")
        print(f"[高亮调试] 预测概率: {pred_proba}")
        
        if self.y_test is not None and current_index >= 0 and current_index < len(self.y_test):
            # 获取当前测试样本的真实标签
            true_class = self.y_test[current_index]
            print(f"[高亮调试] 真实标签: {true_class}")
            print(f"[高亮调试] 类型比较: pred_class={type(pred_class)}, true_class={type(true_class)}")
            
            # 使用字符串比较来避免类型不匹配问题
            is_correct = (str(pred_class) == str(true_class))
            # 更新正确预测计数
            if is_correct:
                self.correct_predictions += 1
                print("[高亮调试] 预测正确!")
            else:
                print("[高亮调试] 预测错误!")
            print(f"[高亮调试] 当前正确预测计数: {self.correct_predictions}")
            
            # 在图上标注预测结果，包括是否正确
            result_text = f'预测: 类别{pred_class}\n'
            result_text += f'真实: 类别{true_class}\n'
            result_text += f'[{"正确" if is_correct else "错误"}]\n'
            result_text += f'概率: {pred_proba[pred_class]:.2f}'
            
            # 根据预测是否正确设置不同的背景色
            bbox_color = 'lightgreen' if is_correct else 'lightcoral'
            
            print(f"[高亮] 在图上添加预测结果文本: {result_text}")
            self.ax1.text(test_point[0] + 0.1, test_point[1] + 0.1, 
                         result_text,
                         fontsize=10, bbox=dict(facecolor=bbox_color, alpha=0.8))
        else:
            print(f"[高亮调试] 索引无效或无测试标签: current_index={current_index}, test_sample_index={self.test_sample_index}")
            # 没有真实标签时的简单显示
            self.ax1.text(test_point[0] + 0.1, test_point[1] + 0.1, 
                         f'预测: 类别{pred_class}\n概率: {pred_proba[pred_class]:.2f}',
                         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # 添加图例
        print("[高亮] 添加图例")
        self.ax1.legend(loc='best')
        print("=====[高亮预测路径方法执行结束]=====\n")
    
    def _show_prediction_summary(self):
        """
        显示预测完成后的总结信息，包括预测成功率
        """
        if self.X_test is not None and self.y_test is not None:
            # 确保使用实际处理的样本数量
            processed_samples = self._total_processed_samples
            accuracy = (self.correct_predictions / processed_samples) * 100 if processed_samples > 0 else 0
            
            # 在控制台输出准确率信息，帮助调试
            print(f"\n[调试信息] 预测准确率计算：")
            print(f"[调试信息] 处理的样本数量: {processed_samples}")
            print(f"[调试信息] 正确预测数量: {self.correct_predictions}")
            print(f"[调试信息] 预测准确率: {accuracy:.1f}%")
            
            # 额外验证：使用模型直接计算准确率
            if hasattr(self, 'model') and self.model is not None:
                try:
                    direct_accuracy = self.model.score(self.X_test[:processed_samples], self.y_test[:processed_samples]) * 100
                    print(f"[调试信息] 模型直接计算的准确率: {direct_accuracy:.1f}%")
                except Exception as e:
                    print(f"[调试信息] 计算直接准确率时出错: {e}")
            
            # 在第一个子图的中央显示预测总结
            summary_text = (
                f"预测完成！\n"
                f"测试样本总数: {processed_samples}\n"
                f"正确预测数: {self.correct_predictions}\n"
                f"预测成功率: {accuracy:.1f}%"
            )
            
            self.ax1.text(0.5, 0.5, summary_text, 
                         transform=self.ax1.transAxes,
                         fontsize=14,
                         ha='center', va='center',
                         bbox=dict(facecolor='gold', alpha=0.9, boxstyle='round,pad=1.5'))
            
            # 在第二个子图中也显示预测总结
            self.ax2.text(0.5, 0.5, summary_text, 
                         transform=self.ax2.transAxes,
                         fontsize=14,
                         ha='center', va='center',
                         bbox=dict(facecolor='gold', alpha=0.9, boxstyle='round,pad=1.5'))
            
            # 更新标题
            self.ax1.set_title('预测完成 - 显示总结信息')
            self.ax2.set_title('预测完成 - 显示总结信息')
    
    def run_animation(self, frames=None, interval=1500, blit=False, auto_run=True, auto_close=True):
        """
        运行动画
        
        参数:
        frames: 动画的总帧数，如果为None则自动计算
        interval: 每帧间隔的毫秒数
        blit: 是否使用blit优化动画性能
        auto_run: 启动后是否自动运行，默认为True
        auto_close: 完成后是否自动关闭窗口，默认为True
        """
        # 添加调试信息
        print(f"[调试] 动画开始运行，interval={interval}ms")
        print(f"[调试] 最大训练步数: {self.max_steps}")
        print(f"[调试] 测试样本数量: {len(self.X_test) if self.X_test is not None else 0}")
        print(f"[调试] 自动运行: {auto_run}, 自动关闭: {auto_close}")
        
        # 保存自动关闭设置
        self._auto_close = auto_close
        
        # 设置自动运行模式
        if auto_run:
            self._manual_step_mode = False
            self._is_paused = False
            if hasattr(self, 'btn_play_pause'):
                self.btn_play_pause.label.set_text('暂停')
        else:
            self._manual_step_mode = True
            self._is_paused = True
            if hasattr(self, 'btn_play_pause'):
                self.btn_play_pause.label.set_text('自动播放')
        
        # 确保动画已初始化
        if self.fig is None:
            self.initialize_visualization()
        
        # 注意：在使用按钮的情况下，blit必须为False以确保按钮正常更新
            
        if frames is None:
            # 根据批量预测重新计算帧数
            if self.X_test is not None:
                # 训练帧数 + 批量预测所需帧数 + 额外显示帧数
                test_batches = (len(self.X_test) + self.prediction_batch_size - 1) // self.prediction_batch_size
                base_frames = self.max_steps + test_batches
                frames = base_frames + 5  # 额外增加5帧以保持预测结果显示
            else:
                frames = self.max_steps + 5
        
        self.animation = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=frames,
            interval=interval,
            repeat=False,
            blit=blit
        )
        
        # 避免tight_layout覆盖按钮区域
        # plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.2)  # 确保底部有足够空间
        
        # 显示图形
        plt.show()
    
    def _setup_axes_limits(self, ax):
        """设置坐标轴范围"""
        if self.X is not None:
            x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
            y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 设置坐标轴标签，优先使用特征名称
            if self.feature_names and len(self.feature_names) >= 2:
                ax.set_xlabel(self.feature_names[0])
                ax.set_ylabel(self.feature_names[1])
            else:
                ax.set_xlabel('特征 1')
                ax.set_ylabel('特征 2')