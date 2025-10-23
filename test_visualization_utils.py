import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from visualization_utils import BaseModelVisualizer

# 为了测试创建一个简单的继承类
class TestVisualizer(BaseModelVisualizer):
    def __init__(self):
        super().__init__()
        # 模拟简单数据
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([0, 1, 0])
        self.X_train = self.X
        self.y_train = self.y
        self.X_test = self.X[:1]
        self.y_test = self.y[:1]
        self.step = 0
        self.max_steps = 5
        self._is_training_complete = False
        self.test_sample_index = 0
    
    def fit_step_by_step(self):
        if self.step < self.max_steps:
            self.step += 1
            if self.step >= self.max_steps:
                self._is_training_complete = True
    
    def predict_visualization(self):
        if self.test_sample_index < len(self.X_test):
            test_point = self.X_test[self.test_sample_index]
            self.test_sample_index += 1
            return test_point, 0, [0.8, 0.2]
        return None
    
    def plot_data(self, ax):
        pass
    
    def plot_decision_boundary(self, ax):
        pass
    
    def plot_model_structure(self, ax):
        pass

class TestVisualizationUtils(unittest.TestCase):
    def setUp(self):
        # 避免实际显示图形
        plt.ioff()
        self.visualizer = TestVisualizer()
        # 初始化但不显示图形
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.tight_layout'):
            self.visualizer.initialize_visualization()
    
    def tearDown(self):
        plt.close('all')
    
    def test_manual_step_mode_initial_state(self):
        """测试手动步进模式的初始状态"""
        self.assertTrue(self.visualizer._manual_step_mode)
        # 在初始化时，_is_paused 可能为 False，我们测试的是手动模式
        self.assertFalse(self.visualizer._step_forward_required)
    
    def test_next_step_button_click(self):
        """测试下一步按钮点击功能"""
        # 模拟按钮点击
        self.visualizer._on_next_step_clicked(None)
        
        # 验证状态变化
        self.assertTrue(self.visualizer._manual_step_mode)
        self.assertTrue(self.visualizer._is_paused)
        self.assertTrue(self.visualizer._step_forward_required)
        
        # 模拟更新plot
        initial_step = self.visualizer.step
        with patch.object(self.visualizer, '_get_animated_artists') as mock_get_artists:
            mock_get_artists.return_value = []
            self.visualizer.update_plot(0)
        
        # 验证step增加
        self.assertEqual(self.visualizer.step, initial_step + 1)
        # 验证步进标志重置
        self.assertFalse(self.visualizer._step_forward_required)
    
    def test_multiple_next_step_clicks(self):
        """测试多次点击下一步按钮"""
        # 初始步骤
        initial_step = self.visualizer.step
        
        # 第一次点击
        self.visualizer._on_next_step_clicked(None)
        with patch.object(self.visualizer, '_get_animated_artists') as mock_get_artists:
            mock_get_artists.return_value = []
            self.visualizer.update_plot(0)
        self.assertEqual(self.visualizer.step, initial_step + 1)
        
        # 第二次点击
        self.visualizer._on_next_step_clicked(None)
        with patch.object(self.visualizer, '_get_animated_artists') as mock_get_artists:
            mock_get_artists.return_value = []
            self.visualizer.update_plot(0)
        self.assertEqual(self.visualizer.step, initial_step + 2)
    
    def test_play_pause_button_toggle(self):
        """测试播放/暂停按钮切换功能"""
        # 初始状态应该是手动模式
        self.assertTrue(self.visualizer._manual_step_mode)
        
        # 第一次点击切换到自动播放
        self.visualizer._on_play_pause_clicked(None)
        self.assertFalse(self.visualizer._manual_step_mode)
        self.assertFalse(self.visualizer._is_paused)
        self.assertFalse(self.visualizer._step_forward_required)
        
        # 第二次点击切换回手动模式
        self.visualizer._on_play_pause_clicked(None)
        self.assertTrue(self.visualizer._manual_step_mode)
        self.assertTrue(self.visualizer._is_paused)
    
    def test_keyboard_space_toggle(self):
        """测试空格键切换功能"""
        # 模拟空格键事件
        event = MagicMock()
        event.key = ' '
        
        # 初始状态
        self.assertTrue(self.visualizer._manual_step_mode)
        
        # 第一次按空格
        self.visualizer._on_key_press(event)
        self.assertFalse(self.visualizer._manual_step_mode)
        self.assertFalse(self.visualizer._is_paused)
        
        # 第二次按空格
        self.visualizer._on_key_press(event)
        self.assertTrue(self.visualizer._manual_step_mode)
        self.assertTrue(self.visualizer._is_paused)
    
    def test_keyboard_n_step_forward(self):
        """测试n键前进一步功能"""
        # 模拟n键事件
        event = MagicMock()
        event.key = 'n'
        
        initial_step = self.visualizer.step
        
        # 按n键
        self.visualizer._on_key_press(event)
        self.assertTrue(self.visualizer._manual_step_mode)
        self.assertTrue(self.visualizer._is_paused)
        self.assertTrue(self.visualizer._step_forward_required)
        
        # 模拟更新plot
        with patch.object(self.visualizer, '_get_animated_artists') as mock_get_artists:
            mock_get_artists.return_value = []
            self.visualizer.update_plot(0)
        
        # 验证step增加
        self.assertEqual(self.visualizer.step, initial_step + 1)
    
    def test_update_plot_in_manual_mode_without_step(self):
        """测试手动模式下没有步进标志时的更新行为"""
        self.visualizer._step_forward_required = False
        
        initial_step = self.visualizer.step
        
        # 调用update_plot，但不应增加step
        with patch.object(self.visualizer, '_get_animated_artists') as mock_get_artists:
            mock_get_artists.return_value = []
            self.visualizer.update_plot(0)
        
        # step应该保持不变
        self.assertEqual(self.visualizer.step, initial_step)
    
    def test_update_plot_in_auto_mode(self):
        """测试自动模式下的更新行为"""
        # 切换到自动模式
        self.visualizer._manual_step_mode = False
        self.visualizer._is_paused = False
        
        initial_step = self.visualizer.step
        
        # 调用update_plot，应该增加step
        with patch.object(self.visualizer, '_get_animated_artists') as mock_get_artists:
            mock_get_artists.return_value = []
            self.visualizer.update_plot(0)
        
        # step应该增加
        self.assertEqual(self.visualizer.step, initial_step + 1)

if __name__ == '__main__':
    unittest.main()
