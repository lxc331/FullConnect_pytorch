import torch
from typing import Tuple, Optional, Union
import numpy as np


class MinMaxScaler:
    """
    PyTorch 实现的 MinMaxScaler，兼容 sklearn 接口

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        期望的转换后数据范围

    copy : bool, default=True
        是否创建数据的副本

    clip : bool, default=False
        是否将转换后的值裁剪到 feature_range 范围内
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1),
                 copy: bool = True, clip: bool = False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.data_min_ = None  # 训练数据中的最小值
        self.data_max_ = None  # 训练数据中的最大值
        self.min_ = None  # 缩放后的最小值 (用于逆变换)
        self.scale_ = None  # 缩放因子
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    def _check_array(self, X: torch.Tensor) -> torch.Tensor:
        """检查并确保输入是二维张量"""
        if X.dim() == 1:
            X = X.unsqueeze(1)
        elif X.dim() > 2:
            raise ValueError(f"Expected 2D array, got {X.dim()}D array instead")
        return X

    def _reset(self):
        """重置所有统计信息"""
        self.data_min_ = None
        self.data_max_ = None
        self.min_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    def fit(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        计算用于后续缩放的最小值和最大值

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            训练数据

        y : None
            忽略，仅用于兼容性
        """
        X = self._check_array(X)

        if self.copy:
            X = X.clone()

        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]

        # 计算最小值和最大值
        self.data_min_ = torch.min(X, dim=0)[0]
        self.data_max_ = torch.max(X, dim=0)[0]

        # 处理所有值都相同的情况
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0

        # 计算缩放参数
        feature_range = torch.tensor(self.feature_range, dtype=X.dtype, device=X.device)
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - self.data_min_ * self.scale_

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        使用拟合的参数进行缩放

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            要转换的数据
        """
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("This scaler has not been fitted yet. Call 'fit' first.")

        X = self._check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but scaler expects {self.n_features_in_} features.")

        if self.copy:
            X = X.clone()

        # 应用缩放: X_scaled = X * scale + min_
        X_scaled = X * self.scale_ + self.min_

        # 可选裁剪
        if self.clip:
            min_val, max_val = self.feature_range
            X_scaled = torch.clamp(X_scaled, min_val, max_val)

        return X_scaled

    def fit_transform(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        拟合数据然后转换

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            训练数据

        y : None
            忽略，仅用于兼容性
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        将数据转换回原始尺度

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            要逆转换的数据
        """
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("This scaler has not been fitted yet. Call 'fit' first.")

        X = self._check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but scaler expects {self.n_features_in_} features.")

        if self.copy:
            X = X.clone()

        # 逆转换: X_original = (X - min_) / scale_
        X_original = (X - self.min_) / self.scale_

        return X_original

    def partial_fit(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        在线学习：增量更新最小值和最大值

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            用于增量学习的数据
        """
        X = self._check_array(X)

        if self.copy:
            X = X.clone()

        if self.data_min_ is None:
            # 第一次调用
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = X.shape[0]
            self.data_min_ = torch.min(X, dim=0)[0]
            self.data_max_ = torch.max(X, dim=0)[0]
        else:
            # 增量更新
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"X has {X.shape[1]} features, but scaler expects {self.n_features_in_} features.")

            self.n_samples_seen_ += X.shape[0]
            self.data_min_ = torch.min(torch.stack([self.data_min_, torch.min(X, dim=0)[0]]), dim=0)[0]
            self.data_max_ = torch.max(torch.stack([self.data_max_, torch.max(X, dim=0)[0]]), dim=0)[0]

        # 更新缩放参数
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0

        feature_range = torch.tensor(self.feature_range, dtype=X.dtype, device=X.device)
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - self.data_min_ * self.scale_

        return self

    def get_params(self, deep: bool = True) -> dict:
        """获取参数"""
        return {
            'feature_range': self.feature_range,
            'copy': self.copy,
            'clip': self.clip
        }

    def set_params(self, **params):
        """设置参数"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"TorchMinMaxScaler(feature_range={self.feature_range}, "
                f"copy={self.copy}, clip={self.clip})")


