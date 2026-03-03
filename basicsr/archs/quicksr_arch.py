import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_

# -------------------------- 补全原QuickSRNet依赖的基础模块/函数 --------------------------
class AddOp(nn.Module):
    """逐元素相加操作，原QuickSRNet依赖"""
    def forward(self, x1, x2):
        return x1 + x2

class AnchorOp(nn.Conv2d):
    """AnchorOp卷积，原QuickSRNet的ITO连接依赖"""
    def __init__(self, scaling_factor, kernel_size=3, stride=None, padding=1, freeze_weights=False, **kwargs):
        out_channels = 3 * (scaling_factor ** 2)
        super().__init__(in_channels=3, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride if stride else scaling_factor, padding=padding, **kwargs)
        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

# DCR转换函数（空实现，原代码to_dcr方法依赖，不影响基础超分功能）
def convert_conv_following_space_to_depth_to_dcr(conv, scale):
    return conv

def convert_conv_preceding_depth_to_space_to_dcr(conv, scale):
    return conv

# -------------------------- 核心QuickSRNetBase（保留原生逻辑） --------------------------
class QuickSRNetBase(nn.Module):
    """
    QuickSRNet基础类，保留原生核心逻辑：
    支持整数倍(2/3/4x)和1.5x超分、Hardtanh激活、PixelShuffle上采样、ITO连接
    """
    def __init__(self, scaling_factor, num_channels, num_intermediate_layers, use_ito_connection,
                 in_channels=3, out_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self._use_ito_connection = use_ito_connection
        # 判断是否为整数倍超分
        self._has_integer_scaling_factor = float(scaling_factor).is_integer()

        # 超分倍率赋值（仅支持整数倍/1.5x）
        if self._has_integer_scaling_factor:
            self.scaling_factor = int(scaling_factor)
        elif scaling_factor == 1.5:
            self.scaling_factor = scaling_factor
        else:
            raise NotImplementedError(f'仅支持1.5x非整数超分，当前输入：{scaling_factor}')

        # 构建中间卷积层
        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.extend([
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1),
                nn.Hardtanh(min_val=0., max_val=1.)
            ])

        # 主干CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1),
            nn.Hardtanh(min_val=0., max_val=1.),
            *intermediate_layers,
        )

        # 最后一层卷积（1.5x和整数倍超分的通道/卷积核不同）
        if scaling_factor == 1.5:
            cl_in_channels = num_channels * (2 ** 2)
            cl_out_channels = out_channels * (3 ** 2)
            cl_kernel_size = 1
            cl_padding = 0
            self.space_to_depth = nn.PixelUnshuffle(2)  # 1.5x专用深度转空间
        else:
            cl_in_channels = num_channels
            cl_out_channels = out_channels * (self.scaling_factor ** 2)
            cl_kernel_size = 3
            cl_padding = 1

        self.conv_last = nn.Conv2d(in_channels=cl_in_channels, out_channels=cl_out_channels,
                                   kernel_size=cl_kernel_size, padding=cl_padding)

        # ITO输入到输出残差连接（Small变体关闭，Large变体开启）
        if use_ito_connection:
            self.add_op = AddOp()
            if scaling_factor == 1.5:
                self.anchor = AnchorOp(scaling_factor=3, kernel_size=3, stride=2, padding=1, freeze_weights=False)
            else:
                self.anchor = AnchorOp(scaling_factor=self.scaling_factor, freeze_weights=False)

        # 空间转深度（上采样核心）
        if scaling_factor == 1.5:
            self.depth_to_space = nn.PixelShuffle(3)
        else:
            self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

        self.clip_output = nn.Hardtanh(min_val=0., max_val=1.)  # 输出裁剪到0-1
        self._is_dcr = False  # DCR模式标记

    def forward(self, x):
        """原生前向传播逻辑，和原QuickSRNet完全一致"""
        feat = self.cnn(x)
        # 1.5x超分先做深度转空间
        if not self._has_integer_scaling_factor:
            feat = self.space_to_depth(feat)
        # ITO连接：残差+Anchor卷积的输入特征
        if self._use_ito_connection:
            residual = self.conv_last(feat)
            input_anchor = self.anchor(x)
            feat = self.add_op(input_anchor, residual)
        else:
            feat = self.conv_last(feat)
        # 输出裁剪+上采样
        feat = self.clip_output(feat)
        return self.depth_to_space(feat)

    def to_dcr(self):
        """DCR模式转换，保留原方法，不影响基础功能"""
        if not self._is_dcr:
            if self.scaling_factor == 1.5:
                self.conv_last = convert_conv_following_space_to_depth_to_dcr(self.conv_last, 2)
                self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, 3)
                if self._use_ito_connection:
                    self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, 3)
            else:
                self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, self.scaling_factor)
                if self._use_ito_connection:
                    self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, self.scaling_factor)
            self._is_dcr = True
    
    def initialize(self):
        """原生QuickSRNet的卷积权重初始化策略：卷积核中心加1实现残差初始化"""
        # 初始化主干CNN的卷积层
        for conv_layer in self.cnn:
            if isinstance(conv_layer, nn.Conv2d):
                middle = conv_layer.kernel_size[0] // 2
                num_residual_channels = min(conv_layer.in_channels, conv_layer.out_channels)
                with torch.no_grad():
                    for idx in range(num_residual_channels):
                        conv_layer.weight[idx, idx, middle, middle] += 1.0

        # 初始化最后一层卷积（仅当关闭ITO连接时）
        if not self._use_ito_connection:
            middle = self.conv_last.kernel_size[0] // 2
            out_channels = self.conv_last.out_channels
            scaling_factor_squared = out_channels // self.out_channels
            with torch.no_grad():
                for idx_out in range(out_channels):
                    idx_in = (idx_out % out_channels) // scaling_factor_squared
                    self.conv_last.weight[idx_out, idx_in, middle, middle] += 1.0
# -------------------------- 对齐CATANet的QuickSRNetSmall（ARCH_REGISTRY注册） --------------------------
@ARCH_REGISTRY.register()
class QuickSRNetSmall(nn.Module):
    """
    QuickSRNetSmall超分模型，完全对齐CATANet的BasicSR架构格式
    原生配置：num_channels=32, num_intermediate_layers=2, use_ito_connection=False
    支持超分倍率：2x/3x/4x/1.5x（和原QuickSRNet一致）
    """
    # 核心配置字典，和CATANet的setting对齐，可直接在配置文件中修改
    setting = dict(
        num_channels=32,        # 卷积特征通道数
        num_intermediate_layers=1,  # 中间卷积层数量
        use_ito_connection=False,   # Small变体关闭ITO连接
        in_chans=3,             # 输入通道（RGB=3）
        out_chans=3             # 输出通道（RGB=3）
    )

    def __init__(self, upscale: int = 4):
        super().__init__()
        # 加载配置参数
        self.dim = self.setting['num_channels']  # 和CATANet的dim对齐命名
        self.num_intermediate_layers = self.setting['num_intermediate_layers']
        self.use_ito_connection = self.setting['use_ito_connection']
        self.in_chans = self.setting['in_chans']
        self.out_chans = self.setting['out_chans']
        self.upscale = upscale  # 超分倍率，支持2/3/4/1.5（传入1.5时需直接写浮点数）

        # 校验超分倍率合法性
        if self.upscale not in [1.5, 2, 3, 4]:
            raise ValueError(f'QuickSRNet仅支持1.5/2/3/4x超分，当前输入：{self.upscale}')

        # ---------- 1. 浅层特征提取（和CATANet的first_conv对齐） ----------
        # 原QuickSRNet的浅层已包含在主干CNN中，此处做一层适配性卷积（保证和CATANet结构对齐）
        self.first_conv = nn.Conv2d(self.in_chans, self.in_chans, 3, 1, 1)

        # ---------- 2. 深层特征提取+超分核心（和CATANet的forward_features对齐） ----------
        self.qsrnet_core = QuickSRNetBase(
            scaling_factor=self.upscale,
            num_channels=self.dim,
            num_intermediate_layers=self.num_intermediate_layers,
            use_ito_connection=self.use_ito_connection,
            in_channels=self.in_chans,
            out_channels=self.out_chans
        )

        # ---------- 3. 重建层（和CATANet的上采样/last_conv对齐） ----------
        # 原QuickSRNet的上采样已包含在core中，此处保留base插值残差（和CATANet的base对齐，提升效果）
        self.last_conv = nn.Conv2d(self.out_chans, self.out_chans, 3, 1, 1)  # 输出微调卷积

        # 权重初始化（和CATANet一致，同时保留QuickSRNet原生的卷积初始化策略）
        self.apply(self._init_weights)
        # 重写QuickSRNet核心的初始化（原生策略：卷积核中心加1，实现残差初始化）
        self.qsrnet_core.initialize()

    def _init_weights(self, m):
        """权重初始化，和CATANet完全一致"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """深层特征提取，和CATANet的forward_features方法对齐"""
        return self.qsrnet_core(x)

    def forward(self, x):
        """前向传播，完全对齐CATANet的流程：base插值+主干+残差"""
        # 基础插值图（和CATANet一致，bilinear插值做残差，提升效果）
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        # 浅层特征适配
        x = self.first_conv(x)
        # 深层核心超分
        x_sr = self.forward_features(x)
        # 输出微调+base残差
        out = self.last_conv(x_sr) + base
        # 裁剪到0-1（和原QuickSRNet一致，符合图像像素范围）
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def __repr__(self):
        """模型参数量打印，和CATANet完全一致"""
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}: {:<.4f} [K]'.format(self._get_name(), num_parameters / 10 ** 3)

# -------------------------- 测试代码（和CATANet的main函数对齐） --------------------------
if __name__ == '__main__':
    # 测试3x超分（可替换为1.5/2/4）
    model = QuickSRNetSmall(upscale=3).cuda()
    # 测试输入：B=2, C=3, H=64, W=64（低分辨率图）
    x = torch.randn(2, 3, 64, 64).cuda()
    # 前向传播
    out = model(x)
    # 打印模型信息+输出形状
    print(model)
    print(f'输入形状: {x.shape}, 输出形状: {out.shape}')