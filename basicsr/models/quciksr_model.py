import torch
import torch.nn as nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

# -------------------------- 保留原QuickSRNet所有网络结构 --------------------------
class AddOp(nn.Module):
    """简单的逐元素相加操作，原代码依赖此模块，补充定义"""
    def forward(self, x1, x2):
        return x1 + x2

class AnchorOp(nn.Conv2d):
    """原代码的AnchorOp，继承Conv2d，补充定义"""
    def __init__(self, scaling_factor, kernel_size=3, stride=None, padding=1, freeze_weights=False, **kwargs):
        out_channels = 3 * (scaling_factor ** 2)
        super().__init__(in_channels=3, out_channels=out_channels, kernel_size=kernel_size, 
                         stride=stride if stride else scaling_factor, padding=padding, **kwargs)
        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

# 原DCR转换函数，原代码to_dcr方法依赖，补充空实现（可根据实际需求完善）
def convert_conv_following_space_to_depth_to_dcr(conv, scale):
    return conv

def convert_conv_preceding_depth_to_space_to_dcr(conv, scale):
    return conv

class QuickSRNetBase(nn.Module):
    """
    Base class for all QuickSRNet variants.

    Note on supported scaling factors: this class supports integer scaling factors. 1.5x upscaling is
    the only non-integer scaling factor supported.
    """

    def __init__(self,
                 scaling_factor,
                 num_channels,
                 num_intermediate_layers,
                 use_ito_connection,
                 in_channels=3,
                 out_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self._use_ito_connection = use_ito_connection
        self._has_integer_scaling_factor = float(scaling_factor).is_integer()

        if self._has_integer_scaling_factor:
            self.scaling_factor = int(scaling_factor)
        elif scaling_factor == 1.5:
            self.scaling_factor = scaling_factor
        else:
            raise NotImplementedError(f'1.5 is the only supported non-integer scaling factor. '
                                      f'Received {scaling_factor}.')

        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.extend([
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
                nn.Hardtanh(min_val=0., max_val=1.)
            ])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.Hardtanh(min_val=0., max_val=1.),
            *intermediate_layers,
        )

        if scaling_factor == 1.5:
            cl_in_channels = num_channels * (2 ** 2)
            cl_out_channels = out_channels * (3 ** 2)
            cl_kernel_size = (1, 1)
            cl_padding = 0
        else:
            cl_in_channels = num_channels
            cl_out_channels = out_channels * (self.scaling_factor ** 2)
            cl_kernel_size = (3, 3)
            cl_padding = 1

        self.conv_last = nn.Conv2d(in_channels=cl_in_channels, out_channels=cl_out_channels, 
                                   kernel_size=cl_kernel_size, padding=cl_padding)

        if use_ito_connection:
            self.add_op = AddOp()
            if scaling_factor == 1.5:
                self.anchor = AnchorOp(scaling_factor=3, kernel_size=3, stride=2, padding=1,
                                      freeze_weights=False)
            else:
                self.anchor = AnchorOp(scaling_factor=self.scaling_factor, freeze_weights=False)

        if scaling_factor == 1.5:
            self.space_to_depth = nn.PixelUnshuffle(2)
            self.depth_to_space = nn.PixelShuffle(3)
        else:
            self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

        self.clip_output = nn.Hardtanh(min_val=0., max_val=1.)
        self.initialize()
        self._is_dcr = False

    def forward(self, input):
        x = self.cnn(input)
        if not self._has_integer_scaling_factor:
            x = self.space_to_depth(x)
        if self._use_ito_connection:
            residual = self.conv_last(x)
            input_convolved = self.anchor(input)
            x = self.add_op(input_convolved, residual)
        else:
            x = self.conv_last(x)
        x = self.clip_output(x)
        return self.depth_to_space(x)
    
    def to_dcr(self):
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

    # def initialize(self):
    #     for conv_layer in self.cnn:
    #         if isinstance(conv_layer, nn.Conv2d):
    #             middle = conv_layer.kernel_size[0] // 2
    #             num_residual_channels = min(conv_layer.in_channels, conv_layer.out_channels)
    #             with torch.no_grad():
    #                 for idx in range(num_residual_channels):
    #                     conv_layer.weight[idx, idx, middle, middle] += 1.

    #     if not self._use_ito_connection:
    #         middle = self.conv_last.kernel_size[0] // 2
    #         out_channels = self.conv_last.out_channels
    #         scaling_factor_squarred = out_channels // self.out_channels
    #         with torch.no_grad():
    #             for idx_out in range(out_channels):
    #                 idx_in = (idx_out % out_channels) // scaling_factor_squarred
    #                 self.conv_last.weight[idx_out, idx_in, middle, middle] += 1.
    def initialize(self):
        # 遍历主干CNN的卷积层（输入层+中间层）
        for idx, conv_layer in enumerate(self.cnn):
            if not isinstance(conv_layer, nn.Conv2d):
                continue
            middle = conv_layer.kernel_size[0] // 2  # 3×3核中心为(1,1)
            in_ch = conv_layer.in_channels
            out_ch = conv_layer.out_channels

            with torch.no_grad():
                if idx == 0:
                    # 1. 输入层：部分身份初始化（论文公式2）- 仅前3个输出通道对应输入RGB
                    for i in range(3):  # 输入为RGB3通道，仅映射前3个输出通道
                        if i < in_ch and i < out_ch:  # 避免通道数不匹配报错
                            conv_layer.weight[i, i, middle, middle] += 1.0
                else:
                    # 2. 中间层：普通身份初始化（论文公式1）- 对角通道加1
                    num_residual_channels = min(in_ch, out_ch)
                    for i in range(num_residual_channels):
                        conv_layer.weight[i, i, middle, middle] += 1.0

        # 3. 输出层：重复交错身份初始化（论文公式3）- 仅关闭ITO连接时执行
        if not self._use_ito_connection:
            conv_last = self.conv_last
            middle = conv_last.kernel_size[0] // 2
            out_channels = conv_last.out_channels
            # 计算S²：输出通道数 = 3 × S² → S² = out_channels // 3（兼容整数倍/1.5×）
            scaling_factor_squared = out_channels // self.out_channels  # 1.5×时S=3，S²=9；2×时S²=4
            in_ch_last = conv_last.in_channels

            with torch.no_grad():
                for idx_out in range(out_channels):
                    # 论文核心逻辑：输出通道idx_out映射到输入3通道（idx_in = idx_out // S²）
                    idx_in = idx_out // scaling_factor_squared
                    # 确保idx_in在输入通道范围内（避免1.5×超分通道数不匹配）
                    if idx_in < in_ch_last:
                        conv_last.weight[idx_out, idx_in, middle, middle] += 1.0

@MODEL_REGISTRY.register()
class QuickSRNetSmall(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=1,
            use_ito_connection=False,
            **kwargs
        )

class QuickSRNetMedium(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=32,
            num_intermediate_layers=5,
            use_ito_connection=False,
            **kwargs
        )

class QuickSRNetLarge(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=64,
            num_intermediate_layers=11,
            use_ito_connection=True,
            **kwargs
        )

# -------------------------- BasicSR框架的Model封装 --------------------------
@MODEL_REGISTRY.register()
class QuickSRNetModel(BaseModel):
    """QuickSRNet模型，适配BasicSR框架，支持Small/Medium/Large三个变体"""

    def __init__(self, opt):
        super().__init__(opt)
        # 构建QuickSRNet网络（从配置文件读取变体和超参）
        self.net_g = build_network(opt['network_g'])
        # 将网络放到指定设备（GPU/CPU）
        self.net_g = self.model_to_device(self.net_g)
        # 打印网络结构和参数量
        self.print_network(self.net_g)

        # 加载预训练权重
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # 若开启DCR模式，初始化DCR转换
        if self.opt.get('use_dcr', False):
            logger = get_root_logger()
            logger.info('Enable DCR mode for QuickSRNet')
            self.net_g.to_dcr()

        # 训练模式下初始化训练配置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # 初始化EMA（指数移动平均），用于测试和保存最优模型
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # 加载预训练权重到EMA网络
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # 直接复制主网络权重
            self.net_g_ema.eval()
            # EMA网络开启DCR（若配置开启）
            if self.opt.get('use_dcr', False):
                self.net_g_ema.to_dcr()

        # 构建损失函数：像素损失（必选）+ 感知损失（可选）
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None, at least one loss is required!')

        # 配置优化器和学习率调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        # 收集需要训练的参数
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # 构建优化器（从配置文件读取优化器类型：Adam/AdamW/SGD等）
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """加载数据到设备，BasicSR数据加载器标准接口"""
        self.lq = data['lq'].to(self.device)  # 低分辨率图
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)  # 高分辨率真值图

    def optimize_parameters(self, current_iter):
        """单步训练，更新网络权重"""
        self.optimizer_g.zero_grad()  # 梯度清零
        self.output = self.net_g(self.lq)  # 前向传播得到超分结果

        l_total = 0.0
        loss_dict = OrderedDict()
        # 像素损失（如L1/L2/MSE）
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # 感知损失（如VGG感知损失+风格损失）
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()  # 反向传播计算梯度
        self.optimizer_g.step()  # 优化器更新权重

        # 归约损失（分布式训练时）
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA权重更新
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """测试/验证模式，前向传播不计算梯度"""
        if hasattr(self, 'net_g_ema'):
            # 使用EMA网络测试（精度更高）
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            # 使用主网络测试
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()  # 恢复训练模式

    def test_selfensemble(self):
        """自集成测试（8倍数据增强，提升测试精度），适配QuickSRNet"""
        def _transform(v, op):
            """数据增强变换：垂直翻转/水平翻转/转置"""
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        # 生成8种增强的低分辨率图
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # 批量推理
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # 逆变换还原，合并结果
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)
        # 取平均得到最终超分结果
        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """分布式验证，仅主进程执行"""
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """非分布式验证，计算指标并保存结果图"""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        # 初始化指标结果
        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            # 获取可视化结果
            visuals = self.get_current_visuals()
            metric_data['img'] = visuals['result']
            if 'gt' in visuals:
                metric_data['img2'] = visuals['gt']
                del self.gt

            # 释放GPU内存
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # 保存超分结果图
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    suffix = self.opt['val'].get('suffix', '')
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}_{suffix if suffix else self.opt["name"]}.png')
                imwrite(tensor2img([visuals['result']]), save_img_path)

            # 计算验证指标（如PSNR/SSIM）
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        # 计算指标平均值并记录日志
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        """记录验证指标到日志和TensorBoard"""
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value.item():.4f}'
            if hasattr(self, 'best_metric_results'):
                best_val = self.best_metric_results[dataset_name][metric]['val'].item()
                best_iter = self.best_metric_results[dataset_name][metric]['iter']
                log_str += f'\tBest: {best_val:.4f} @ {best_iter} iter'
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        """获取当前的可视化张量（LQ/Result/GT）"""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        """保存模型权重和训练状态"""
        if hasattr(self, 'net_g_ema'):
            # 同时保存主网络和EMA网络权重
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, 
                              param_key=['params', 'params_ema'])
        else:
            # 仅保存主网络权重
            self.save_network(self.net_g, 'net_g', current_iter)
        # 保存训练状态（优化器、调度器等）
        self.save_training_state(epoch, current_iter)