# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import math
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # ---------------------------
# # 基础组件定义
# # ---------------------------
# class AddOp(nn.Module):
#     """简单的加法操作层，适配张量广播"""
#     def forward(self, x1, x2):
#         return x1 + x2

# class AnchorOp(nn.Conv2d):
#     """Anchor操作层，本质是卷积层"""
#     def __init__(self, scaling_factor, kernel_size=3, stride=1, padding=1, freeze_weights=False, in_channels=3, out_channels=None):
#         if out_channels is None:
#             out_channels = 3 * (scaling_factor ** 2)
#         super().__init__(in_channels=in_channels, out_channels=out_channels, 
#                         kernel_size=kernel_size, stride=stride, padding=padding)
#         if freeze_weights:
#             for param in self.parameters():
#                 param.requires_grad = False

# def convert_conv_following_space_to_depth_to_dcr(conv, scale):
#     """适配space_to_depth后的卷积转换（保持原接口）"""
#     return conv

# def convert_conv_preceding_depth_to_space_to_dcr(conv, scale):
#     """适配depth_to_space前的卷积转换（保持原接口）"""
#     return conv

# # ---------------------------
# # 教师/学生模型定义
# # ---------------------------
# class QuickSRNetBase(nn.Module):
#     """基础超分辨率网络"""
#     def __init__(self,
#                  scaling_factor,
#                  num_channels,
#                  num_intermediate_layers,
#                  use_ito_connection,
#                  in_channels=3,
#                  out_channels=3):
#         super().__init__()
#         self.out_channels = out_channels
#         self._use_ito_connection = use_ito_connection
#         self._has_integer_scaling_factor = float(scaling_factor).is_integer()

#         if self._has_integer_scaling_factor:
#             self.scaling_factor = int(scaling_factor)
#         elif scaling_factor == 1.5:
#             self.scaling_factor = scaling_factor
#         else:
#             raise NotImplementedError(f'1.5 is the only supported non-integer scaling factor. '
#                                       f'Received {scaling_factor}.')

#         # 构建中间层
#         intermediate_layers = []
#         for _ in range(num_intermediate_layers):
#             intermediate_layers.extend([
#                 nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
#                 nn.Hardtanh(min_val=0., max_val=1.)
#             ])

#         # 构建完整CNN序列
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
#             nn.Hardtanh(min_val=0., max_val=1.),
#             *intermediate_layers,
#         )

#         # 最后一层卷积配置（适配1.5x缩放）
#         if scaling_factor == 1.5:
#             cl_in_channels = num_channels * (2 ** 2)  # 1.5x需要先做space_to_depth(2)
#             cl_out_channels = out_channels * (3 ** 2)  # 然后depth_to_space(3)
#             cl_kernel_size = (1, 1)
#             cl_padding = 0
#         else:
#             cl_in_channels = num_channels
#             cl_out_channels = out_channels * (self.scaling_factor ** 2)
#             cl_kernel_size = (3, 3)
#             cl_padding = 1

#         self.conv_last = nn.Conv2d(in_channels=cl_in_channels, out_channels=cl_out_channels, 
#                                    kernel_size=cl_kernel_size, padding=cl_padding)

#         # ITO连接（当前未使用）
#         if use_ito_connection:
#             self.add_op = AddOp()
#             if scaling_factor == 1.5:
#                 self.anchor = AnchorOp(scaling_factor=3, kernel_size=3, stride=2, padding=1,
#                                               freeze_weights=False)
#             else:
#                 self.anchor = AnchorOp(scaling_factor=self.scaling_factor,
#                                               freeze_weights=False)

#         # 空间变换层
#         if scaling_factor == 1.5:
#             self.space_to_depth = nn.PixelUnshuffle(2)
#             self.depth_to_space = nn.PixelShuffle(3)
#         else:
#             self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

#         self.clip_output = nn.Hardtanh(min_val=0., max_val=1.)

#         # 初始化权重
#         self.initialize()
#         self._is_dcr = False

#     def forward(self, input):
#         x = self.cnn(input)
#         if not self._has_integer_scaling_factor:
#             x = self.space_to_depth(x)

#         if self._use_ito_connection:
#             residual = self.conv_last(x)
#             input_convolved = self.anchor(input)
#             x = self.add_op(input_convolved, residual)
#         else:
#             x = self.conv_last(x)

#         x = self.clip_output(x)
#         return self.depth_to_space(x)
    
#     def to_dcr(self):
#         if not self._is_dcr:
#             if self.scaling_factor == 1.5:
#                 self.conv_last = convert_conv_following_space_to_depth_to_dcr(self.conv_last, 2)
#                 self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, 3)
#                 if self._use_ito_connection:
#                     self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, 3)
#             else:
#                 self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, self.scaling_factor)
#                 if self._use_ito_connection:
#                     self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, self.scaling_factor)
#             self._is_dcr = True

#     def initialize(self):
#         """初始化卷积层权重，添加残差连接"""
#         for conv_layer in self.cnn:
#             if isinstance(conv_layer, nn.Conv2d):
#                 middle = conv_layer.kernel_size[0] // 2
#                 num_residual_channels = min(conv_layer.in_channels, conv_layer.out_channels)
#                 with torch.no_grad():
#                     for idx in range(num_residual_channels):
#                         conv_layer.weight[idx, idx, middle, middle] += 1.

#         if not self._use_ito_connection:
#             middle = self.conv_last.kernel_size[0] // 2
#             out_channels = self.conv_last.out_channels
#             scaling_factor_squarred = out_channels // self.out_channels
#             with torch.no_grad():
#                 for idx_out in range(out_channels):
#                     idx_in = (idx_out % out_channels) // scaling_factor_squarred
#                     self.conv_last.weight[idx_out, idx_in, middle, middle] += 1.

# # 教师模型（32通道，2中间层，1.5x超分）
# class QuickSRNetSmall(QuickSRNetBase):
#     def __init__(self, scaling_factor, num_channels=32, num_intermediate_layers=2, **kwargs):
#         super().__init__(
#             scaling_factor,
#             num_channels=num_channels,
#             num_intermediate_layers=num_intermediate_layers,
#             use_ito_connection=False,
#             **kwargs
#         )

# # 学生模型（16通道，1中间层，轻量化）
# class QuickSRNetTiny(QuickSRNetBase):
#     def __init__(self, scaling_factor, **kwargs):
#         super().__init__(
#             scaling_factor,
#             num_channels=16,
#             num_intermediate_layers=1,
#             use_ito_connection=False,
#             **kwargs
#         )

# # ---------------------------
# # 工具函数：加载教师模型权重
# # ---------------------------
# def load_teacher_checkpoint(model, checkpoint_path, device):
#     """加载教师模型的checkpoint，自动适配参数匹配"""
#     try:
#         # 加载checkpoint
#         checkpoint = torch.load(
#             checkpoint_path, 
#             map_location=device,
#             weights_only=False
#         )
        
#         # 提取模型权重（兼容多种格式）
#         state_dict = None
#         for key in ['state_dict', 'model_state_dict', 'model']:
#             if key in checkpoint:
#                 state_dict = checkpoint[key]
#                 break
#         if state_dict is None:
#             state_dict = checkpoint
        
#         # 移除module.前缀（多GPU训练的模型）
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 new_state_dict[k[7:]] = v
#             else:
#                 new_state_dict[k] = v
        
#         # 自动过滤不匹配的权重键
#         model_state_dict = model.state_dict()
#         filtered_state_dict = {}
        
#         for k, v in new_state_dict.items():
#             if k in model_state_dict and model_state_dict[k].shape == v.shape:
#                 filtered_state_dict[k] = v
#             else:
#                 if k in model_state_dict:
#                     print(f"⚠️ 权重尺寸不匹配 {k}: checkpoint={v.shape}, model={model_state_dict[k].shape}")
#                 else:
#                     print(f"⚠️ 模型中无此权重键 {k}，已跳过")
        
#         # 加载过滤后的权重
#         model.load_state_dict(filtered_state_dict, strict=False)
        
#         # 验证加载结果
#         loaded_keys = len(filtered_state_dict)
#         total_keys = len(model_state_dict)
#         print(f"✅ 成功加载教师模型权重：{checkpoint_path}")
#         print(f"📌 加载统计：共{loaded_keys}/{total_keys}个权重参数匹配并加载")
#         print(f"📌 模型参数：通道数={model.cnn[0].out_channels}, 中间层数={model._init_kwargs['num_intermediate_layers']}")
        
#         # 验证缩放因子
#         if hasattr(model, 'scaling_factor') and model.scaling_factor != 1.5:
#             warnings.warn(f"⚠️ 模型缩放因子为{model.scaling_factor}，但checkpoint对应1.5x超分！")
        
#         return model
    
#     except Exception as e:
#         raise RuntimeError(
#             f"❌ 加载教师模型权重失败：{str(e)}\n"
#             f"💡 检查点：\n"
#             f"  1. 确认checkpoint路径正确：{checkpoint_path}\n"
#             f"  2. 确认模型参数匹配（当前通道数={model.cnn[0].out_channels}, 中间层数={model._init_kwargs.get('num_intermediate_layers', '未知')}）\n"
#             f"  3. 确认PyTorch版本兼容（建议2.0+）"
#         )

# # ---------------------------
# # 蒸馏损失定义
# # ---------------------------
# class SRDistillationLoss(nn.Module):
#     """超分辨率蒸馏损失：硬损失 + 软损失"""
#     def __init__(self, alpha=0.3):
#         super().__init__()
#         self.alpha = alpha
#         self.mse_loss = nn.MSELoss()

#     def forward(self, student_output, teacher_output, hr_target):
#         # 硬损失：学生输出与真实HR图像的MSE
#         hard_loss = self.mse_loss(student_output, hr_target)
#         # 软损失：学生输出与教师输出的MSE
#         soft_loss = self.mse_loss(student_output, teacher_output)
#         # 总损失
#         total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
#         return total_loss, hard_loss, soft_loss

# # ---------------------------
# # 数据集定义（核心：直接裁剪1023x1023）
# # ---------------------------
# class SRDataset(Dataset):
#     """
#     超分辨率数据集：
#     1. 直接从原始HR图像 左上/右上/左下/右下 裁剪出 1023x1023
#     2. HR 固定为 1023x1023
#     3. LR = 682x682 (1023/1.5)，偶数，适配网络
#     """
#     def __init__(self, 
#                  hr_dir, 
#                  scaling_factor=1.5, 
#                  transform=None):
#         self.hr_dir = hr_dir
#         self.scaling_factor = scaling_factor
#         self.transform = transform
#         self.hr_filenames = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

#         # 固定裁剪尺寸：1023x1023
#         self.crop_h, self.crop_w = 1023, 1023
#         self.fixed_hr_size = (1023, 1023)
#         # LR尺寸：1023 * 2/3 = 682（偶数，适配space_to_depth(2)）
#         self.fixed_lr_h = 682
#         self.fixed_lr_w = 682

#         self.lr_interpolation = Image.BICUBIC
#         self.hr_interpolation = Image.BICUBIC

#     def _crop_1023_from_four_corners(self, img):
#         """从图像四个角落裁剪1023x1023，尺寸不足先缩放"""
#         W, H = img.size
#         target_w, target_h = self.crop_w, self.crop_h

#         # 图像尺寸不足时，先缩放到≥1023x1023
#         if W < target_w or H < target_h:
#             scale = max(target_w / W, target_h / H)
#             new_W = int(W * scale)
#             new_H = int(H * scale)
#             img = img.resize((new_W, new_H), self.hr_interpolation)
#             W, H = new_W, new_H

#         # 四个角落的裁剪坐标
#         corners = [
#             (0,          0,          target_w,                target_h               ),  # 左上
#             (W-target_w, 0,          W,                       target_h               ),  # 右上
#             (0,          H-target_h, target_w,                H                      ),  # 左下
#             (W-target_w, H-target_h, W,                       H                      ),  # 右下
#         ]

#         # 随机选择一个角落裁剪
#         import random
#         chosen = random.choice(corners)
#         return img.crop(chosen)

#     def __len__(self):
#         return len(self.hr_filenames)

#     def __getitem__(self, idx):
#         # 加载HR图像
#         hr_path = os.path.join(self.hr_dir, self.hr_filenames[idx])
#         try:
#             hr_img = Image.open(hr_path).convert('RGB')
#         except Exception as e:
#             raise RuntimeError(f"加载图像失败 {hr_path}：{str(e)}")

#         # 核心步骤：直接裁剪1023x1023
#         hr = self._crop_1023_from_four_corners(hr_img)
#         # 生成LR图像（682x682）
#         lr = hr.resize((self.fixed_lr_w, self.fixed_lr_h), self.lr_interpolation)

#         # 应用变换
#         if self.transform is not None:
#             lr = self.transform(lr)
#             hr = self.transform(hr)

#         return lr, hr

# # ---------------------------
# # 评估指标计算
# # ---------------------------
# def calculate_psnr(img1, img2, max_val=1.0):
#     """计算PSNR（峰值信噪比）"""
#     img1 = img1.clamp(0.0, 1.0)
#     img2 = img2.clamp(0.0, 1.0)
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(max_val / math.sqrt(mse.item()))

# # ---------------------------
# # 训练函数
# # ---------------------------
# def train_distillation(
#     teacher_model,
#     student_model,
#     train_loader,
#     val_loader,
#     criterion,
#     optimizer,
#     device,
#     epochs=50,
#     save_path='sr_distillation_best.pth',
#     log_interval=10,
#     lr_scheduler=None
# ):
#     """超分辨率蒸馏训练函数"""
#     # 固定教师模型参数
#     teacher_model.eval()
#     for param in teacher_model.parameters():
#         param.requires_grad = False
    
#     student_model.train()
#     best_val_psnr = 0.0
    
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         epoch_hard_loss = 0.0
#         epoch_soft_loss = 0.0
        
#         for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
#             lr_imgs = lr_imgs.to(device, non_blocking=True)
#             hr_imgs = hr_imgs.to(device, non_blocking=True)
            
#             # 前向传播
#             optimizer.zero_grad()
            
#             # 教师模型推理（无梯度）
#             with torch.no_grad():
#                 teacher_output = teacher_model(lr_imgs)
            
#             # 学生模型推理
#             student_output = student_model(lr_imgs)
            
#             # 计算蒸馏损失
#             total_loss, hard_loss, soft_loss = criterion(student_output, teacher_output, hr_imgs)
            
#             # 反向传播
#             total_loss.backward()
#             optimizer.step()
            
#             # 累计损失
#             epoch_loss += total_loss.item()
#             epoch_hard_loss += hard_loss.item()
#             epoch_soft_loss += soft_loss.item()
            
#             # 打印日志
#             if (batch_idx + 1) % log_interval == 0:
#                 print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
#                       f'Total Loss: {total_loss.item():.6f}, '
#                       f'Hard Loss: {hard_loss.item():.6f}, Soft Loss: {soft_loss.item():.6f}')
        
#         # 学习率调度
#         if lr_scheduler is not None:
#             lr_scheduler.step()
#             current_lr = lr_scheduler.get_last_lr()[0]
#             print(f'Current learning rate: {current_lr:.6e}')
        
#         # 计算epoch平均损失
#         avg_loss = epoch_loss / len(train_loader)
#         avg_hard_loss = epoch_hard_loss / len(train_loader)
#         avg_soft_loss = epoch_soft_loss / len(train_loader)
#         print(f'\nEpoch [{epoch+1}/{epochs}] Summary:')
#         print(f'Average Total Loss: {avg_loss:.6f}, Average Hard Loss: {avg_hard_loss:.6f}, Average Soft Loss: {avg_soft_loss:.6f}')
        
#         # 验证阶段
#         val_psnr = validate(student_model, val_loader, device)
#         print(f'Validation PSNR: {val_psnr:.2f} dB\n')
        
#         # 保存最优模型
#         if val_psnr > best_val_psnr:
#             best_val_psnr = val_psnr
#             torch.save({
#                 'epoch': epoch + 1,
#                 'student_model_state_dict': student_model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
#                 'best_psnr': best_val_psnr,
#                 'loss': avg_loss
#             }, save_path)
#             print(f'Saved best model with PSNR: {best_val_psnr:.2f} dB\n')

# def validate(model, val_loader, device):
#     """验证函数，计算PSNR"""
#     model.eval()
#     total_psnr = 0.0
#     with torch.no_grad():
#         for lr_imgs, hr_imgs in val_loader:
#             lr_imgs = lr_imgs.to(device, non_blocking=True)
#             hr_imgs = hr_imgs.to(device, non_blocking=True)
            
#             # 模型推理
#             sr_imgs = model(lr_imgs)
            
#             # 计算PSNR
#             psnr = calculate_psnr(sr_imgs, hr_imgs)
#             total_psnr += psnr
    
#     avg_psnr = total_psnr / len(val_loader)
#     model.train()
#     return avg_psnr

# # ---------------------------
# # 主函数（训练入口）
# # ---------------------------
# def main():
#     # 1. 核心配置
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     scaling_factor = 1.5  # 匹配教师模型的1.5x超分
#     batch_size = 16
#     epochs = 50
#     base_lr = 1e-4
#     alpha = 0.3  # 蒸馏软损失权重
#     weight_decay = 1e-5
    
#     # 2. 路径配置（请根据你的实际路径修改）
#     teacher_ckpt_path = '/home/huqiongyang/code/CATANet/experiments/train_quicksr_x1.5_scratch_channel16_layer_1/models/checkpoint_float32.pth.tar'
#     train_hr_dir = '/home/huqiongyang/code/CATANet/datasets/Div2K/DIV2K_train_HR/DIV2K_train_HR/DIV2K_train_HR'
#     val_hr_dir = '/home/huqiongyang/code/CATANet/datasets/bsd100/HR'
#     save_path = 'quick_sr_1.5x_distillation_best.pth'
    
#     # 3. 数据变换（仅保留ToTensor，裁剪逻辑已在Dataset内实现）
#     transform = transforms.Compose([
#         transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
#     ])
    
#     # 4. 构建数据集和数据加载器
#     print(f"📥 加载数据集，缩放因子：{scaling_factor}x")
#     # 训练集
#     train_dataset = SRDataset(
#         train_hr_dir, 
#         scaling_factor=scaling_factor, 
#         transform=transform
#     )
#     # 验证集
#     val_dataset = SRDataset(
#         val_hr_dir, 
#         scaling_factor=scaling_factor, 
#         transform=transform
#     )
    
#     # 数据加载器（无需自定义collate_fn，尺寸已完全统一）
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=4,
#         pin_memory=True,
#         drop_last=True
#     )
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # 打印数据集信息
#     print(f"📊 训练集大小：{len(train_dataset)}, 验证集大小：{len(val_dataset)}")
#     print(f"📏 固定HR尺寸：{train_dataset.fixed_hr_size}, 固定LR尺寸：({train_dataset.fixed_lr_w}, {train_dataset.fixed_lr_h})")
    
#     # 5. 初始化教师模型
#     print(f"🔧 初始化教师模型（32通道，2中间层），加载权重：{teacher_ckpt_path}")
#     teacher_model = QuickSRNetSmall(
#         scaling_factor=scaling_factor,
#         num_channels=32,          
#         num_intermediate_layers=2
#     ).to(device)
#     # 保存初始化参数（用于日志）
#     teacher_model._init_kwargs = {
#         'num_channels': 32,
#         'num_intermediate_layers': 2
#     }
    
#     # 加载预训练权重
#     teacher_model = load_teacher_checkpoint(teacher_model, teacher_ckpt_path, device)
    
#     # 6. 初始化学生模型
#     print("🔧 初始化学生模型（QuickSRNetTiny，16通道，1中间层）")
#     student_model = QuickSRNetTiny(scaling_factor=scaling_factor).to(device)
    
#     # 7. 定义损失函数、优化器、学习率调度器
#     criterion = SRDistillationLoss(alpha=alpha)
#     optimizer = optim.AdamW(
#         student_model.parameters(),
#         lr=base_lr,
#         betas=(0.9, 0.999),
#         weight_decay=weight_decay
#     )
#     # 学习率调度器
#     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=epochs,
#         eta_min=1e-6
#     )
    
#     # 8. 开始蒸馏训练
#     print(f'\n🚀 开始蒸馏训练 on {device}...')
#     print(f'👨‍🏫 教师模型：QuickSRNetSmall (32通道, 2中间层, 1.5x超分)')
#     print(f'👨‍🎓 学生模型：QuickSRNetTiny (16通道, 1中间层, 1.5x超分)')
#     print(f'📊 批次大小：{batch_size}, 训练轮数：{epochs}, 初始学习率：{base_lr}')
    
#     train_distillation(
#         teacher_model=teacher_model,
#         student_model=student_model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         device=device,
#         epochs=epochs,
#         save_path=save_path,
#         log_interval=10,
#         lr_scheduler=lr_scheduler
#     )

# if __name__ == '__main__':
#     main()

#---------------------------            -----   
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import math
# import os
# import random
# import warnings
# warnings.filterwarnings('ignore')

# # ---------------------------
# # 基础组件定义（保留原结构）
# # ---------------------------
# class AddOp(nn.Module):
#     """简单的加法操作层，适配张量广播"""
#     def forward(self, x1, x2):
#         return x1 + x2

# class AnchorOp(nn.Conv2d):
#     """Anchor操作层，本质是卷积层"""
#     def __init__(self, scaling_factor, kernel_size=3, stride=1, padding=1, freeze_weights=False, in_channels=3, out_channels=None):
#         if out_channels is None:
#             out_channels = 3 * (scaling_factor ** 2)
#         super().__init__(in_channels=in_channels, out_channels=out_channels, 
#                         kernel_size=kernel_size, stride=stride, padding=padding)
#         if freeze_weights:
#             for param in self.parameters():
#                 param.requires_grad = False

# def convert_conv_following_space_to_depth_to_dcr(conv, scale):
#     """适配space_to_depth后的卷积转换（保持原接口）"""
#     return conv

# def convert_conv_preceding_depth_to_space_to_dcr(conv, scale):
#     """适配depth_to_space前的卷积转换（保持原接口）"""
#     return conv

# # ---------------------------
# # 教师/学生模型定义（优化初始化+残差）
# # ---------------------------
# class QuickSRNetBase(nn.Module):
#     """基础超分辨率网络（优化初始化，加速收敛）"""
#     def __init__(self,
#                  scaling_factor,
#                  num_channels,
#                  num_intermediate_layers,
#                  use_ito_connection,
#                  in_channels=3,
#                  out_channels=3):
#         super().__init__()
#         self.out_channels = out_channels
#         self._use_ito_connection = use_ito_connection
#         self._has_integer_scaling_factor = float(scaling_factor).is_integer()

#         if self._has_integer_scaling_factor:
#             self.scaling_factor = int(scaling_factor)
#         elif scaling_factor == 1.5:
#             self.scaling_factor = scaling_factor
#         else:
#             raise NotImplementedError(f'1.5 is the only supported non-integer scaling factor. '
#                                       f'Received {scaling_factor}.')

#         # 构建中间层（添加残差连接）
#         intermediate_layers = []
#         self.residual_weights = []  # 保存残差权重，用于初始化
#         for i in range(num_intermediate_layers):
#             conv = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1)
#             intermediate_layers.extend([
#                 conv,
#                 nn.Hardtanh(min_val=0., max_val=1.)
#             ])
#             self.residual_weights.append(conv)  # 记录卷积层，用于残差初始化

#         # 构建完整CNN序列
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
#             nn.Hardtanh(min_val=0., max_val=1.),
#             *intermediate_layers,
#         )

#         # 最后一层卷积配置（适配1.5x缩放）
#         if scaling_factor == 1.5:
#             cl_in_channels = num_channels * (2 ** 2)  # 1.5x需要先做space_to_depth(2)
#             cl_out_channels = out_channels * (3 ** 2)  # 然后depth_to_space(3)
#             cl_kernel_size = (1, 1)
#             cl_padding = 0
#         else:
#             cl_in_channels = num_channels
#             cl_out_channels = out_channels * (self.scaling_factor ** 2)
#             cl_kernel_size = (3, 3)
#             cl_padding = 1

#         self.conv_last = nn.Conv2d(in_channels=cl_in_channels, out_channels=cl_out_channels, 
#                                    kernel_size=cl_kernel_size, padding=cl_padding)

#         # ITO连接（当前未使用）
#         if use_ito_connection:
#             self.add_op = AddOp()
#             if scaling_factor == 1.5:
#                 self.anchor = AnchorOp(scaling_factor=3, kernel_size=3, stride=2, padding=1,
#                                               freeze_weights=False)
#             else:
#                 self.anchor = AnchorOp(scaling_factor=self.scaling_factor,
#                                               freeze_weights=False)

#         # 空间变换层
#         if scaling_factor == 1.5:
#             self.space_to_depth = nn.PixelUnshuffle(2)
#             self.depth_to_space = nn.PixelShuffle(3)
#         else:
#             self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

#         self.clip_output = nn.Hardtanh(min_val=0., max_val=1.)

#         # 优化初始化（关键：残差连接+ kaiming初始化）
#         self.initialize()
#         self._is_dcr = False

#     def forward(self, input):
#         # 前向传播优化：添加残差连接，防止梯度消失
#         x = self.cnn[0](input)
#         x = self.cnn[1](x)
        
#         # 中间层带残差
#         residual = x
#         for i in range(2, len(self.cnn)):
#             x = self.cnn[i](x)
#             if i % 2 == 0 and i > 2:  # 每一层卷积后加残差
#                 x = x + residual * 0.1  # 残差缩放，稳定训练
#                 residual = x
        
#         if not self._has_integer_scaling_factor:
#             x = self.space_to_depth(x)

#         if self._use_ito_connection:
#             residual = self.conv_last(x)
#             input_convolved = self.anchor(input)
#             x = self.add_op(input_convolved, residual)
#         else:
#             x = self.conv_last(x)

#         x = self.clip_output(x)
#         out = self.depth_to_space(x)
        
#         # 关键修复：尺寸对齐（解决1023x1023匹配问题）
#         if out.size(-1) != 1023 or out.size(-2) != 1023:
#             out = nn.functional.interpolate(out, size=(1023, 1023), mode='bicubic', align_corners=False)
        
#         return out
    
#     def to_dcr(self):
#         if not self._is_dcr:
#             if self.scaling_factor == 1.5:
#                 self.conv_last = convert_conv_following_space_to_depth_to_dcr(self.conv_last, 2)
#                 self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, 3)
#                 if self._use_ito_connection:
#                     self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, 3)
#             else:
#                 self.conv_last = convert_conv_preceding_depth_to_space_to_dcr(self.conv_last, self.scaling_factor)
#                 if self._use_ito_connection:
#                     self.anchor = convert_conv_preceding_depth_to_space_to_dcr(self.anchor, self.scaling_factor)
#             self._is_dcr = True

#     def initialize(self):
#         """优化初始化：kaiming + 残差初始化，加速收敛"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # 使用kaiming初始化，适配Hardtanh激活
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
        
#         # 残差连接初始化（核心：让初始输出接近输入）
#         for conv in self.residual_weights:
#             nn.init.constant_(conv.weight, 0.01)  # 小权重，避免初始震荡
        
#         # 最后一层初始化（降低初始输出幅值）
#         nn.init.normal_(self.conv_last.weight, 0, 0.01)
#         if self.conv_last.bias is not None:
#             nn.init.zeros_(self.conv_last.bias)

# # 教师模型（32通道，2中间层，1.5x超分）
# class QuickSRNetSmall(QuickSRNetBase):
#     def __init__(self, scaling_factor, num_channels=32, num_intermediate_layers=2, **kwargs):
#         super().__init__(
#             scaling_factor,
#             num_channels=num_channels,
#             num_intermediate_layers=num_intermediate_layers,
#             use_ito_connection=False,
#             **kwargs
#         )

# # 学生模型（16通道，1中间层，轻量化）
# class QuickSRNetTiny(QuickSRNetBase):
#     def __init__(self, scaling_factor, **kwargs):
#         super().__init__(
#             scaling_factor,
#             num_channels=16,
#             num_intermediate_layers=1,
#             use_ito_connection=False,
#             **kwargs
#         )

# # ---------------------------
# # 工具函数：加载教师模型权重
# # ---------------------------
# def load_teacher_checkpoint(model, checkpoint_path, device):
#     """加载教师模型的checkpoint，自动适配参数匹配"""
#     try:
#         # 加载checkpoint
#         checkpoint = torch.load(
#             checkpoint_path, 
#             map_location=device,
#             weights_only=False
#         )
        
#         # 提取模型权重（兼容多种格式）
#         state_dict = None
#         for key in ['state_dict', 'model_state_dict', 'model']:
#             if key in checkpoint:
#                 state_dict = checkpoint[key]
#                 break
#         if state_dict is None:
#             state_dict = checkpoint
        
#         # 移除module.前缀（多GPU训练的模型）
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 new_state_dict[k[7:]] = v
#             else:
#                 new_state_dict[k] = v
        
#         # 自动过滤不匹配的权重键
#         model_state_dict = model.state_dict()
#         filtered_state_dict = {}
        
#         for k, v in new_state_dict.items():
#             if k in model_state_dict and model_state_dict[k].shape == v.shape:
#                 filtered_state_dict[k] = v
#             else:
#                 if k in model_state_dict:
#                     print(f"⚠️ 权重尺寸不匹配 {k}: checkpoint={v.shape}, model={model_state_dict[k].shape}")
#                 else:
#                     print(f"⚠️ 模型中无此权重键 {k}，已跳过")
        
#         # 加载过滤后的权重
#         model.load_state_dict(filtered_state_dict, strict=False)
        
#         # 验证加载结果
#         loaded_keys = len(filtered_state_dict)
#         total_keys = len(model_state_dict)
#         print(f"✅ 成功加载教师模型权重：{checkpoint_path}")
#         print(f"📌 加载统计：共{loaded_keys}/{total_keys}个权重参数匹配并加载")
        
#         return model
    
#     except Exception as e:
#         raise RuntimeError(
#             f"❌ 加载教师模型权重失败：{str(e)}\n"
#             f"💡 检查点：\n"
#             f"  1. 确认checkpoint路径正确：{checkpoint_path}\n"
#             f"  2. 确认模型参数匹配\n"
#             f"  3. 确认PyTorch版本兼容（建议2.0+）"
#         )

# # ---------------------------
# # 蒸馏损失定义（优化损失权重）
# # ---------------------------
# class SRDistillationLoss(nn.Module):
#     """超分辨率蒸馏损失：优化权重，加速收敛"""
#     def __init__(self, alpha=0.7):  # 核心：提升软损失权重到0.7，更依赖教师
#         super().__init__()
#         self.alpha = alpha
#         self.mse_loss = nn.MSELoss()
#         self.l1_loss = nn.L1Loss()  # 加入L1损失，稳定训练

#     def forward(self, student_output, teacher_output, hr_target):
#         # 混合损失：L1+MSE，降低震荡
#         hard_loss = 0.3 * self.l1_loss(student_output, hr_target) + 0.7 * self.mse_loss(student_output, hr_target)
#         # 软损失：学生对齐教师（权重提升）
#         soft_loss = self.mse_loss(student_output, teacher_output)
#         # 总损失：重点偏向软损失
#         total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
#         return total_loss, hard_loss, soft_loss

# # ---------------------------
# # 数据集定义（优化数据增强）
# # ---------------------------
# class SRDataset(Dataset):
#     """
#     超分辨率数据集：
#     1. 直接从原始HR图像 左上/右上/左下/右下 裁剪出 1023x1023
#     2. 增加数据增强，提升泛化
#     """
#     def __init__(self, 
#                  hr_dir, 
#                  scaling_factor=1.5, 
#                  transform=None,
#                  is_train=True):
#         self.hr_dir = hr_dir
#         self.scaling_factor = scaling_factor
#         self.transform = transform
#         self.is_train = is_train
#         self.hr_filenames = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

#         # 固定裁剪尺寸：1023x1023
#         self.crop_h, self.crop_w = 1023, 1023
#         self.fixed_hr_size = (1023, 1023)
#         # LR尺寸：1023 * 2/3 = 682（偶数）
#         self.fixed_lr_h = 682
#         self.fixed_lr_w = 682

#         self.lr_interpolation = Image.BICUBIC
#         self.hr_interpolation = Image.BICUBIC

#     def _crop_1023_from_four_corners(self, img):
#         """从图像四个角落裁剪1023x1023，尺寸不足先缩放"""
#         W, H = img.size
#         target_w, target_h = self.crop_w, self.crop_h

#         # 图像尺寸不足时，先缩放到≥1023x1023
#         if W < target_w or H < target_h:
#             scale = max(target_w / W, target_h / H)
#             new_W = int(W * scale)
#             new_H = int(H * scale)
#             img = img.resize((new_W, new_H), self.hr_interpolation)
#             W, H = new_W, new_H

#         # 四个角落的裁剪坐标
#         corners = [
#             (0,          0,          target_w,                target_h               ),  # 左上
#             (W-target_w, 0,          W,                       target_h               ),  # 右上
#             (0,          H-target_h, target_w,                H                      ),  # 左下
#             (W-target_w, H-target_h, W,                       H                      ),  # 右下
#         ]

#         # 训练时随机选，验证时固定左上
#         if self.is_train:
#             chosen = random.choice(corners)
#         else:
#             chosen = corners[0]
#         return img.crop(chosen)

#     def __len__(self):
#         return len(self.hr_filenames)

#     def __getitem__(self, idx):
#         # 加载HR图像
#         hr_path = os.path.join(self.hr_dir, self.hr_filenames[idx])
#         try:
#             hr_img = Image.open(hr_path).convert('RGB')
#         except Exception as e:
#             raise RuntimeError(f"加载图像失败 {hr_path}：{str(e)}")

#         # 核心步骤：直接裁剪1023x1023
#         hr = self._crop_1023_from_four_corners(hr_img)
#         # 生成LR图像（682x682）
#         lr = hr.resize((self.fixed_lr_w, self.fixed_lr_h), self.lr_interpolation)

#         # 训练集数据增强（核心：提升泛化，加速收敛）
#         if self.is_train:
#             # 随机水平翻转
#             if random.random() < 0.5:
#                 lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
#                 hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
#             # 随机亮度调整（轻微）
#             if random.random() < 0.3:
#                 from torchvision.transforms import ColorJitter
#                 jitter = ColorJitter(brightness=0.1)
#                 lr = jitter(lr)
#                 hr = jitter(hr)

#         # 应用变换
#         if self.transform is not None:
#             lr = self.transform(lr)
#             hr = self.transform(hr)

#         return lr, hr

# # ---------------------------
# # 评估指标计算（优化稳定性）
# # ---------------------------
# def calculate_psnr(img1, img2, max_val=1.0):
#     """计算PSNR（峰值信噪比），增加数值稳定性"""
#     img1 = img1.clamp(0.0, 1.0)
#     img2 = img2.clamp(0.0, 1.0)
#     # 增加极小值，避免除零
#     mse = torch.mean((img1 - img2) ** 2) + 1e-8
#     return 20 * math.log10(max_val / math.sqrt(mse.item()))

# # ---------------------------
# # 训练函数（优化训练策略）
# # ---------------------------
# def train_distillation(
#     teacher_model,
#     student_model,
#     train_loader,
#     val_loader,
#     criterion,
#     optimizer,
#     device,
#     epochs=50,
#     save_path='sr_distillation_best.pth',
#     log_interval=10,
#     lr_scheduler=None
# ):
#     """超分辨率蒸馏训练函数（优化收敛策略）"""
#     # 固定教师模型参数
#     teacher_model.eval()
#     for param in teacher_model.parameters():
#         param.requires_grad = False
    
#     student_model.train()
#     best_val_psnr = 0.0
#     # 早停机制（防止过拟合，同时监控收敛）
#     early_stop_patience = 15
#     early_stop_counter = 0
    
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         epoch_hard_loss = 0.0
#         epoch_soft_loss = 0.0
        
#         for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
#             lr_imgs = lr_imgs.to(device, non_blocking=True)
#             hr_imgs = hr_imgs.to(device, non_blocking=True)
            
#             # 前向传播
#             optimizer.zero_grad()
            
#             # 教师模型推理（无梯度）
#             with torch.no_grad():
#                 teacher_output = teacher_model(lr_imgs)
            
#             # 学生模型推理
#             student_output = student_model(lr_imgs)
            
#             # 计算蒸馏损失
#             total_loss, hard_loss, soft_loss = criterion(student_output, teacher_output, hr_imgs)
            
#             # 梯度裁剪（核心：防止梯度爆炸，稳定训练）
#             torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
            
#             # 反向传播
#             total_loss.backward()
#             optimizer.step()
            
#             # 累计损失
#             epoch_loss += total_loss.item()
#             epoch_hard_loss += hard_loss.item()
#             epoch_soft_loss += soft_loss.item()
            
#             # 打印日志
#             if (batch_idx + 1) % log_interval == 0:
#                 print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
#                       f'Total Loss: {total_loss.item():.6f}, '
#                       f'Hard Loss: {hard_loss.item():.6f}, Soft Loss: {soft_loss.item():.6f}')
        
#         # 学习率调度（优化：预热+余弦退火）
#         if lr_scheduler is not None:
#             lr_scheduler.step()
#             current_lr = lr_scheduler.get_last_lr()[0]
#             print(f'Current learning rate: {current_lr:.6e}')
        
#         # 计算epoch平均损失
#         avg_loss = epoch_loss / len(train_loader)
#         avg_hard_loss = epoch_hard_loss / len(train_loader)
#         avg_soft_loss = epoch_soft_loss / len(train_loader)
#         print(f'\nEpoch [{epoch+1}/{epochs}] Summary:')
#         print(f'Average Total Loss: {avg_loss:.6f}, Average Hard Loss: {avg_hard_loss:.6f}, Average Soft Loss: {avg_soft_loss:.6f}')
        
#         # 验证阶段
#         val_psnr = validate(student_model, val_loader, device)
#         print(f'Validation PSNR: {val_psnr:.2f} dB\n')
        
#         # 保存最优模型
#         if val_psnr > best_val_psnr:
#             best_val_psnr = val_psnr
#             torch.save({
#                 'epoch': epoch + 1,
#                 'student_model_state_dict': student_model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
#                 'best_psnr': best_val_psnr,
#                 'loss': avg_loss
#             }, save_path)
#             print(f'Saved best model with PSNR: {best_val_psnr:.2f} dB\n')
#             early_stop_counter = 0  # 重置早停计数器
#         else:
#             early_stop_counter += 1
#             if early_stop_counter >= early_stop_patience:
#                 print(f'⚠️ 早停触发：验证集PSNR连续{early_stop_patience}轮未提升')
#                 break

# def validate(model, val_loader, device):
#     """验证函数，计算PSNR"""
#     model.eval()
#     total_psnr = 0.0
#     with torch.no_grad():
#         for lr_imgs, hr_imgs in val_loader:
#             lr_imgs = lr_imgs.to(device, non_blocking=True)
#             hr_imgs = hr_imgs.to(device, non_blocking=True)
            
#             # 模型推理
#             sr_imgs = model(lr_imgs)
            
#             # 计算PSNR
#             psnr = calculate_psnr(sr_imgs, hr_imgs)
#             total_psnr += psnr
    
#     avg_psnr = total_psnr / len(val_loader)
#     model.train()
#     return avg_psnr

# # ---------------------------
# # 主函数（训练入口，优化参数）
# # ---------------------------
# def main():
#     # 1. 核心配置（优化收敛参数）
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     scaling_factor = 1.5  # 匹配教师模型的1.5x超分
#     batch_size = 8  # 核心：减小批次，提升稳定性（从16→8）
#     epochs = 80
#     base_lr = 2e-4  # 核心：提高初始学习率，加速收敛（从1e-4→2e-4）
#     alpha = 0.7     # 核心：提升软损失权重，更依赖教师（从0.3→0.7）
#     weight_decay = 1e-6  # 核心：降低正则，提升拟合（从1e-5→1e-6）
    
#     # 2. 路径配置（请根据你的实际路径修改）
#     teacher_ckpt_path = '/home/huqiongyang/code/CATANet/experiments/train_quicksr_x1.5_scratch_channel16_layer_1/models/checkpoint_float32.pth.tar'
#     train_hr_dir = '/home/huqiongyang/code/CATANet/datasets/Div2K/DIV2K_train_HR/DIV2K_train_HR/DIV2K_train_HR'
#     val_hr_dir = '/home/huqiongyang/code/CATANet/datasets/bsd100/HR'
#     save_path = 'quick_sr_1.5x_distillation_best.pth'
    
#     # 3. 数据变换（简化，避免过度归一化）
#     transform = transforms.Compose([
#         transforms.ToTensor(),  # 仅转换为Tensor，保持[0,1]范围
#     ])
    
#     # 4. 构建数据集和数据加载器
#     print(f"📥 加载数据集，缩放因子：{scaling_factor}x")
#     # 训练集（启用增强）
#     train_dataset = SRDataset(
#         train_hr_dir, 
#         scaling_factor=scaling_factor, 
#         transform=transform,
#         is_train=True
#     )
#     # 验证集（禁用增强）
#     val_dataset = SRDataset(
#         val_hr_dir, 
#         scaling_factor=scaling_factor, 
#         transform=transform,
#         is_train=False
#     )
    
#     # 数据加载器
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=4,
#         pin_memory=True,
#         drop_last=True
#     )
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=batch_size*2,  # 验证集批次翻倍，加快评估
#         shuffle=False, 
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # 打印数据集信息
#     print(f"📊 训练集大小：{len(train_dataset)}, 验证集大小：{len(val_dataset)}")
#     print(f"📏 固定HR尺寸：{train_dataset.fixed_hr_size}, 固定LR尺寸：({train_dataset.fixed_lr_w}, {train_dataset.fixed_lr_h})")
    
#     # 5. 初始化教师模型
#     print(f"🔧 初始化教师模型（32通道，2中间层），加载权重：{teacher_ckpt_path}")
#     teacher_model = QuickSRNetSmall(
#         scaling_factor=scaling_factor,
#         num_channels=32,          
#         num_intermediate_layers=2
#     ).to(device)
#     # 保存初始化参数（用于日志）
#     teacher_model._init_kwargs = {
#         'num_channels': 32,
#         'num_intermediate_layers': 2
#     }
    
#     # 加载预训练权重
#     teacher_model = load_teacher_checkpoint(teacher_model, teacher_ckpt_path, device)
    
#     # 6. 初始化学生模型
#     print("🔧 初始化学生模型（QuickSRNetTiny，16通道，1中间层）")
#     student_model = QuickSRNetTiny(scaling_factor=scaling_factor).to(device)
    
#     # 7. 定义损失函数、优化器、学习率调度器（优化）
#     criterion = SRDistillationLoss(alpha=alpha)
#     optimizer = optim.AdamW(
#         student_model.parameters(),
#         lr=base_lr,
#         betas=(0.9, 0.999),
#         weight_decay=weight_decay,
#         eps=1e-8  # 增加数值稳定性
#     )
    
#     # 学习率调度器（优化：预热+余弦退火）
#     def lr_lambda(epoch):
#         # 前5轮预热：学习率从0→base_lr
#         if epoch < 5:
#             return epoch / 5
#         # 5轮后余弦退火：缓慢降低
#         else:
#             return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (epochs - 5)))
    
#     lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
#     # 8. 开始蒸馏训练
#     print(f'\n🚀 开始蒸馏训练 on {device}...')
#     print(f'👨‍🏫 教师模型：QuickSRNetSmall (32通道, 2中间层, 1.5x超分)')
#     print(f'👨‍🎓 学生模型：QuickSRNetTiny (16通道, 1中间层, 1.5x超分)')
#     print(f'📊 批次大小：{batch_size}, 训练轮数：{epochs}, 初始学习率：{base_lr}')
#     print(f'⚙️ 蒸馏权重α：{alpha}, 权重衰减：{weight_decay}')
    
#     train_distillation(
#         teacher_model=teacher_model,
#         student_model=student_model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         device=device,
#         epochs=epochs,
#         save_path=save_path,
#         log_interval=10,
#         lr_scheduler=lr_scheduler
#     )

# if __name__ == '__main__':
#     main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import math
import os
import random
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 基础组件定义（保留原结构）
# ---------------------------
class AddOp(nn.Module):
    """简单的加法操作层，适配张量广播"""
    def forward(self, x1, x2):
        return x1 + x2

class AnchorOp(nn.Conv2d):
    """Anchor操作层，本质是卷积层"""
    def __init__(self, scaling_factor, kernel_size=3, stride=1, padding=1, freeze_weights=False, in_channels=3, out_channels=None):
        if out_channels is None:
            out_channels = 3 * (scaling_factor ** 2)
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=kernel_size, stride=stride, padding=padding)
        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

def convert_conv_following_space_to_depth_to_dcr(conv, scale):
    """适配space_to_depth后的卷积转换（保持原接口）"""
    return conv

def convert_conv_preceding_depth_to_space_to_dcr(conv, scale):
    """适配depth_to_space前的卷积转换（保持原接口）"""
    return conv

# ---------------------------
# 教师/学生模型定义（优化初始化+残差）
# ---------------------------
class QuickSRNetBase(nn.Module):
    """基础超分辨率网络（优化初始化，加速收敛）"""
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

        # 构建中间层（添加残差连接）
        intermediate_layers = []
        self.residual_weights = []  # 保存残差权重，用于初始化
        for i in range(num_intermediate_layers):
            conv = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1)
            intermediate_layers.extend([
                conv,
                nn.Hardtanh(min_val=0., max_val=1.)
            ])
            self.residual_weights.append(conv)  # 记录卷积层，用于残差初始化

        # 构建完整CNN序列
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.Hardtanh(min_val=0., max_val=1.),
            *intermediate_layers,
        )

        # 最后一层卷积配置（适配1.5x缩放）
        if scaling_factor == 1.5:
            cl_in_channels = num_channels * (2 ** 2)  # 1.5x需要先做space_to_depth(2)
            cl_out_channels = out_channels * (3 ** 2)  # 然后depth_to_space(3)
            cl_kernel_size = (1, 1)
            cl_padding = 0
        else:
            cl_in_channels = num_channels
            cl_out_channels = out_channels * (self.scaling_factor ** 2)
            cl_kernel_size = (3, 3)
            cl_padding = 1

        self.conv_last = nn.Conv2d(in_channels=cl_in_channels, out_channels=cl_out_channels, 
                                   kernel_size=cl_kernel_size, padding=cl_padding)

        # ITO连接（当前未使用）
        if use_ito_connection:
            self.add_op = AddOp()
            if scaling_factor == 1.5:
                self.anchor = AnchorOp(scaling_factor=3, kernel_size=3, stride=2, padding=1,
                                              freeze_weights=False)
            else:
                self.anchor = AnchorOp(scaling_factor=self.scaling_factor,
                                              freeze_weights=False)

        # 空间变换层
        if scaling_factor == 1.5:
            self.space_to_depth = nn.PixelUnshuffle(2)
            self.depth_to_space = nn.PixelShuffle(3)
        else:
            self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

        self.clip_output = nn.Hardtanh(min_val=0., max_val=1.)

        # 优化初始化（关键：残差连接+ kaiming初始化）
        self.initialize()
        self._is_dcr = False

    def forward(self, input):
        # 前向传播优化：添加残差连接，防止梯度消失
        x = self.cnn[0](input)
        x = self.cnn[1](x)
        
        # 中间层带残差
        residual = x
        for i in range(2, len(self.cnn)):
            x = self.cnn[i](x)
            if i % 2 == 0 and i > 2:  # 每一层卷积后加残差
                x = x + residual * 0.1  # 残差缩放，稳定训练
                residual = x
        
        if not self._has_integer_scaling_factor:
            x = self.space_to_depth(x)

        if self._use_ito_connection:
            residual = self.conv_last(x)
            input_convolved = self.anchor(input)
            x = self.add_op(input_convolved, residual)
        else:
            x = self.conv_last(x)

        x = self.clip_output(x)
        out = self.depth_to_space(x)
        
        # 关键修复：尺寸对齐（解决1023x1023匹配问题）
        if out.size(-1) != 1023 or out.size(-2) != 1023:
            out = nn.functional.interpolate(out, size=(1023, 1023), mode='bicubic', align_corners=False)
        
        return out
    
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

    def initialize(self):
        """优化初始化：kaiming + 残差初始化，加速收敛"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用kaiming初始化，适配Hardtanh激活
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 残差连接初始化（核心：让初始输出接近输入）
        for conv in self.residual_weights:
            nn.init.constant_(conv.weight, 0.01)  # 小权重，避免初始震荡
        
        # 最后一层初始化（降低初始输出幅值）
        nn.init.normal_(self.conv_last.weight, 0, 0.01)
        if self.conv_last.bias is not None:
            nn.init.zeros_(self.conv_last.bias)

# 教师模型（32通道，2中间层，1.5x超分）
class QuickSRNetSmall(QuickSRNetBase):
    def __init__(self, scaling_factor, num_channels=32, num_intermediate_layers=2, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=num_channels,
            num_intermediate_layers=num_intermediate_layers,
            use_ito_connection=False,
            **kwargs
        )

# 学生模型（16通道，1中间层，轻量化）
class QuickSRNetTiny(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(
            scaling_factor,
            num_channels=16,
            num_intermediate_layers=1,
            use_ito_connection=False,
            **kwargs
        )

# ---------------------------
# 工具函数：加载教师模型权重
# ---------------------------
def load_teacher_checkpoint(model, checkpoint_path, device):
    """加载教师模型的checkpoint，自动适配参数匹配"""
    try:
        # 加载checkpoint
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=device,
            weights_only=False
        )
        
        # 提取模型权重（兼容多种格式）
        state_dict = None
        for key in ['state_dict', 'model_state_dict', 'model']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
        
        # 移除module.前缀（多GPU训练的模型）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # 自动过滤不匹配的权重键
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for k, v in new_state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                if k in model_state_dict:
                    print(f"⚠️ 权重尺寸不匹配 {k}: checkpoint={v.shape}, model={model_state_dict[k].shape}")
                else:
                    print(f"⚠️ 模型中无此权重键 {k}，已跳过")
        
        # 加载过滤后的权重
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # 验证加载结果
        loaded_keys = len(filtered_state_dict)
        total_keys = len(model_state_dict)
        print(f"✅ 成功加载教师模型权重：{checkpoint_path}")
        print(f"📌 加载统计：共{loaded_keys}/{total_keys}个权重参数匹配并加载")
        
        return model
    
    except Exception as e:
        raise RuntimeError(
            f"❌ 加载教师模型权重失败：{str(e)}\n"
            f"💡 检查点：\n"
            f"  1. 确认checkpoint路径正确：{checkpoint_path}\n"
            f"  2. 确认模型参数匹配\n"
            f"  3. 确认PyTorch版本兼容（建议2.0+）"
        )

# ---------------------------
# 蒸馏损失定义（优化损失权重）
# # ---------------------------
class SRDistillationLoss(nn.Module):
    """超分辨率蒸馏损失：优化权重，加速收敛"""
    def __init__(self, alpha=0.1):  # 核心：提升软损失权重到0.7，更依赖教师
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()  # 加入L1损失，稳定训练

    def forward(self, student_output, teacher_output, hr_target):
        # 混合损失：L1+MSE，降低震荡
        hard_loss = 0.8 * self.l1_loss(student_output, hr_target) + 0.2 * self.mse_loss(student_output, hr_target)
        # 软损失：学生对齐教师（权重提升）
        soft_loss = self.mse_loss(student_output, teacher_output)
        # 总损失：重点偏向软损失
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return total_loss, hard_loss, soft_loss
# class SRDistillationLoss(nn.Module):
#     def __init__(self, alpha=0.0):
#         super().__init__()
#         self.l1 = nn.L1Loss()

#     def forward(self, student_output, teacher_output, hr_target):
#         # 只学真实图，不学教师
#         loss = self.l1(student_output, hr_target)
#         return loss, loss, loss
# ---------------------------
# 数据集定义（优化数据增强）
# ---------------------------
class SRDataset(Dataset):
    """
    超分辨率数据集：
    1. 直接从原始HR图像 左上/右上/左下/右下 裁剪出 1023x1023
    2. 增加数据增强，提升泛化
    """
    def __init__(self, 
                 hr_dir, 
                 scaling_factor=1.5, 
                 transform=None,
                 is_train=True):
        self.hr_dir = hr_dir
        self.scaling_factor = scaling_factor
        self.transform = transform
        self.is_train = is_train
        self.hr_filenames = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 固定裁剪尺寸：1023x1023
        self.crop_h, self.crop_w = 1023, 1023
        self.fixed_hr_size = (1023, 1023)
        # LR尺寸：1023 * 2/3 = 682（偶数）
        self.fixed_lr_h = 682
        self.fixed_lr_w = 682

        self.lr_interpolation = Image.BICUBIC
        self.hr_interpolation = Image.BICUBIC

    def _crop_1023_from_four_corners(self, img):
        """从图像四个角落裁剪1023x1023，尺寸不足先缩放"""
        W, H = img.size
        target_w, target_h = self.crop_w, self.crop_h

        # 图像尺寸不足时，先缩放到≥1023x1023
        if W < target_w or H < target_h:
            scale = max(target_w / W, target_h / H)
            new_W = int(W * scale)
            new_H = int(H * scale)
            img = img.resize((new_W, new_H), self.hr_interpolation)
            W, H = new_W, new_H

        # 四个角落的裁剪坐标
        corners = [
            (0,          0,          target_w,                target_h               ),  # 左上
            (W-target_w, 0,          W,                       target_h               ),  # 右上
            (0,          H-target_h, target_w,                H                      ),  # 左下
            (W-target_w, H-target_h, W,                       H                      ),  # 右下
        ]

        # 训练时随机选，验证时固定左上
        if self.is_train:
            chosen = random.choice(corners)
        else:
            chosen = corners[0]
        return img.crop(chosen)

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, idx):
        # 加载HR图像
        hr_path = os.path.join(self.hr_dir, self.hr_filenames[idx])
        try:
            hr_img = Image.open(hr_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"加载图像失败 {hr_path}：{str(e)}")

        # 核心步骤：直接裁剪1023x1023
        hr = self._crop_1023_from_four_corners(hr_img)
        # 生成LR图像（682x682）
        lr = hr.resize((self.fixed_lr_w, self.fixed_lr_h), self.lr_interpolation)

        # 训练集数据增强（核心：提升泛化，加速收敛）
        if self.is_train:
            # 随机水平翻转
            if random.random() < 0.5:
                lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            # 随机亮度调整（轻微）
            if random.random() < 0.3:
                from torchvision.transforms import ColorJitter
                jitter = ColorJitter(brightness=0.1)
                lr = jitter(lr)
                hr = jitter(hr)

        # 应用变换
        if self.transform is not None:
            lr = self.transform(lr)
            hr = self.transform(hr)

        return lr, hr

# ---------------------------
# 评估指标计算（优化稳定性）
# ---------------------------
def calculate_psnr(img1, img2, max_val=1.0):
    """计算PSNR（峰值信噪比），增加数值稳定性"""
    img1 = img1.clamp(0.0, 1.0)
    img2 = img2.clamp(0.0, 1.0)
    # 增加极小值，避免除零
    mse = torch.mean((img1 - img2) ** 2) + 1e-8
    return 20 * math.log10(max_val / math.sqrt(mse.item()))

# ---------------------------
# 训练函数（优化训练策略）
# ---------------------------
def train_distillation(
    teacher_model,
    student_model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=50,
    save_path='sr_distillation_best.pth',
    log_interval=10,
    lr_scheduler=None
):
    """超分辨率蒸馏训练函数（优化收敛策略）"""
    # 固定教师模型参数
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    student_model.train()
    best_val_psnr = 0.0
    # 早停机制（防止过拟合，同时监控收敛）
    early_stop_patience = 15
    early_stop_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_hard_loss = 0.0
        epoch_soft_loss = 0.0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 教师模型推理（无梯度）
            with torch.no_grad():
                teacher_output = teacher_model(lr_imgs)
            
            # 学生模型推理
            student_output = student_model(lr_imgs)
            
            # 计算蒸馏损失
            total_loss, hard_loss, soft_loss = criterion(student_output, teacher_output, hr_imgs)
            
            # 梯度裁剪（核心：防止梯度爆炸，稳定训练）
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 累计损失
            epoch_loss += total_loss.item()
            epoch_hard_loss += hard_loss.item()
            epoch_soft_loss += soft_loss.item()
            
            # 打印日志
            if (batch_idx + 1) % log_interval == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Total Loss: {total_loss.item():.6f}, '
                      f'Hard Loss: {hard_loss.item():.6f}, Soft Loss: {soft_loss.item():.6f}')
        
        # 学习率调度（优化：预热+余弦退火）
        if lr_scheduler is not None:
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
            print(f'Current learning rate: {current_lr:.6e}')
        
        # 计算epoch平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_hard_loss = epoch_hard_loss / len(train_loader)
        avg_soft_loss = epoch_soft_loss / len(train_loader)
        print(f'\nEpoch [{epoch+1}/{epochs}] Summary:')
        print(f'Average Total Loss: {avg_loss:.6f}, Average Hard Loss: {avg_hard_loss:.6f}, Average Soft Loss: {avg_soft_loss:.6f}')
        
        # 验证阶段
        val_psnr = validate(student_model, val_loader, device)
        print(f'Validation PSNR: {val_psnr:.2f} dB\n')
        
        # 保存最优模型
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                'epoch': epoch + 1,
                'student_model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                'best_psnr': best_val_psnr,
                'loss': avg_loss
            }, save_path)
            print(f'Saved best model with PSNR: {best_val_psnr:.2f} dB\n')
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f'⚠️ 早停触发：验证集PSNR连续{early_stop_patience}轮未提升')
                break

def validate(model, val_loader, device):
    """验证函数，计算PSNR"""
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            
            # 模型推理
            sr_imgs = model(lr_imgs)
            
            # 计算PSNR
            psnr = calculate_psnr(sr_imgs, hr_imgs)
            total_psnr += psnr
    
    avg_psnr = total_psnr / len(val_loader)
    model.train()
    return avg_psnr

# ---------------------------
# 主函数（训练入口，优化参数）
# ---------------------------
def main():
    # 1. 核心配置（优化收敛参数）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaling_factor = 1.5  # 匹配教师模型的1.5x超分
    batch_size = 8  # 核心：减小批次，提升稳定性（从16→8）
    epochs = 80
    base_lr = 2e-4  # 核心：提高初始学习率，加速收敛（从1e-4→2e-4）
    alpha = 0.2     # 核心：提升软损失权重，更依赖教师（从0.3→0.7）
    weight_decay = 1e-6  # 核心：降低正则，提升拟合（从1e-5→1e-6）
    
    # 2. 路径配置（请根据你的实际路径修改）
    teacher_ckpt_path = '/home/huqiongyang/code/CATANet/experiments/train_quicksr_x1.5_scratch_channel16_layer_1/models/checkpoint_float32.pth.tar'
    train_hr_dir = '/home/huqiongyang/code/CATANet/datasets/Div2K/DIV2K_train_HR/DIV2K_train_HR/DIV2K_train_HR'
    val_hr_dir = '/home/huqiongyang/code/CATANet/datasets/bsd100/HR'
    save_path = 'quick_sr_1.5x_distillation_best.pth'
    
    # 3. 数据变换（简化，避免过度归一化）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 仅转换为Tensor，保持[0,1]范围
    ])
    
    # 4. 构建数据集和数据加载器
    print(f"📥 加载数据集，缩放因子：{scaling_factor}x")
    # 训练集（启用增强）
    train_dataset = SRDataset(
        train_hr_dir, 
        scaling_factor=scaling_factor, 
        transform=transform,
        is_train=True
    )
    # 验证集（禁用增强）
    val_dataset = SRDataset(
        val_hr_dir, 
        scaling_factor=scaling_factor, 
        transform=transform,
        is_train=False
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size*2,  # 验证集批次翻倍，加快评估
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 打印数据集信息
    print(f"📊 训练集大小：{len(train_dataset)}, 验证集大小：{len(val_dataset)}")
    print(f"📏 固定HR尺寸：{train_dataset.fixed_hr_size}, 固定LR尺寸：({train_dataset.fixed_lr_w}, {train_dataset.fixed_lr_h})")
    
    # 5. 初始化教师模型
    print(f"🔧 初始化教师模型（32通道，2中间层），加载权重：{teacher_ckpt_path}")
    teacher_model = QuickSRNetSmall(
        scaling_factor=scaling_factor,
        num_channels=32,          
        num_intermediate_layers=2
    ).to(device)
    # 保存初始化参数（用于日志）
    teacher_model._init_kwargs = {
        'num_channels': 32,
        'num_intermediate_layers': 2
    }
    
    # 加载预训练权重
    teacher_model = load_teacher_checkpoint(teacher_model, teacher_ckpt_path, device)
    
    # 6. 初始化学生模型
    print("🔧 初始化学生模型（QuickSRNetTiny，16通道，1中间层）")
    student_model = QuickSRNetTiny(scaling_factor=scaling_factor).to(device)
    
    # 7. 定义损失函数、优化器、学习率调度器（优化）
    criterion = SRDistillationLoss(alpha=alpha)
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        eps=1e-8  # 增加数值稳定性
    )
    
    # 学习率调度器（优化：预热+余弦退火）
    def lr_lambda(epoch):
        # 前5轮预热：学习率从0→base_lr
        if epoch < 5:
            return epoch / 5
        # 5轮后余弦退火：缓慢降低
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (epochs - 5)))
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 8. 开始蒸馏训练
    print(f'\n🚀 开始蒸馏训练 on {device}...')
    print(f'👨‍🏫 教师模型：QuickSRNetSmall (32通道, 2中间层, 1.5x超分)')
    print(f'👨‍🎓 学生模型：QuickSRNetTiny (16通道, 1中间层, 1.5x超分)')
    print(f'📊 批次大小：{batch_size}, 训练轮数：{epochs}, 初始学习率：{base_lr}')
    print(f'⚙️ 蒸馏权重α：{alpha}, 权重衰减：{weight_decay}')
    
    train_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        save_path=save_path,
        log_interval=10,
        lr_scheduler=lr_scheduler
    )

if __name__ == '__main__':
    main()