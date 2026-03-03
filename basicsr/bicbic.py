import cv2
import os
import glob

# -------------------------- 配置参数（无需修改，和你的路径完全一致） --------------------------
# HR高清图目录
HR_DIR = '/home/huqiongyang/dataset/0116/test'
#'/home/huqiongyang/code/CATANet/datasets/Div2K/DIV2K_train_HR/DIV2K_train_HR/DIV2K_train_HR'
# LR低分辨率图保存根目录
LR_ROOT_DIR = '/home/huqiongyang/dataset/0116/LR'
#'/home/huqiongyang/code/CATANet/datasets/Div2K/DIV2K_train_HR/DIV2K_train_HR/DIV2K_LR_bicubic'
# 下采样倍率（1.5倍）
SCALE = 1.5
# 要处理的图片格式
IMG_FORMATS = ['*.png', '*.jpg', '*.jpeg']
# 插值方式：bicubic双三次插值（和BasicSR/Matlab一致）
INTERPOLATION = cv2.INTER_CUBIC

# -------------------------- 核心下采样函数 --------------------------
def bicubic_downsample(img, scale):
    """
    双三次下采样图片
    :param img: OpenCV读取的BGR格式图片
    :param scale: 下采样倍率（如1.5）
    :return: 下采样后的低分辨率图片
    """
    h, w = img.shape[:2]
    # 计算下采样后的尺寸，取整为整数（避免浮点尺寸）
    new_h = int(h / scale)
    new_w = int(w / scale)
    # 双三次下采样
    lr_img = cv2.resize(img, (new_w, new_h), interpolation=INTERPOLATION)
    return lr_img

# -------------------------- 批量处理主函数 --------------------------
def main():
    # 构建LR保存目录（x1.5子目录，和你的配置文件对应）
    LR_SAVE_DIR = os.path.join(LR_ROOT_DIR, f'x{SCALE}')
    # 自动创建目录（若不存在，递归创建）
    if not os.path.exists(LR_SAVE_DIR):
        os.makedirs(LR_SAVE_DIR)
        print(f"成功创建目录：{LR_SAVE_DIR}")
    else:
        print(f"目录已存在：{LR_SAVE_DIR}，将直接保存图片")

    # 遍历HR目录下所有指定格式的图片
    img_paths = []
    for fmt in IMG_FORMATS:
        img_paths.extend(glob.glob(os.path.join(HR_DIR, fmt)))
    # 按文件名排序（保证DIV2K图片按数字顺序处理）
    img_paths.sort()

    if not img_paths:
        print(f"错误：在{HR_DIR}中未找到{IMG_FORMATS}格式的图片！")
        return

    print(f"共找到 {len(img_paths)} 张图片，开始批量下采样1.5倍...")
    # 批量处理每张图片
    for idx, img_path in enumerate(img_paths, 1):
        # 获取图片文件名（如0001.png）
        img_name = os.path.basename(img_path)
        # 读取HR高清图（OpenCV默认BGR格式，不影响下采样，保存时自动还原）
        hr_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if hr_img is None:
            print(f"警告：跳过无法读取的图片：{img_name}")
            continue
        # 下采样生成LR图
        lr_img = bicubic_downsample(hr_img, SCALE)
        # 构建LR图保存路径
        lr_save_path = os.path.join(LR_SAVE_DIR, img_name)
        # 保存LR图（无损保存，和HR图格式一致）
        cv2.imwrite(lr_save_path, lr_img)
        # 打印进度
        if idx % 10 == 0 or idx == len(img_paths):
            print(f"处理进度：{idx}/{len(img_paths)} | 已保存：{lr_save_path}")

    print(f"\n批量处理完成！所有1.5倍低分辨率图已保存至：{LR_SAVE_DIR}")

if __name__ == '__main__':
    main()