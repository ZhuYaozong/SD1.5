# 导入必要的库：命令行参数解析、操作系统功能、文件路径处理
import argparse, os, sys, glob
# 导入 OpenCV 库，用于图像处理
import cv2
# 导入 PyTorch，深度学习框架
import torch
# 导入 NumPy，用于数值计算
import numpy as np
# 导入 OmegaConf，用于配置文件解析
from omegaconf import OmegaConf
# 导入 PIL，用于图像处理
from PIL import Image
# 导入 tqdm，用于进度条显示
from tqdm import tqdm, trange
# 导入 WatermarkEncoder，用于添加水印
from imwatermark import WatermarkEncoder
# 导入 islice，用于迭代器切片
from itertools import islice
# 导入 rearrange，用于张量维度重排
from einops import rearrange
# 导入 make_grid，用于创建图像网格
from torchvision.utils import make_grid
# 导入 time，用于计时
import time
# 导入 seed_everything，用于设置随机种子
from pytorch_lightning import seed_everything
# 导入 autocast，用于自动混合精度计算
from torch import autocast
# 导入上下文管理器相关工具
from contextlib import contextmanager, nullcontext

# 导入配置实例化工具
from ldm.util import instantiate_from_config
# 导入 DDIM 采样器
from ldm.models.diffusion.ddim import DDIMSampler
# 导入 PLMS 采样器
from ldm.models.diffusion.plms import PLMSSampler
# 导入 DPM-Solver 采样器
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# 导入安全检查器
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# 导入特征提取器
from transformers import AutoFeatureExtractor


# 以下是原始的安全模型加载代码，已被注释掉
# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

# 用户注释：替换原始的 safety_feature_extractor / safety_checker
# 自定义安全检查函数，不执行实际检查，直接返回原图
def check_safety(x_image):
    # 不做任何 NSFW 检查，直接返回原图和全为 False 的列表
    return x_image, [False] * len(x_image)


# 定义 chunk 函数，用于将迭代器分割成指定大小的块
def chunk(it, size):
    # 将输入转换为迭代器
    it = iter(it)
    # 返回一个新的迭代器，每次产生 size 大小的元组，直到原迭代器耗尽
    return iter(lambda: tuple(islice(it, size)), ())


# 定义 numpy_to_pil 函数，将 NumPy 数组转换为 PIL 图像
def numpy_to_pil(images):
    # 函数文档字符串：将 NumPy 图像或批量图像转换为 PIL 图像
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    # 如果是单张图像（3维），添加批次维度
    if images.ndim == 3:
        images = images[None, ...]
    # 将像素值从 [0,1] 范围转换为 [0,255] 整数范围
    images = (images * 255).round().astype("uint8")
    # 将每个 NumPy 数组转换为 PIL 图像
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# 定义 load_model_from_config 函数，从配置和检查点加载模型
def load_model_from_config(config, ckpt, verbose=False):
    # 打印正在加载的模型路径
    print(f"Loading model from {ckpt}")
    # 加载模型检查点，使用 CPU 作为加载位置
    pl_sd = torch.load(ckpt, map_location="cpu")
    # 如果检查点包含 global_step 信息，打印出来
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    # 提取模型状态字典
    sd = pl_sd["state_dict"]
    # 根据配置实例化模型
    model = instantiate_from_config(config.model)
    # 加载模型权重，不使用严格匹配
    m, u = model.load_state_dict(sd, strict=False)
    # 如果有缺失的键并且启用了详细模式，打印缺失的键
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    # 如果有意外的键并且启用了详细模式，打印意外的键
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # 将模型移至 CUDA 设备
    model.cuda()
    # 设置模型为评估模式
    model.eval()
    return model


# 定义 put_watermark 函数，为图像添加水印
def put_watermark(img, wm_encoder=None):
    # 如果提供了水印编码器
    if wm_encoder is not None:
        # 将 PIL 图像转换为 OpenCV 格式（BGR）
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # 使用 dwtDct 方法添加水印
        img = wm_encoder.encode(img, 'dwtDct')
        # 将图像转换回 PIL 格式（RGB）
        img = Image.fromarray(img[:, :, ::-1])
    return img


# 定义 load_replacement 函数，加载替换图像
def load_replacement(x):
    try:
        # 获取输入图像的形状
        hwc = x.shape
        # 打开并调整替换图像的大小以匹配输入图像
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        # 将替换图像转换为 NumPy 数组并归一化
        y = (np.array(y)/255.0).astype(x.dtype)
        # 确保替换图像与输入图像形状相同
        assert y.shape == x.shape
        return y
    except Exception:
        # 如果出现任何错误，返回原始输入
        return x


# 以下是原始的 check_safety 函数，已被注释掉
# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept


# 定义主函数
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加提示词参数
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    # 添加输出目录参数
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    # 添加是否跳过网格保存的参数
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    # 添加是否跳过保存的参数
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    # 添加 DDIM 采样步数参数
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    # 添加是否使用 PLMS 采样的参数
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    # 添加是否使用 DPM-Solver 采样的参数
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    # 添加是否使用 LAION400M 模型的参数
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    # 添加是否使用固定初始代码的参数
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    # 添加 DDIM eta 参数
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    # 添加迭代次数参数
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    # 添加图像高度参数
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    # 添加图像宽度参数
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    # 添加潜在通道数参数
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    # 添加下采样因子参数
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    # 添加每个提示词生成的样本数参数
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    # 添加网格行数参数
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    # 添加无条件引导比例参数
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    # 添加从文件加载提示词的参数
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    # 添加配置文件路径参数
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    # 添加模型检查点路径参数
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    # 添加随机种子参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    # 添加计算精度参数
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # 解析命令行参数
    opt = parser.parse_args()

    # 如果指定了使用 LAION400M 模型
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        # 更新配置文件路径
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        # 更新模型检查点路径
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        # 更新输出目录
        opt.outdir = "outputs/txt2img-samples-laion400m"

    # 设置随机种子以确保结果可复现
    seed_everything(opt.seed)

    # 加载配置文件
    config = OmegaConf.load(f"{opt.config}")
    # 根据配置和检查点加载模型
    model = load_model_from_config(config, f"{opt.ckpt}")

    # 确定使用的设备（CUDA 或 CPU）
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 将模型移至指定设备
    model = model.to(device)

    # 根据参数选择采样器
    if opt.dpm_solver:
        # 使用 DPM-Solver 采样器
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        # 使用 PLMS 采样器
        sampler = PLMSSampler(model)
    else:
        # 默认使用 DDIM 采样器
        sampler = DDIMSampler(model)

    # 创建输出目录（如果不存在）
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # 创建不可见水印编码器
    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # 设置批次大小
    batch_size = opt.n_samples
    # 设置网格行数
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    # 如果不是从文件加载提示词
    if not opt.from_file:
        # 使用命令行参数中的提示词
        prompt = opt.prompt
        # 确保提示词不为 None
        assert prompt is not None
        # 创建数据列表，包含 batch_size 个相同的提示词
        data = [batch_size * [prompt]]

    else:
        # 从文件加载提示词
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            # 读取文件中的所有行作为提示词
            data = f.read().splitlines()
            # 将提示词分组成大小为 batch_size 的块
            data = list(chunk(data, batch_size))

    # 创建样本保存目录
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    # 获取当前样本数量，用于命名新生成的样本
    base_count = len(os.listdir(sample_path))
    # 获取当前网格数量，用于命名新生成的网格
    grid_count = len(os.listdir(outpath)) - 1

    # 初始代码设置为 None
    start_code = None
    # 如果启用了固定初始代码
    if opt.fixed_code:
        # 创建随机初始代码
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    # 设置精度上下文管理器
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    # 禁用梯度计算以提高性能
    with torch.no_grad():
        # 使用指定的精度上下文管理器
        with precision_scope("cuda"):
            # 使用模型的 EMA 权重（如果有）
            with model.ema_scope():
                # 记录开始时间
                tic = time.time()
                # 用于存储所有样本的列表
                all_samples = list()
                # 对每个迭代进行采样
                for n in trange(opt.n_iter, desc="Sampling"):
                    # 对每个提示词批次进行处理
                    for prompts in tqdm(data, desc="data"):
                        # 无条件条件设置为 None
                        uc = None
                        # 如果无条件引导比例不等于 1.0
                        if opt.scale != 1.0:
                            # 获取空提示词的条件嵌入
                            uc = model.get_learned_conditioning(batch_size * [""])
                        # 如果提示词是元组，转换为列表
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        # 获取提示词的条件嵌入
                        c = model.get_learned_conditioning(prompts)
                        # 计算潜在空间的形状
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        # 使用采样器生成样本
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        # 解码生成的潜在样本到像素空间
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        # 将像素值从 [-1, 1] 范围调整到 [0, 1] 范围
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        # 将张量从 (B, C, H, W) 转换为 (B, H, W, C) 并移至 CPU，转换为 NumPy 数组
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        # 执行安全检查（当前版本不执行实际检查）
                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        # 将检查后的图像转换回 PyTorch 张量，形状为 (B, C, H, W)
                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        # 如果不跳过保存
                        if not opt.skip_save:
                            # 对每个样本进行处理
                            for x_sample in x_checked_image_torch:
                                # 将像素值从 [0, 1] 范围转换为 [0, 255] 范围，并调整维度顺序为 (H, W, C)
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                # 将 NumPy 数组转换为 PIL 图像
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # 添加水印
                                img = put_watermark(img, wm_encoder)
                                # 保存图像
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                # 更新样本计数
                                base_count += 1

                        # 如果不跳过网格保存
                        if not opt.skip_grid:
                            # 将当前批次的样本添加到所有样本列表
                            all_samples.append(x_checked_image_torch)

                # 如果不跳过网格保存
                if not opt.skip_grid:
                    # 额外保存网格图像
                    # 将所有批次的样本堆叠成一个张量
                    grid = torch.stack(all_samples, 0)
                    # 重排维度，从 (n, b, c, h, w) 到 ((n*b), c, h, w)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    # 创建网格图像，每行显示 n_rows 个样本
                    grid = make_grid(grid, nrow=n_rows)

                    # 将网格张量转换为图像
                    # 将像素值从 [0, 1] 范围转换为 [0, 255] 范围，并调整维度顺序为 (H, W, C)
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    # 将 NumPy 数组转换为 PIL 图像
                    img = Image.fromarray(grid.astype(np.uint8))
                    # 添加水印
                    img = put_watermark(img, wm_encoder)
                    # 保存网格图像
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    # 更新网格计数
                    grid_count += 1

                # 记录结束时间
                toc = time.time()

    # 打印样本保存位置信息
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n" \
          f" \nEnjoy.")


# 如果作为主程序运行，执行 main 函数
if __name__ == "__main__":
    main()