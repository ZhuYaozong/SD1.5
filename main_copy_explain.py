# 导入基础库：命令行参数解析、操作系统接口、系统功能、日期时间、文件路径匹配、动态导入模块和CSV处理
import argparse, os, sys, datetime, glob, importlib, csv
# 导入数值计算库NumPy
import numpy as np
# 导入时间处理库
import time
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch视觉处理库
import torchvision
# 导入PyTorch Lightning高级训练框架
import pytorch_lightning as pl

# 导入版本处理库
from packaging import version
# 导入配置管理库OmegaConf
from omegaconf import OmegaConf
# 从PyTorch数据模块导入数据集相关功能
from torch.utils.data import random_split, DataLoader, Dataset, Subset
# 导入偏函数功能
from functools import partial
# 导入图像处理库PIL
from PIL import Image

# 从PyTorch Lightning导入随机种子设置函数
from pytorch_lightning import seed_everything
# 从PyTorch Lightning导入训练器类
from pytorch_lightning.trainer import Trainer
# 从PyTorch Lightning导入回调函数相关类
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
# 从PyTorch Lightning导入分布式训练相关功能
from pytorch_lightning.utilities.distributed import rank_zero_only
# 从PyTorch Lightning导入日志输出功能
from pytorch_lightning.utilities import rank_zero_info

# 从LDMB库导入基础数据集类
from ldm.data.base import Txt2ImgIterableBaseDataset
# 从LDMB库导入配置实例化工具函数
from ldm.util import instantiate_from_config


# 创建命令行参数解析器函数
def get_parser(**parser_kwargs):
    # 定义一个将字符串转换为布尔值的函数
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # 创建参数解析器对象
    parser = argparse.ArgumentParser(**parser_kwargs)
    # 添加name参数，用于日志目录的后缀
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    # 添加resume参数，用于从已有日志目录或检查点恢复训练
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    # 添加base参数，用于指定基础配置文件路径
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`." ,
        default=list(),
    )
    # 添加train参数，用于启用训练模式
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    # 添加no-test参数，用于禁用测试
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    # 添加project参数，用于指定新项目名称或现有项目路径
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    # 添加debug参数，用于启用调试模式
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    # 添加seed参数，用于设置随机种子
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    # 添加postfix参数，用于添加名称的后缀
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    # 添加logdir参数，用于指定日志目录
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    # 添加scale_lr参数，用于控制是否按GPU数量、批次大小等缩放学习率
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


# 获取与默认值不同的训练器参数
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # 向解析器添加训练器参数
    parser = Trainer.add_argparse_args(parser)
    # 获取默认参数
    args = parser.parse_args([])
    # 返回与默认值不同的参数名称列表
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# 数据集包装类，用于将任意具有__len__和__getitem__方法的对象包装成PyTorch数据集
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 数据加载器工作进程初始化函数
def worker_init_fn(_):
    # 获取工作进程信息
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    # 处理迭代式数据集的情况
    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # 重置记录数以保持可靠的长度信息
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


# 从配置创建数据模块的类
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        # 根据传入的配置设置不同阶段的数据加载器
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    # 准备数据的方法
    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    # 设置数据的方法
    def setup(self, stage=None):
        # 根据配置实例化各个数据集
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        # 如果需要包装，则用WrappedDataset包装每个数据集
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    # 创建训练数据加载器
    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        # 根据数据集类型决定是否打乱数据
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    # 创建验证数据加载器
    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    # 创建测试数据加载器
    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # 不对迭代式数据集进行打乱
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    # 创建预测数据加载器
    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


# 设置回调类，负责日志目录的创建和配置的保存
class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    # 处理键盘中断事件，保存检查点
    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # 在预训练例程开始时执行
    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # 创建日志目录并保存配置
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # 如果有训练步骤检查点回调，创建相应目录
            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            # 保存项目配置
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            # 保存Lightning配置
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint回调创建了日志目录，将其移除
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


# 图像日志回调类，用于记录和保存生成的图像
class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        # 定义不同日志器的图像记录方法
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        # 计算记录步骤
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    # 仅在rank 0进程执行
    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            # 创建图像网格
            grid = torchvision.utils.make_grid(images[k])
            # 将图像值从[-1,1]转换到[0,1]
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            # 添加图像到日志
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    # 仅在rank 0进程执行，将图像保存到本地
    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            # 转换图像维度顺序并转换为numpy数组
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            # 将图像值从[0,1]转换到[0,255]
            grid = (grid * 255).astype(np.uint8)
            # 创建文件名，包含全局步骤、当前轮次和批次索引
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # 保存图像
            Image.fromarray(grid).save(path)

    # 记录图像的主要方法
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # 根据配置确定检查索引
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        # 检查是否需要记录图像
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            # 如果在训练模式，则切换到评估模式
            if is_train:
                pl_module.eval()

            # 在不计算梯度的上下文中生成图像
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            # 限制记录的图像数量
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    # 将张量移至CPU并分离
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        # 将图像值限制在[-1,1]
                        images[k] = torch.clamp(images[k], -1., 1.)

            # 保存图像到本地和日志
            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            # 如果之前是训练模式，切换回训练模式
            if is_train:
                pl_module.train()

    # 检查是否达到记录频率
    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    # 在训练批次结束时执行
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    # 在验证批次结束时执行
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        # 如果模型有校准梯度范数的功能
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


# CUDA回调类，用于监控CUDA内存使用和训练时间
class CUDACallback(Callback):
    # 参考：https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    # 在训练轮次开始时执行
    def on_train_epoch_start(self, trainer, pl_module):
        # 重置内存使用计数器
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    # 在训练轮次结束时执行
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        # 计算最大内存使用量（MB）
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        # 计算轮次训练时间
        epoch_time = time.time() - self.start_time

        try:
            # 在分布式训练中聚合内存使用量和时间
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            # 输出平均轮次时间和最大内存使用量
            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


# 主程序入口
if __name__ == "__main__":
    # 自定义解析器用于指定配置文件、训练、测试和调试模式、后缀、恢复等
    # `--key value` 参数被解释为训练器的参数
    # `nested.key=value` 参数被解释为配置参数
    # 配置从左到右合并，后跟命令行参数

    # 配置格式示例：
    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (可选，有合理的默认值，可在命令行指定)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    # 获取当前时间并格式化
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # 添加当前工作目录到系统路径，以便在以`python main.py`运行时可以使用该文件中的类
    # （特别是`main.DataModuleFromConfig`）
    sys.path.append(os.getcwd())

    # 创建命令行参数解析器
    parser = get_parser()
    # 向解析器添加训练器参数
    parser = Trainer.add_argparse_args(parser)

    # 解析命令行参数
    opt, unknown = parser.parse_known_args()
    # 检查冲突的参数
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    # 处理恢复训练的情况
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        # 加载日志目录中的配置文件
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        # 确定日志目录名称
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    # 定义检查点目录和配置目录
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    # 设置随机种子
    seed_everything(opt.seed)

    try:
        # 初始化并保存配置
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # 合并训练器命令行参数和配置
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # 默认使用DDP加速
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # 实例化模型
        model = instantiate_from_config(config.model)

        # 训练器和回调函数
        trainer_kwargs = dict()

        # 默认日志器配置
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        # 合并默认日志器配置和用户指定的配置
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # 模型检查点配置
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        # 如果模型有monitor属性，使用它作为检查点指标
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        # 根据PyTorch Lightning版本决定如何添加检查点回调
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # 添加设置日志目录的回调
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        # 对于PyTorch Lightning 1.4.0及以上版本，以不同方式添加检查点回调
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        # 处理基于训练步骤的检查点
        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        # 合并回调配置
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        # 处理忽略键回调
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        # 实例化所有回调
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # 创建训练器
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # 实例化数据模块
        data = instantiate_from_config(config.data)
        # 根据PyTorch Lightning文档，虽然不应该需要手动调用这些方法，但实际上是必要的
        # Lightning仍然会处理正确的多进程
        data.prepare_data()
        data.setup()
        # 打印数据信息
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # 配置学习率
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            # 计算GPU数量
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        # 根据配置缩放学习率
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # 允许通过USR1信号触发检查点保存
        def melk(*args, **kwargs):
            # 运行所有检查点钩子
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        # 允许通过USR2信号进入调试模式
        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()

        # 导入信号处理模块
        import signal

        # 设置信号处理器
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # 运行训练和测试
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                # 发生异常时保存检查点
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        # 处理调试模式下的异常
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # 将新创建的调试项目移动到debug_runs目录
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        # 打印训练器性能分析摘要
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
