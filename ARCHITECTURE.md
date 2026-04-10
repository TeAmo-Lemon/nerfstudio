# Nerfstudio 项目架构文档

## 目录
1. [项目概述](#项目概述)
2. [核心架构](#核心架构)
3. [配置系统](#配置系统)
4. [接口与指令](#接口与指令)
5. [数据流向](#数据流向)
6. [自定义开发指南](#自定义开发指南)

---

## 项目概述

Nerfstudio 是一个模块化的神经辐射场（NeRF）开发框架，主要用于神经渲染和3D重建。当前版本主要实现了 **Gaussian Splatting (Splatfacto)** 方法。

### 核心特性
- 模块化设计：模型、数据管理器、管道分离
- 灵活的配置系统：基于dataclass的配置管理
- 多种可视化支持：Web Viewer、Tensorboard、WandB、Comet
- 完整的训练流程：支持checkpoint、恢复训练、评估

---

## 核心架构

### 1. 目录结构

```
nerfstudio/
├── train.py                    # 简化的训练入口点（Splatfacto专用）
├── nerfstudio/
│   ├── models/                 # 模型定义
│   │   ├── base_model.py      # 模型基类
│   │   └── splatfacto.py      # Gaussian Splatting实现
│   ├── pipelines/              # 管道定义
│   │   ├── base_pipeline.py   # 管道基类
│   │   └── dynamic_batch.py   # 动态批处理
│   ├── engine/                 # 训练引擎
│   │   ├── trainer.py         # 训练器
│   │   ├── optimizers.py      # 优化器
│   │   ├── schedulers.py      # 学习率调度器
│   │   └── callbacks.py       # 训练回调
│   ├── data/                   # 数据处理
│   │   ├── datamanagers/      # 数据管理器
│   │   │   ├── base_datamanager.py
│   │   │   └── full_images_datamanager.py
│   │   ├── dataparsers/       # 数据解析器
│   │   │   ├── base_dataparser.py
│   │   │   └── colmap_dataparser.py
│   │   └── datasets/          # 数据集
│   ├── configs/                # 配置系统
│   │   ├── base_config.py     # 基础配置
│   │   └── experiment_config.py
│   ├── model_components/       # 模型组件
│   │   ├── losses.py          # 损失函数
│   │   ├── renderers.py       # 渲染器
│   │   └── ray_samplers.py    # 射线采样器
│   ├── cameras/                # 相机模型
│   ├── viewer/                 # 可视化工具
│   └── scripts/                # 脚本工具
│       ├── render.py
│       ├── eval.py
│       └── exporter.py
└── pyproject.toml              # 项目配置
```

### 2. 核心模块关系

```
┌─────────────────────────────────────────────────────────────┐
│                         train.py                             │
│                    (训练入口点)                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      TrainerConfig                           │
│                    (训练配置)                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      DirectPipeline                          │
│                   (管道包装器)                                │
│  ┌──────────────────────┬───────────────────────────────┐  │
│  │   DataManager        │         Model                 │  │
│  │  (数据管理器)         │       (模型)                  │  │
│  │  - 加载数据           │  - 前向传播                   │  │
│  │  - 数据预处理         │  - 计算loss                   │  │
│  │  - 批处理             │  - 优化参数                   │  │
│  └──────────────────────┴───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Optimizers                             │
│                    (优化器管理)                               │
└─────────────────────────────────────────────────────────────┘
```

### 3. 关键类说明

#### 3.1 Model（模型基类）
位置：[nerfstudio/models/base_model.py](nerfstudio/models/base_model.py)

**核心方法：**
- `populate_modules()`: 初始化模型组件
- `get_outputs(ray_bundle)`: 前向传播，返回渲染结果
- `get_loss_dict(outputs, batch, metrics_dict)`: **计算损失字典（关键）**
- `get_metrics_dict(outputs, batch)`: 计算评估指标
- `get_param_groups()`: 返回优化器参数组

#### 3.2 DataManager（数据管理器）
位置：[nerfstudio/data/datamanagers/full_images_datamanager.py](nerfstudio/data/datamanagers/full_images_datamanager.py)

**核心功能：**
- 加载和解析数据集
- 管理训练/验证数据
- 提供数据迭代器

#### 3.3 Pipeline（管道）
位置：[nerfstudio/pipelines/base_pipeline.py](nerfstudio/pipelines/base_pipeline.py)

**核心功能：**
- 连接DataManager和Model
- 提供统一的训练/评估接口
- 管理模型状态

---

## 配置系统

### 1. 配置层次结构

```python
TrainerConfig                    # 训练器配置
├── pipeline: VanillaPipelineConfig
│   ├── datamanager: FullImageDatamanagerConfig
│   │   └── dataparser: ColmapDataParserConfig
│   └── model: SplatfactoModelConfig
├── optimizers: Dict[str, OptimizerConfig]
├── viewer: ViewerConfig
├── machine: MachineConfig
└── logging: LoggingConfig
```

### 2. 配置示例

```python
# train.py 中的配置构建
config = TrainerConfig(
    method_name="splatfacto",
    experiment_name=args.model_path.name,
    output_dir=args.model_path.parent,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=ColmapDataParserConfig(data=args.source_path),
            cache_images="gpu",
        ),
        model=SplatfactoModelConfig(),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6),
        },
        # ... 其他参数组
    },
)
```

### 3. 配置文件格式

配置会保存为YAML格式：
```yaml
# outputs/poster/config.yml
method_name: splatfacto
experiment_name: poster
max_num_iterations: 30000
pipeline:
  datamanager:
    dataparser:
      data: data/nerfstudio/poster
  model:
    ssim_lambda: 0.2
    sh_degree: 3
```

---

## 接口与指令

### 1. 主要CLI命令

#### 训练命令
```bash
# 基本训练
python train.py -s data/nerfstudio/poster -m outputs/poster

# 完整参数
python train.py \
    -s /path/to/dataset \           # 数据集路径
    -m /path/to/output \            # 输出路径
    --max-steps 30000 \             # 最大训练步数
    --save-every 2000 \             # 保存间隔
    --eval-image-every 100 \        # 评估间隔
    --mixed-precision \             # 混合精度
    --vis viewer \                  # 可视化方式
    --device cuda                   # 设备
```

#### 恢复训练
```bash
# 从最新checkpoint恢复
python train.py -s data/nerfstudio/poster -m outputs/poster --resume

# 从指定checkpoint恢复
python train.py -s data/nerfstudio/poster -m outputs/poster --load-checkpoint outputs/poster/nerfstudio_models/step-000010000.ckpt
```

#### 可视化命令
```bash
# 启动viewer
ns-viewer --load-config outputs/poster/config.yml
```

#### 渲染命令
```bash
# 渲染视频
ns-render --load-config outputs/poster/config.yml \
    --traj filename \
    --output-path renders/video.mp4
```

#### 评估命令
```bash
# 评估模型
ns-eval --load-config outputs/poster/config.yml
```

#### 导出命令
```bash
# 导出点云
ns-export pointcloud --load-config outputs/poster/config.yml
```

### 2. 数据处理命令

```bash
# 处理自定义数据
ns-process-data images \
    --data /path/to/images \
    --output-dir /path/to/output \
    --colmap-model sparse
```

---

## 数据流向

### 1. 训练数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 数据加载阶段                                              │
└─────────────────────────────────────────────────────────────┘
   ColmapDataParser
        │ 解析COLMAP数据
        ▼
   DataparserOutputs
        │ 包含相机参数、图像路径、点云等
        ▼
   InputDataset
        │ 创建数据集对象
        ▼
   FullImageDatamanager
        │ 管理数据加载、缓存
        ▼
   next_train(step)
        │ 返回相机和图像数据
        ▼
   batch = {
       "image": Tensor[H, W, 3],
       "camera": Cameras,
   }

┌─────────────────────────────────────────────────────────────┐
│ 2. 模型前向传播阶段                                          │
└─────────────────────────────────────────────────────────────┘
   SplatfactoModel.forward(camera)
        │
        ├─> camera_optimizer.apply_to_camera()
        │   (相机优化)
        │
        ├─> rasterization()
        │   (高斯光栅化)
        │
        └─> outputs = {
            "rgb": Tensor[H, W, 3],
            "depth": Tensor[H, W, 1],
            "accumulation": Tensor[H, W, 1],
        }

┌─────────────────────────────────────────────────────────────┐
│ 3. 损失计算阶段                                              │
└─────────────────────────────────────────────────────────────┘
   get_metrics_dict(outputs, batch)
        │ 计算PSNR等指标
        ▼
   metrics_dict = {"psnr": float, "gaussian_count": int}
        │
        ▼
   get_loss_dict(outputs, batch, metrics_dict)
        │ 计算损失
        ├─> L1 loss
        ├─> SSIM loss
        ├─> scale regularization
        └─> 其他损失
        ▼
   loss_dict = {
       "main_loss": Tensor,
       "scale_reg": Tensor,
   }

┌─────────────────────────────────────────────────────────────┐
│ 4. 优化阶段                                                  │
└─────────────────────────────────────────────────────────────┘
   loss = sum(loss_dict.values())
        │
        ▼
   loss.backward()
        │
        ▼
   optimizer.step()
        │
        ▼
   scheduler.step()
        │
        ▼
   strategy.step_post_backward()
        (高斯密度化/剪枝)
```

### 2. 评估数据流

```
FullImageDatamanager.next_eval(step)
        │
        ▼
   Model.get_outputs_for_camera(camera)
        │
        ▼
   Model.get_image_metrics_and_images(outputs, batch)
        │
        ▼
   metrics = {
       "psnr": float,
       "ssim": float,
       "lpips": float,
   }
```

### 3. 关键数据结构

#### Cameras
```python
Cameras(
    camera_to_worlds: Float[Tensor, "num_cameras 3 4"],
    fx: Float[Tensor, "num_cameras"],
    fy: Float[Tensor, "num_cameras"],
    cx: Float[Tensor, "num_cameras"],
    cy: Float[Tensor, "num_cameras"],
    width: Int[Tensor, "num_cameras"],
    height: Int[Tensor, "num_cameras"],
)
```

#### Batch
```python
batch = {
    "image": Float[Tensor, "H W 3"],      # RGB图像 [0, 1]
    "camera": Cameras,                     # 相机对象
    "mask": Optional[Float[Tensor, "H W 1"]],  # 可选掩码
}
```

#### Outputs
```python
outputs = {
    "rgb": Float[Tensor, "H W 3"],        # 渲染的RGB图像
    "depth": Float[Tensor, "H W 1"],      # 深度图
    "accumulation": Float[Tensor, "H W 1"], # 累积权重
    "background": Float[Tensor, "3"],      # 背景颜色
}
```

---

## 自定义开发指南

### 1. 如何添加自定义Loss

#### 方法一：在模型中直接添加（推荐）

**位置：** 继承或修改 `SplatfactoModel` 的 `get_loss_dict` 方法

**示例：** [nerfstudio/models/splatfacto.py](nerfstudio/models/splatfacto.py#L653-L712)

```python
# 在 nerfstudio/models/splatfacto.py 中
class SplatfactoModel(Model):
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """计算损失字典"""
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), 
            outputs["background"]
        )
        pred_img = outputs["rgb"]
        
        # 原有的loss
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img, pred_img)
        
        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + 
                         self.config.ssim_lambda * simloss,
        }
        
        # ===== 添加自定义loss =====
        # 示例1: 添加深度一致性loss
        if "depth" in outputs and "depth_gt" in batch:
            depth_loss = torch.abs(
                outputs["depth"] - batch["depth_gt"]
            ).mean()
            loss_dict["depth_loss"] = 0.1 * depth_loss
        
        # 示例2: 添加感知loss
        if self.training:
            perceptual_loss = self.compute_perceptual_loss(pred_img, gt_img)
            loss_dict["perceptual_loss"] = 0.05 * perceptual_loss
        
        # 示例3: 添加正则化loss
        if self.config.use_custom_regularization:
            reg_loss = self.compute_custom_regularization()
            loss_dict["custom_reg"] = reg_loss
        
        return loss_dict
    
    def compute_perceptual_loss(self, pred, gt):
        """计算感知损失"""
        # 实现你的感知损失
        pass
    
    def compute_custom_regularization(self):
        """计算自定义正则化"""
        # 实现你的正则化
        pass
```

#### 方法二：在losses.py中定义新loss函数

**位置：** [nerfstudio/model_components/losses.py](nerfstudio/model_components/losses.py)

```python
# 在 nerfstudio/model_components/losses.py 中添加
def my_custom_loss(
    pred: Float[Tensor, "*batch 3"],
    gt: Float[Tensor, "*batch 3"],
    weight: Optional[Float[Tensor, "*batch"]] = None,
) -> Float[Tensor, "0"]:
    """
    自定义损失函数
    
    Args:
        pred: 预测值
        gt: 真实值
        weight: 可选权重
    
    Returns:
        损失值
    """
    loss = torch.abs(pred - gt)
    if weight is not None:
        loss = loss * weight
    return loss.mean()

# 然后在模型的 get_loss_dict 中使用
from nerfstudio.model_components.losses import my_custom_loss

class MyModel(SplatfactoModel):
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        # 使用自定义loss
        custom_loss = my_custom_loss(
            outputs["rgb"], 
            batch["image"].float() / 255.0
        )
        loss_dict["custom_loss"] = custom_loss
        
        return loss_dict
```

#### 方法三：创建新的模型类

**步骤：**

1. 创建新模型文件 `nerfstudio/models/my_model.py`:

```python
from dataclasses import dataclass, field
from typing import Type
from nerfstudio.models.base_model import Model, ModelConfig

@dataclass
class MyModelConfig(ModelConfig):
    """自定义模型配置"""
    _target: Type = field(default_factory=lambda: MyModel)
    custom_loss_weight: float = 0.1
    """自定义loss权重"""

class MyModel(Model):
    """自定义模型"""
    
    config: MyModelConfig
    
    def populate_modules(self):
        """初始化模型组件"""
        super().populate_modules()
        # 初始化你的组件
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """返回参数组"""
        param_groups = {}
        # 添加你的参数
        return param_groups
    
    def get_outputs(self, ray_bundle):
        """前向传播"""
        outputs = {}
        # 实现你的前向传播
        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """计算损失"""
        loss_dict = {}
        # 实现你的损失计算
        return loss_dict
    
    def get_image_metrics_and_images(self, outputs, batch):
        """计算评估指标"""
        metrics = {}
        images = {}
        return metrics, images
```

2. 在 `nerfstudio/models/__init__.py` 中注册：

```python
from nerfstudio.models.my_model import MyModel, MyModelConfig
```

3. 在配置中使用：

```python
from nerfstudio.models.my_model import MyModelConfig

config = TrainerConfig(
    pipeline=VanillaPipelineConfig(
        model=MyModelConfig(
            custom_loss_weight=0.2,
        ),
    ),
)
```

### 2. 如何自定义Train

#### 方法一：修改train.py（简单修改）

**位置：** [train.py](train.py)

**可修改的部分：**

1. **修改训练循环** - 在 `_run_train_loop` 函数中：

```python
def _run_train_loop(config, pipeline, optimizers, ...):
    for step in range(start_step, config.max_num_iterations):
        # ===== 自定义训练逻辑 =====
        
        # 1. 自定义数据采样
        camera, batch = custom_data_sampling(pipeline.datamanager, step)
        
        # 2. 自定义前向传播
        with torch.autocast(device_type=autocast_device, enabled=config.mixed_precision):
            outputs = custom_forward(pipeline.model, camera, batch)
            metrics_dict = pipeline.model.get_metrics_dict(outputs, batch)
            loss_dict = pipeline.model.get_loss_dict(outputs, batch, metrics_dict)
            
            # 添加自定义loss
            custom_loss = compute_custom_loss(outputs, batch)
            loss_dict["custom"] = custom_loss
            
            loss = functools.reduce(torch.add, loss_dict.values())
        
        # 3. 自定义优化步骤
        custom_backward_step(loss, optimizers, grad_scaler)
        
        # 4. 自定义回调
        if step % 100 == 0:
            run_custom_evaluation(pipeline.model)
```

2. **修改优化器配置** - 在 `_build_config` 函数中：

```python
def _build_config(args):
    config = TrainerConfig(
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6),
            },
            # 添加自定义参数组
            "custom_params": {
                "optimizer": AdamOptimizerConfig(lr=1e-3),
                "scheduler": None,
            },
        },
    )
    return config
```

3. **添加自定义回调** - 在 `_setup_pipeline` 函数中：

```python
def _setup_pipeline(config, device, grad_scaler):
    pipeline, optimizers, callbacks = ...
    
    # 添加自定义回调
    from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation
    
    callbacks.append(
        TrainingCallback(
            where=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
            func=my_custom_callback,
            args=[optimizers],
        )
    )
    
    return pipeline, optimizers, callbacks

def my_custom_callback(optimizers, step):
    """自定义回调函数"""
    if step % 1000 == 0:
        print(f"Custom callback at step {step}")
        # 执行自定义操作
```

#### 方法二：使用Trainer类（完整自定义）

**位置：** [nerfstudio/engine/trainer.py](nerfstudio/engine/trainer.py)

**创建自定义训练脚本：**

```python
#!/usr/bin/env python
"""自定义训练脚本"""

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

class CustomTrainer(Trainer):
    """自定义训练器"""
    
    def custom_training_step(self, step: int):
        """自定义训练步骤"""
        # 获取数据
        ray_bundle, batch = self.pipeline.datamanager.next_train(step)
        
        # 自定义前向传播
        model_outputs = self.pipeline.model(ray_bundle, batch)
        
        # 自定义损失计算
        metrics_dict = self.pipeline.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.pipeline.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
        # 添加自定义损失
        custom_loss = self.compute_custom_loss(model_outputs, batch)
        loss_dict["custom"] = custom_loss
        
        loss = sum(loss_dict.values())
        
        # 反向传播
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizers)
        self.grad_scaler.update()
        
        return loss, loss_dict, metrics_dict
    
    def compute_custom_loss(self, outputs, batch):
        """计算自定义损失"""
        # 实现你的损失计算
        return torch.tensor(0.0)
    
    def train(self):
        """自定义训练循环"""
        for step in range(self._start_step, self.config.max_num_iterations):
            # 自定义训练逻辑
            loss, loss_dict, metrics_dict = self.custom_training_step(step)
            
            # 自定义日志记录
            if step % self.config.logging.steps_per_log == 0:
                self.log_metrics(loss_dict, metrics_dict, step)
            
            # 自定义评估
            if step % self.config.steps_per_eval_image == 0:
                self.custom_eval(step)
    
    def custom_eval(self, step: int):
        """自定义评估"""
        # 实现你的评估逻辑
        pass

def main():
    # 创建配置
    config = TrainerConfig(
        method_name="custom",
        experiment_name="my_experiment",
        max_num_iterations=30000,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=ColmapDataParserConfig(data="data/nerfstudio/poster"),
            ),
            model=SplatfactoModelConfig(),
        ),
    )
    
    # 创建训练器
    trainer = CustomTrainer(config)
    trainer.setup()
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
```

#### 方法三：创建新的Pipeline

**位置：** [nerfstudio/pipelines/base_pipeline.py](nerfstudio/pipelines/base_pipeline.py)

```python
from nerfstudio.pipelines.base_pipeline import Pipeline

class CustomPipeline(Pipeline):
    """自定义管道"""
    
    def get_train_loss_dict(self, step: int):
        """自定义训练损失计算"""
        # 自定义数据获取
        ray_bundle, batch = self.custom_data_sampling(step)
        
        # 自定义前向传播
        model_outputs = self.model(ray_bundle, batch)
        
        # 自定义损失计算
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
        # 添加管道级别的损失
        pipeline_loss = self.compute_pipeline_loss(model_outputs, batch)
        loss_dict["pipeline_loss"] = pipeline_loss
        
        return model_outputs, loss_dict, metrics_dict
    
    def custom_data_sampling(self, step: int):
        """自定义数据采样"""
        # 实现你的采样逻辑
        return self.datamanager.next_train(step)
    
    def compute_pipeline_loss(self, outputs, batch):
        """计算管道级别的损失"""
        # 实现你的损失计算
        return torch.tensor(0.0)
```

### 3. 完整的自定义开发流程示例

#### 示例：添加深度监督的Gaussian Splatting

**步骤1：创建自定义模型**

```python
# nerfstudio/models/depth_splatfacto.py
from dataclasses import dataclass, field
from typing import Type
import torch
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

@dataclass
class DepthSplatfactoModelConfig(SplatfactoModelConfig):
    """带深度监督的Splatfacto配置"""
    _target: Type = field(default_factory=lambda: DepthSplatfactoModel)
    depth_loss_weight: float = 0.1
    """深度损失权重"""
    use_depth_prior: bool = True
    """是否使用深度先验"""

class DepthSplatfactoModel(SplatfactoModel):
    """带深度监督的Splatfacto模型"""
    
    config: DepthSplatfactoModelConfig
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """计算损失，添加深度损失"""
        # 调用父类的损失计算
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        # 添加深度损失
        if self.config.use_depth_prior and "depth_prior" in batch:
            depth_loss = self.compute_depth_loss(
                outputs["depth"], 
                batch["depth_prior"]
            )
            loss_dict["depth_loss"] = self.config.depth_loss_weight * depth_loss
        
        return loss_dict
    
    def compute_depth_loss(self, pred_depth, gt_depth):
        """计算深度损失"""
        # 归一化深度
        pred_normalized = pred_depth / (pred_depth.max() + 1e-8)
        gt_normalized = gt_depth / (gt_depth.max() + 1e-8)
        
        # L1损失
        loss = torch.abs(pred_normalized - gt_normalized).mean()
        return loss
```

**步骤2：创建自定义数据解析器**

```python
# nerfstudio/data/dataparsers/depth_colmap_dataparser.py
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

@dataclass
class DepthColmapDataParserConfig(ColmapDataParserConfig):
    """支持深度的COLMAP数据解析器配置"""
    depth_prior_path: Optional[Path] = None
    """深度先验路径"""

class DepthColmapDataParser(ColmapDataParser):
    """支持深度的COLMAP数据解析器"""
    
    def _get_dataparser_outputs(self, split="train"):
        # 调用父类方法获取基础输出
        dataparser_outputs = super()._get_dataparser_outputs(split)
        
        # 加载深度先验
        if self.config.depth_prior_path is not None:
            depth_priors = self.load_depth_priors()
            dataparser_outputs.metadata["depth_prior"] = depth_priors
        
        return dataparser_outputs
    
    def load_depth_priors(self):
        """加载深度先验"""
        # 实现深度加载逻辑
        pass
```

**步骤3：创建自定义训练脚本**

```python
# train_depth_splatfacto.py
#!/usr/bin/env python
from pathlib import Path
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.models.depth_splatfacto import DepthSplatfactoModelConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.depth_colmap_dataparser import DepthColmapDataParserConfig

def main():
    # 创建配置
    config = TrainerConfig(
        method_name="depth_splatfacto",
        experiment_name="depth_experiment",
        max_num_iterations=30000,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=DepthColmapDataParserConfig(
                    data=Path("data/nerfstudio/poster"),
                    depth_prior_path=Path("data/nerfstudio/poster/depth_priors"),
                ),
            ),
            model=DepthSplatfactoModelConfig(
                depth_loss_weight=0.1,
                use_depth_prior=True,
            ),
        ),
    )
    
    # 保存配置
    config.save_config()
    
    # 运行训练
    trainer = config.setup()
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    main()
```

**步骤4：运行训练**

```bash
python train_depth_splatfacto.py
```

---

## 总结

### 关键文件位置速查

| 功能 | 文件位置 |
|------|---------|
| 训练入口 | [train.py](train.py) |
| 模型基类 | [nerfstudio/models/base_model.py](nerfstudio/models/base_model.py) |
| Splatfacto模型 | [nerfstudio/models/splatfacto.py](nerfstudio/models/splatfacto.py) |
| 损失函数库 | [nerfstudio/model_components/losses.py](nerfstudio/model_components/losses.py) |
| 训练器 | [nerfstudio/engine/trainer.py](nerfstudio/engine/trainer.py) |
| 数据管理器 | [nerfstudio/data/datamanagers/full_images_datamanager.py](nerfstudio/data/datamanagers/full_images_datamanager.py) |
| 配置系统 | [nerfstudio/configs/base_config.py](nerfstudio/configs/base_config.py) |
| 管道基类 | [nerfstudio/pipelines/base_pipeline.py](nerfstudio/pipelines/base_pipeline.py) |

### 自定义开发要点

1. **添加自定义Loss：**
   - 在模型的 `get_loss_dict()` 方法中添加
   - 或在 `model_components/losses.py` 中定义新函数

2. **自定义Train：**
   - 简单修改：直接修改 `train.py`
   - 完整自定义：继承 `Trainer` 类
   - 管道级别：继承 `Pipeline` 类

3. **创建新模型：**
   - 继承 `Model` 基类
   - 实现必要的方法
   - 在配置中使用

4. **数据处理：**
   - 继承 `DataManager` 或 `DataParser`
   - 添加自定义数据字段

### 最佳实践

1. **保持模块化：** 尽量通过继承和配置来扩展，而不是直接修改源码
2. **使用配置系统：** 所有参数都应通过配置类管理
3. **遵循命名规范：** 参考现有代码的命名方式
4. **添加文档：** 为自定义组件添加docstring
5. **测试：** 编写单元测试验证功能

---

## 参考资料

- [Nerfstudio官方文档](https://docs.nerf.studio/)
- [Gaussian Splatting论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [项目GitHub](https://github.com/nerfstudio-project/nerfstudio)
