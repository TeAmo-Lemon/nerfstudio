# Phase 1 技术报告: Nerfstudio 3DGS 去框架化审计

审计日期: 2026-04-08

目标: 在不破坏 `Splatfacto + gsplat` 性能特征的前提下，将当前仓库裁剪为一个直接通过 `python train.py` 启动的 3D Gaussian Splatting 训练工具。

## 结论摘要

当前分支里，`3DGS` 的性能核心并不在 `ns-train`，而在下面这条链路:

`FullImageDatamanager -> SplatfactoModel.get_outputs() -> gsplat.rendering.rasterization() -> strategy.step_pre_backward()/step_post_backward()`

`ns-train` 真正承担的是 CLI 选择器和配置拼装职责，而不是高性能训练本身。换句话说:

- 可以切掉: `tyro` 子命令系统、`method_configs`/`dataparser_configs` 的动态 Union、`pyproject` 中的 `ns-train` 脚本入口。
- 不能动: `SplatfactoModel` 内部的 `rasterization()` 调用参数、densify/prune 的 callback 时序、`FullImageDatamanager` 的全图相机采样与图像缓存逻辑、viewer 的 `update_scene()` 触发节奏。

另外要先澄清一个事实: 这个仓库快照里并没有你描述的那种 runtime plugin scanner 或 `@register` Registry 体系。这里的“动态注册”主要是 `tyro.extras.subcommand_type_from_defaults(...)` 生成 CLI 子命令，而不是 decorator 式模型注册。

---

## 1. 当前训练主链路

### 1.1 CLI 到训练执行

`ns-train` 的入口是 [`nerfstudio/scripts/train.py`](nerfstudio/scripts/train.py) 第 269-278 行。它做的事情很有限:

1. 用 `tyro.cli(AnnotatedBaseConfigUnion)` 解析 CLI。
2. 将解析结果交给 `main(config)`。
3. `main()` 调用 `launch()`，再调用 `train_loop()`。
4. `train_loop()` 实例化 `Trainer`，然后 `trainer.setup(); trainer.train()`。

关键代码:

- `tyro` 入口: `nerfstudio/scripts/train.py:269-278`
- 单/多卡启动壳: `nerfstudio/scripts/train.py:161-266`

### 1.2 Trainer 到 Pipeline

`Trainer.setup()` 会实例化 `pipeline`、`optimizers`、viewer、callbacks、writer。`Trainer.train()` 再驱动主循环。

关键代码:

- `pipeline.setup(...)`: `nerfstudio/engine/trainer.py:158-164`
- viewer 启动: `nerfstudio/engine/trainer.py:167-196`
- callbacks 收集: `nerfstudio/engine/trainer.py:201-205`
- 主循环: `nerfstudio/engine/trainer.py:233-312`
- 单步反传与优化: `nerfstudio/engine/trainer.py:486-530`

### 1.3 Pipeline 到 DataManager / Model

`VanillaPipeline` 的职责是两件事:

1. 实例化 `FullImageDatamanager`
2. 从 dataparser metadata 中抽取 `points3D_xyz / points3D_rgb` 作为 Gaussian 初始化种子，再实例化 `SplatfactoModel`

关键代码:

- datamanager 初始化: `nerfstudio/pipelines/base_pipeline.py:254-256`
- 从 COLMAP 3D 点提取 `seed_pts`: `nerfstudio/pipelines/base_pipeline.py:257-275`
- 训练时只做 `next_train -> model -> get_metrics -> get_loss`: `nerfstudio/pipelines/base_pipeline.py:289-303`

这说明 `Pipeline` 主要是装配壳，不是算法核心。Phase 2 可以把这层内联进新的 `train.py`。

---

## 2. 性能优势来源: Splatfacto 如何配合 gsplat

## 2.1 高速训练的根因不是 CLI，而是全图高斯渲染路径

`Splatfacto` 不是 ray-based NeRF。它直接接收一个完整相机，而不是大批量 `RayBundle`。这点由 `FullImageDatamanager` 明确保证:

- 类说明写明“outputs cameras/images instead of raybundles”: `nerfstudio/data/datamanagers/full_images_datamanager.py:95-100`
- `next_train()` 返回 `(Cameras, Dict)`，而不是 rays: `nerfstudio/data/datamanagers/full_images_datamanager.py:386-412`

这带来两个直接收益:

- 省掉逐 ray 采样、拼装、ray marching 等 CPU/GPU 开销。
- 让渲染完全落在 `gsplat.rendering.rasterization()` 的高斯 splat CUDA 路径上。

## 2.2 gsplat 的真正热路径

核心渲染在 [`nerfstudio/models/splatfacto.py`](nerfstudio/models/splatfacto.py) 第 557-577 行:

- 输入: `means / quats / exp(scales) / sigmoid(opacities) / SH colors / viewmats / Ks`
- 输出: `render, alpha, self.info`
- `self.info` 会被 densify 策略复用

这一段是整个 3DGS 的性能底线。Phase 2 必须保持以下事实不变:

- 仍调用同一个 `gsplat.rendering.rasterization`
- 参数语义一致
- `self.info` 继续流向 `strategy.step_pre_backward()` 与 `strategy.step_post_backward()`
- 调用顺序不变

## 2.3 低显存占用来自几层共同约束

### A. 图像缓存是可控的，而且默认方法配置已经偏向省显存

`method_configs["splatfacto"]` 将 datamanager 设置为:

- `cache_images_type="uint8"`: `nerfstudio/configs/method_configs.py:53-57`
- dataparser 为 `ColmapDataParserConfig(load_3D_points=True)`: `nerfstudio/configs/method_configs.py:53-57`

这意味着训练图像不是默认以 `float32` 常驻缓存，先天更省内存。

### B. FullImageDatamanager 不会盲目把所有图像都放上 GPU

关键逻辑:

- 支持 `cpu/gpu/disk` 三种缓存: `nerfstudio/data/datamanagers/full_images_datamanager.py:66-72`
- 若训练集图片数大于 500，且配置为 `gpu`，自动降回 `cpu`: `nerfstudio/data/datamanagers/full_images_datamanager.py:139-145`
- `disk` 模式直接走 DataLoader 流式读取: `nerfstudio/data/datamanagers/full_images_datamanager.py:314-348`

这说明 Nerfstudio 的 3DGS 路线已经内建了“按数据规模退让显存”的保护。

### C. 训练期分辨率渐进提升

`SplatfactoModel` 并不是一开始就全分辨率训练:

- `num_downscales` 和 `resolution_schedule` 控制训练期降采样: `nerfstudio/models/splatfacto.py:93-100`, `434-447`
- GT 图和相机输出一起降采样: `nerfstudio/models/splatfacto.py:534-540`, `610-619`

这会显著降低前期显存和算力压力，也是 3DGS “快起飞” 的来源之一。

### D. 训练时默认不输出 depth

只有在 `output_depth_during_training=True` 或 eval 时才走 `RGB+ED`:

- `nerfstudio/models/splatfacto.py:546-549`

默认训练路径只渲染 `RGB`，避免多余通道带来的额外显存占用。

### E. 单相机全图训练，避免额外 batch 膨胀

训练阶段显式要求:

- `assert camera.shape[0] == 1`: `nerfstudio/models/splatfacto.py:501-503`

即每次一步只处理一个相机视图，配合高斯 rasterization 更稳定地控制峰值显存。

## 2.4 梯度控制与 densify/prune 是性能和质量共同关键

### A. densify 使用 gsplat 返回的 `info`

训练前向后立刻执行:

- `self.strategy.step_pre_backward(..., self.info)`: `nerfstudio/models/splatfacto.py:578-581`

反传结束后，训练 callback 再执行:

- `self.strategy.step_post_backward(...)`: `nerfstudio/models/splatfacto.py:366-387`

这意味着 densify/prune 并不是普通的“每 N 步做一次 Python 统计”，而是紧耦合 `gsplat` 渲染过程中产生的中间信息。

如果 Phase 2 改错这两个 callback 的顺序，最直接的后果不是“代码风格变化”，而是:

- densify 触发条件失真
- split/prune/reset alpha 失真
- 训练 FPS 和质量一起偏离原版

### B. 当前参数选择是性能优先而非激进稀疏梯度

调用 `rasterization()` 时固定:

- `packed=False`
- `sparse_grad=False`
- `absgrad=...`

见 `nerfstudio/models/splatfacto.py:567-574`

也就是说，这个实现并没有启用“packed/sparse-grad”那条更激进、也更脆弱的路径，而是走相对稳的 dense grad 路径。Phase 2 若想“优化”这里，反而更容易破坏可比性。

### C. 参数按语义拆组优化

`SplatfactoModel.get_param_groups()` 将高斯参数拆成:

- `means`
- `scales`
- `quats`
- `features_dc`
- `features_rest`
- 可选 `bilateral_grid`
- `camera_opt`

见 `nerfstudio/models/splatfacto.py:413-432`

对应优化器配置在 `method_configs["splatfacto"]` 中逐组指定学习率和 scheduler:

- `nerfstudio/configs/method_configs.py:59-96`

这不是框架噪音，而是 3DGS 收敛稳定性的组成部分。必须原样保留。

## 2.5 3DGS 启动快、收敛快的另一来源: COLMAP 点云种子

`ColmapDataParser` 会把 COLMAP 的 3D 点和颜色放到 metadata:

- 3D 点加载: `nerfstudio/data/dataparsers/colmap_dataparser.py:403-447`

`VanillaPipeline` 再把这些点拿来做 `seed_points`:

- `nerfstudio/pipelines/base_pipeline.py:257-275`

`SplatfactoModel.populate_modules()` 若拿到 `seed_points`，会直接用它初始化:

- `means`
- 近邻距离导出的初始 `scales`
- 颜色 SH / RGB 特征

见 `nerfstudio/models/splatfacto.py:189-239`

这部分不是“装配层”，是原版性能和收敛速度的组成部分。Phase 2 的手写 `train.py` 必须保留这段 seed 提取逻辑。

---

## 3. 去框架化影响评估

## 3.1 属于 `ns-train` 强依赖的部分

这些模块主要服务于 CLI 发现、配置选择、分发启动，不是 3DGS 算法核心。

### A. `tyro` CLI 与动态子命令 Union

- `nerfstudio/scripts/train.py:269-278`
- `nerfstudio/configs/method_configs.py:313-323`
- `nerfstudio/configs/dataparser_configs.py:26-44`

实质上:

- `method_configs.py` 把一组默认 `TrainerConfig` 暴露成 `splatfacto / splatfacto-big / splatfacto-mcmc / splatfacto-dino`
- `dataparser_configs.py` 把 dataparser 暴露成 `colmap` 子命令
- `tyro` 再把这些对象拼成 CLI

这套东西是去框架化的首要删减对象。

### B. `pyproject.toml` 中的命令入口

当前脚本入口:

- `ns-train`: `pyproject.toml:127-139`
- `ns-train-splatfacto-dino`: `pyproject.toml:127-139`

这些入口本身不会影响训练 FPS，但会保留一整套 CLI 结构、帮助文本和脚本耦合。

### C. `Trainer` / `Pipeline` 里的“装配壳”

以下逻辑可以内联到新的 `train.py`:

- `Trainer.setup()` 的 pipeline/viewer/optimizer 构造: `nerfstudio/engine/trainer.py:158-221`
- `Trainer.train()` 主循环: `nerfstudio/engine/trainer.py:233-312`
- `Trainer.train_iteration()`: `nerfstudio/engine/trainer.py:486-530`
- `VanillaPipeline.__init__()` 的 datamanager/model 装配: `nerfstudio/pipelines/base_pipeline.py:242-282`

它们不是无用代码，但属于“框架封装层”，符合“保持结构、切除封装”的裁剪目标。

## 3.2 属于 3DGS 核心逻辑的部分

这些模块应该保留，最多只允许“挪位置”，不允许“重写算法”。

### 必保留

- `nerfstudio/models/splatfacto.py`
  - 高斯参数初始化
  - `gsplat.rendering.rasterization`
  - `get_loss_dict`
  - `step_cb`
  - `step_post_backward`
- `nerfstudio/data/datamanagers/full_images_datamanager.py`
  - 全图图像缓存
  - 单相机采样
  - `cam_idx` metadata 注入
- `nerfstudio/data/dataparsers/colmap_dataparser.py`
  - COLMAP 相机解析
  - pose 归一化
  - 3D 点载入
- `nerfstudio/engine/optimizers.py`
  - 参数组优化与 scheduler
- `nerfstudio/engine/callbacks.py`
  - callback 时序
- `nerfstudio/viewer/viewer.py`
  - `Viewer(...)`
  - `init_scene()`
  - `update_scene()`
  - `training_complete()`
- `nerfstudio/cameras/*`
  - `Cameras`
  - `CameraOptimizer`
- `nerfstudio/configs/base_config.py`
  - dataclass 风格 config 和 `InstantiateConfig.setup()`

### 需要谨慎保留

- `nerfstudio/engine/trainer.py`
  - Phase 2 可以不再作为主入口使用
  - 但其中主循环、checkpoint、viewer 更新顺序应被逐行对照迁移
- `nerfstudio/pipelines/base_pipeline.py`
  - 可以去掉类壳
  - 但 `seed_points` 提取逻辑必须保留

## 3.3 当前分支里“不存在”的东西

针对你的目标，需要明确一点，避免 Phase 2 走偏:

- 没有发现 `@register` decorator
- 没有发现模型类上的 `@deprecated` 装饰器
- 没有发现训练路径里的运行时 entry-point/plugin 扫描

本次检索结果:

- `rg "@deprecated|@register"` 在 `nerfstudio/models nerfstudio/data nerfstudio/configs nerfstudio/scripts` 下无命中
- `train` 主链上未看到 `importlib.metadata.entry_points()` 或类似 plugin discovery

所以在这个代码快照里，“删除 Registry” 应理解为:

1. 删除 `method_configs` 这种方法名到默认配置对象的映射层
2. 删除 `dataparser_configs` 这种 CLI 子命令发现层
3. 删除 `tyro` 基于 Union 的命令发现逻辑

而不是去寻找并清除一个实际存在的 decorator registry。

---

## 4. 清理计划

## 4.1 建议直接删除的非核心文件/机制

### 第一批: 训练 CLI 壳

1. `nerfstudio/scripts/train.py`
   - 原因: 完全围绕 `tyro + TrainerConfig` 展开，Phase 2 将被新的根级 `train.py` 取代。
2. `nerfstudio/scripts/train_splatfacto_dino.py`
   - 原因: 只是给 `ns-train` 预填一个子命令前缀。
3. `pyproject.toml` 中以下入口:
   - `ns-train`
   - `ns-train-splatfacto-dino`

### 第二批: 动态配置发现层

4. `nerfstudio/configs/method_configs.py`
   - 原因: 仅服务于 CLI 选择 `splatfacto`/`splatfacto-big`/`splatfacto-mcmc` 等 profile。
5. `nerfstudio/configs/dataparser_configs.py`
   - 原因: 仅服务于 CLI 子命令 dataparser 发现。

### 第三批: 文档/测试中的旧入口引用

6. `README.md` 里所有 `ns-train` 用法
7. `tests/test_splatfacto_integration.py`
   - 原因: 目前是围绕 `ns-download-data / ns-train / ns-eval` 的端到端脚本测试

## 4.2 建议改为“绕过使用”，而不是立刻物理删除

这批文件在 Phase 2 初版建议先保留，待 `python train.py` 跑通并验证性能后，再决定是否物理删除:

1. `nerfstudio/engine/trainer.py`
2. `nerfstudio/pipelines/base_pipeline.py`
3. `nerfstudio/utils/eval_utils.py`

原因:

- 它们提供了现成的行为对照基线。
- 其中包含 checkpoint、viewer、eval 加载等边缘逻辑。
- 先“停用”比直接“删掉”更安全。

## 4.3 必须保留的文件

### 算法核心

- `nerfstudio/models/splatfacto.py`
- `nerfstudio/data/datamanagers/full_images_datamanager.py`
- `nerfstudio/data/dataparsers/colmap_dataparser.py`
- `nerfstudio/engine/optimizers.py`
- `nerfstudio/engine/callbacks.py`

### 支撑结构

- `nerfstudio/configs/base_config.py`
- `nerfstudio/configs/experiment_config.py` 或其简化版
- `nerfstudio/cameras/*`
- `nerfstudio/data/datasets/*`
- `nerfstudio/viewer/*`
- `nerfstudio/utils/*` 中被上述模块直接依赖的部分

---

## 5. Phase 2 的最小实现建议

## 5.1 推荐的新 `train.py` 结构

新的根级 `train.py` 应直接做下面几件事:

1. 解析最少量参数
   - `-s/--source-path`
   - `-m/--model-path`
   - 可选: `--max-steps`, `--vis`, `--data-device`, `--load-checkpoint`
2. 直接实例化一份集中配置对象
   - 继续沿用 dataclass 风格即可
   - 不再通过 `tyro` 子命令发现
3. 直接实例化:
   - `ColmapDataParserConfig`
   - `FullImageDatamanagerConfig`
   - `SplatfactoModelConfig`
   - `Optimizers`
4. 手动从 dataparser metadata 提取 `points3D_xyz / points3D_rgb`
5. 手动拼出训练 callbacks
6. 手动写训练循环
7. 手动启动 `Viewer` 并每步调用 `update_scene(step, num_rays_per_batch)`

## 5.2 必须原样迁移的训练顺序

下面的顺序不能改:

1. `model.train()`
2. 先执行 `BEFORE_TRAIN_ITERATION` callbacks
3. `camera, batch = datamanager.next_train(step)`
4. `outputs = model(camera)`
5. `metrics_dict = model.get_metrics_dict(outputs, batch)`
6. `loss_dict = model.get_loss_dict(outputs, batch, metrics_dict)`
7. `loss.backward()`
8. `optimizer.step()`
9. 再执行 `AFTER_TRAIN_ITERATION` callbacks
10. `viewer.update_scene(step, datamanager.get_train_rays_per_batch())`

特别注意:

- `step_cb()` 必须发生在前向之前，否则 `self.step / self.optimizers / self.schedulers` 未更新
- `step_post_backward()` 必须发生在 backward/optimizer 之后，否则 densify/prune 状态错位

## 5.3 可以简化但不建议第一版动的点

- DDP 多卡启动
- eval 流程
- tensorboard / wandb / comet
- legacy viewer
- `splatfacto-big` / `mcmc` / `dino` 多 profile

建议第一版只保留:

- 单机单卡
- COLMAP 数据
- 标准 `Splatfacto`
- 新 viewer
- checkpoint 保存/加载

---

## 6. 风险清单

## 高风险

1. 漏掉 `seed_points` 传递
   - 结果: 初始高斯从随机点开始，收敛速度和质量会明显偏离原版。
2. 打乱 callback 顺序
   - 结果: densify/prune/reset alpha 行为错位，FPS 和显存占用都可能失真。
3. 修改 `rasterization()` 参数
   - 结果: 性能不再可比，尤其是 `packed/sparse_grad/absgrad/render_mode`。
4. 忽略 `cam_idx` metadata
   - 结果: bilateral grid 路径失效。

## 中风险

5. 不保留 `uint8` 图像缓存默认值
   - 结果: 内存占用上升。
6. 不保留前期降分辨率训练
   - 结果: 启动更慢，显存更高。
7. viewer 更新节奏变化太大
   - 结果: 训练线程阻塞或实时性下降。

## 低风险

8. 删除 `tyro.conf.Suppress` 注解
   - 结果: 对训练性能无影响，只影响 CLI 暴露行为。

---

## 7. 建议的 Phase 2 执行顺序

1. 先新增根级 `train.py`，直接跑通单卡 `Splatfacto + Colmap + Viewer`
2. 对照 `Trainer.train_iteration()` 迁移 loss/backward/optimizer/scheduler/callback 顺序
3. 对照 `VanillaPipeline.__init__()` 补齐 `seed_points` 注入
4. 跑通你给出的测试命令
5. 再删除 `ns-train` 和 `method_configs` 等旧壳
6. 最后清理 README / tests / pyproject 入口

这样做的原因很简单: 先建立一个“性能等价的新入口”，再物理删壳，风险最小。

---

## 审计结论

这次“去框架化”是可行的，而且适合做成“保留算法模块，切掉 CLI 装配层”的精简版本。

真正需要守住的不是 `ns-train`，而是下面四个点:

1. `FullImageDatamanager` 的全图相机训练模式
2. `SplatfactoModel.get_outputs()` 中原样的 `gsplat.rasterization()` 调用
3. `step_pre_backward / step_post_backward` 的时序
4. `COLMAP 3D points -> seed_points` 的初始化链路

只要这四点不动，Phase 2 完全可以把外围封装削到只剩一个透明的 `python train.py`。
