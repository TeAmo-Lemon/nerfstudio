# Nerfstudio Direct Training

This repository provides a direct training entrypoint for Gaussian splatting on COLMAP datasets. The main script is [train.py](train.py), which supports:

- `splatfacto` training
- `dino-splatfacto` training with DINO features
- checkpoint saving and resume
- the Viser viewer
- mixed precision and device selection

## 安装

建议先激活你的 Python 环境，然后在仓库根目录安装依赖：

```bash
pip install -e .
```

如果你已经在 conda 环境里工作，也可以直接在该环境中执行上面的命令。

## 数据集格式

`train.py` 读取的是 COLMAP 数据集目录，默认结构如下：

```text
data_root/
	images/
	sparse/0/
	masks/        # optional
	depths/       # optional
```

其中 `images/` 存放训练图像，`sparse/0/` 存放 COLMAP 重建结果。路径也可以通过 dataparser 配置调整，但默认按这个结构查找。

## 训练

最基础的训练命令：

```bash
python train.py -s /path/to/data_root -m outputs/experiment_name
```

常用参数：

- `--pipeline splatfacto`：3dgs训练
- `--pipeline dino-splatfacto`：启用 DINO 特征训练 默认管线
- `--device cuda|mps|cpu`：训练设备，默认 `cuda`
- `--data-device cpu|gpu|disk`：图像缓存位置，默认 `gpu`
- `--mixed-precision`：启用自动混合精度
- `--disable-viewer` 或 `--vis none`：关闭 viewer
- `--viewer-port 7007`：指定 viewer 端口
- `--max-steps 30000`：最大训练步数
- `--save-every 2000`：checkpoint 保存周期
- `--eval-image-every 100`：单图评估周期
- `--eval-all-every 1000`：全量评估周期
- `--resume`：从模型目录下最新 checkpoint 恢复
- `--load-checkpoint /path/to/step-xxxxx.ckpt`：加载指定 checkpoint，优先级高于 `--resume`

### 示例

关闭 viewer 的普通训练：

```bash
python train.py \
	-s /path/to/data_root \
	-m outputs/garden_splatfacto \
	--disable-viewer
```

DINO 版本训练：

```bash
python train.py \
	-s /path/to/data_root \
	-m outputs/garden_dino \
	--pipeline dino-splatfacto
```

## DINO 特征

当使用 `dino-splatfacto` 时，脚本会在训练前检查 DINO 特征是否齐全；缺失时会自动提取并保存到默认目录 `<model-path>/dino_features/`。你也可以手动指定目录：

```bash
python train.py \
	-s /path/to/data_root \
	-m outputs/garden_dino \
	--pipeline dino-splatfacto \
	--dino-feature-dir /path/to/dino_features
```

如果需要先手动提取特征，也可以直接运行：

```bash
python scripts/extract_dino_features.py \
	--input-dir /path/to/data_root/images \
	--output-dir outputs/garden_dino/dino_features \
	--feature-dim 16 \
	--device cuda
```

默认会生成合并后的 `dino_features.pt`。

## 输出目录

训练脚本会在 `-m / --model-path` 指定的目录下写入：

- `config.yml`：本次训练使用的配置
- `dataparser_transforms.json`：数据解析器导出的相机变换
- `nerfstudio_models/`：checkpoint 目录，文件名形如 `step-000003000.ckpt`

如果使用 `--resume`，脚本会自动从 `nerfstudio_models/` 中最新的 checkpoint 继续训练。

## 说明

- 默认 viewer 端口是 `7007`
- 默认日志和输出目录都以 `-m / --model-path` 为基准
- 训练过程中会周期性打印 loss、PSNR 和显存信息