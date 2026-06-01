# Assignment 4 - Implement Simplified 3D Gaussian Splatting

### Data

```
data/
├── chair/images/   # 100 张 multi-view 渲染图像

```

## Task 1: Structure-from-Motion with COLMAP

使用 COLMAP 恢复相机内外参，并得到一组稀疏 3D 点作为 3DGS 的初始化：

```bash
python mvs_with_colmap.py --data_dir data/chair
结果：
data/chair/sparse/0_text/points3D.txt
```
<img width="480" height="292" alt="animation_edited" src="https://github.com/user-attachments/assets/86bbf657-ed9d-48b5-8efc-1a5b391e81bb" />

将恢复的 3D 点重投影回各视角进行验证：

```bash
python debug_mvs_by_projecting_pts.py --data_dir data/chair
结果：
data/chair/projections/*.png
```

---

## Task 2: Simplified 3D Gaussian Splatting (主要部分)
### Requirements
这部分是在autodl 算力云平台完成的。

GPU:RTX 4090D(24GB) * 1

CPU:16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
```setup
conda env create -f environment.yml
conda activate pyto
```

观察 Task 1 的输出可以发现，COLMAP 恢复的 3D 点对于稠密渲染来说过于稀疏。我们将每个点扩展为一个 3D 高斯，使其覆盖周围空间。

### 2.1 3D Gaussian Initialization

参考 paper 公式 (6)：协方差矩阵由缩放矩阵 *S* 和旋转矩阵 *R* 构造。每个高斯需要以下可优化参数：

| 参数 | 说明 |
|------|------|
| Position μ | 初始化为 SfM 3D 点 |
| Rotation R | 用单位四元数参数化 |
| Scaling S | 3 维向量 |
| Opacity o | 标量 |
| Color c | RGB 三通道 |

已在 gaussian_model.py 中由四元数和缩放参数构造 **3D 协方差矩阵**。

### 2.2 Project 3D Gaussians to 2D

参考 paper 公式 (5)，将 3D 高斯投影到图像平面需要：

- 世界到相机的变换矩阵 *W*
- 投影变换的雅可比矩阵 *J*

投影后的 2D 协方差为 $\Sigma' = J W \Sigma W^T J^T$。

已在 gaussian_renderer.py中实现 3D → 2D 投影。

### 2.3 Compute 2D Gaussian Values

2D Gaussian 在像素 $\mathbf{x}$ 处的取值：

$$
f(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \frac{1}{2\pi\sqrt{|\boldsymbol{\Sigma}_i|}} \exp\left(P_{(\mathbf{x},i)}\right), \quad P_{(\mathbf{x},i)} = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)
$$

其中 **μᵢ** 与 **Σᵢ** 为投影后的 2D 高斯中心与协方差。

已在gaussian_renderer.py中计算 Gaussian 取值。

### 2.4 Volume Rendering via α-blending

给定 *N* 个按深度排序的 2D 高斯，每个高斯在像素 $\mathbf{x}$ 处的 alpha 与透射率为：

$$
\alpha_{(\mathbf{x}, i)} = o_i \cdot f(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i), \qquad T_{(\mathbf{x}, i)} = \prod_{j<i} (1 - \alpha_{(\mathbf{x}, j)})
$$

最终像素颜色由各高斯按 α-blending 累加（paper 公式 1-3）。

已在 gaussian_renderer.py中实现最终渲染。

### Train your 3DGS

完成上述代码后，启动训练：

```bash
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

### Render a Multi-view Video (Optional)

训练完成后，用 render_3dgs_mv.py沿一个绕场景中心的**水平圆轨迹**渲染一段连续视角视频，便于直观检查重建质量：

```bash
python render_3dgs_mv.py \
    --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000060.pt \
    --num_frames 240 --fps 30
# 默认输出: <colmap_dir>/render_mv.mp4
```

up 轴由训练相机的 y 轴平均自动估计（NeRF 合成数据图像均为正放），orbit 半径与高度取训练相机的均值。

---

## Task 3: Compare with the Official 3DGS Implementation

[gaussian-splatting](assignment4/gaussian-splatting)为官方的3D Gaussian Splatting
Task3是在google colab中完成的。

```bash
# 在colab克隆仓库
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
```

```bash
# 
!python /content/drive/MyDrive/gaussian-splatting/train.py -s /content/drive/MyDrive/gaussian-splatting/chair --eval
```
## 实验设置

| 项目 | 简化版 PyTorch 3DGS | 官方 3DGS |
|---|---|---|
| 数据集 | `data/chair` | `gaussian-splatting/chair`，与 `data/chair` 同为 100 张 chair 多视角图像 |
| 输入图像 | 原始图像为 800 x 800 RGBA，训练时下采样 8 倍，约 100 x 100 | `resolution=-1`，使用原始分辨率 |
| 训练视图 | 100 张图像 | `--eval` 模式，约 87 张 train views 和 13 张 test views |
| 初始点云 | COLMAP 稀疏点云，约 13,556 个点 | 同源 COLMAP 点云初始化 |
| 最终高斯数量 | 固定为初始点数附近，未做 densification | 30,000 iter 后为 370,042 个高斯 |
| 主要输出 | `data/chair/checkpoints/` | `gaussian-splatting/output/7dc68038-2/` |

需要注意的是，两者的训练分辨率和 train/test 划分并不完全一致，因此当前结果更适合作为实现差异的实验观察。若需要严格公平的数值比较，应在相同 GPU、相同分辨率、相同 train/test split 下重新运行两套代码。

## 渲染质量对比

官方 3DGS 已经完成 `train.py`、`render.py` 和 `metrics.py` 三步，指标文件为 `gaussian-splatting/output/7dc68038-2/results.json`。在 `ours_30000` 下得到的测试集指标如下：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 官方 3DGS, 30,000 iter | 4.3386 | 0.2383 | 0.2928 |
| 简化版 PyTorch 3DGS | 暂未计算 | 暂未计算 | 暂未计算 |

从已有可视化结果看，简化版 PyTorch 3DGS 能够在训练视角上重建出椅子的主体形状、颜色和大致轮廓，但细节较模糊，边缘存在扩散和破碎现象。官方 3DGS 的局部物体细节更丰富，例如椅背、椅腿和绿色纹理保留得更清楚；但当前这次官方实验在背景区域出现了明显的白色或灰色 floaters，导致黑色背景与渲染结果差异很大，因此 PSNR 和 SSIM 被显著拉低。

简化版最终 debug 图：

![Simplified PyTorch 3DGS debug result](data/chair/checkpoints/debug_images/epoch_0199.png)

官方 3DGS 测试视角示例：

| Ground Truth | Official 3DGS Render |
|---|---|
| ![Official GT](gaussian-splatting/output/7dc68038-2/test/ours_30000/gt/00000.png) | ![Official render](gaussian-splatting/output/7dc68038-2/test/ours_30000/renders/00000.png) |

官方 3DGS 在 TensorBoard 中记录的 test PSNR 在 7,000 iter 时约为 13.07 dB，但 30,000 iter 时降至约 4.34 dB。这说明在本数据设置下，后期 densification 和优化可能引入了较多背景区域的高斯 floaters。由于 chair 数据集的 PNG 带 alpha 通道，背景区域的处理方式会对指标造成较大影响；当前官方配置中 `white_background=False`，即使用黑色背景合成。

## 训练速度对比

根据当前保存的文件时间戳和官方 TensorBoard 日志，可以得到如下粗略速度统计：

| 方法 | 训练规模 | 训练耗时 | 平均速度 |
|---|---:|---:|---:|
| 简化版 PyTorch 3DGS | 200 epochs，约 20,000 次图像更新 | 约 174.7 min | 约 524 ms / update |
| 官方 3DGS | 30,000 iterations | 约 20.54 min | TensorBoard 最后 100 次约 34.0 ms / iter |

简化版 PyTorch 3DGS 的训练时间由 `checkpoint_000000.pt` 到 `epoch_0199.png` 的时间戳粗略估计：从 2026-05-29 20:07:02 到 2026-05-29 23:01:44，约 174.7 分钟。官方 3DGS 的训练时间由 TensorBoard 事件中首个和最后一个 `iter_time` 记录估计，约为 20.54 分钟。

即使官方实现使用了更高图像分辨率，并且最终高斯数量增长到 370,042 个，它的单次迭代速度仍明显快于简化版。这主要来自官方 CUDA rasterizer、tile-based 并行渲染、可见性裁剪和更高效的显存访问。相比之下，简化版使用纯 PyTorch 在图像平面上直接计算高斯贡献，接近对所有高斯和所有像素做全量计算，计算复杂度和中间张量开销都较大。

## 显存占用对比

当前实验文件中没有保存两种方法的峰值显存记录，因此无法给出严格的显存数值对比。就实现方式而言，二者的显存占用来源不同：

| 方法 | 显存占用特点 |
|---|---|
| 简化版 PyTorch 3DGS | 没有 adaptive densification，高斯数量较少；但渲染时会在 PyTorch 中构造大量像素级中间张量，显存效率较低 |
| 官方 3DGS | 高斯数量通过 densification 增长很多；但渲染核心由 CUDA kernel 实现，并采用 tile-based rasterization、排序和裁剪，避免了大量无效像素计算 |

因此，简化版虽然高斯数量少，但未必具有更好的显存效率；官方实现虽然高斯数量更多，但其 rasterizer 对显存访问和并行计算做了专门优化。若补充显存实验，应在训练时记录 `nvidia-smi` 的峰值 `memory.used`，或在代码中记录 `torch.cuda.max_memory_allocated()`。

## 差异来源分析

两种实现的差异主要来自以下几个方面。

首先，官方 3DGS 使用专门的 CUDA rasterizer，并采用 tile-based rendering。它会先判断每个 Gaussian 会影响哪些 tile 和像素，再进行排序、混合和裁剪，避免对整张图像做全量计算。简化版 PyTorch 实现则更接近直接投影全部 Gaussian，并在像素网格上计算 2D Gaussian 值和 alpha blending，因此速度较慢，中间显存开销也更大。

其次，官方实现包含 adaptive Gaussian densification 和 pruning。训练过程中，高梯度区域会被克隆或拆分出更多 Gaussian，从而表达细节和边缘结构。本实验中官方模型从初始稀疏点云增长到 370,042 个 Gaussian。简化版没有 densification，只能依赖 COLMAP 初始稀疏点，因此对椅子纹理、细边缘和遮挡区域的表达能力有限。

第三，官方 3DGS 使用球谐函数表示视角相关颜色，并优化 opacity、scale、rotation、position 等参数；简化版主要使用 RGB 颜色和基础的 3D covariance 投影，外观表达能力更弱。官方损失中还结合了 L1 和 SSIM，而简化版主要使用 RGB L1 loss。

最后，当前官方实验出现了明显背景 floaters，这可能与透明背景图像、COLMAP 稀疏点云、无显式物体 mask 以及后期 densification 有关。由于评价指标会统计整张图像，背景区域的大面积错误会严重影响 PSNR 和 SSIM。因此，本次官方 3DGS 虽然在物体局部细节上更强，但最终量化指标并没有体现出预期优势。

## 小结

总体来看，官方 3DGS 在算法完整性和工程效率上明显优于本作业的纯 PyTorch 简化版。官方实现借助 CUDA tile-based rasterizer、adaptive densification、球谐颜色和更高效的优化流程，在训练速度和细节表达上具有明显优势。简化版的优势是结构清晰，便于理解 3D Gaussian 初始化、投影、2D Gaussian 计算和 alpha blending 的基本流程，但由于缺少 densification 和高性能 rasterizer，训练速度、显存效率和最终渲染质量都受到限制。

当前报告仍缺少严格的峰值显存数值，以及简化版在相同 test split 上的 PSNR、SSIM、LPIPS 指标。若后续补充这两部分，Task 3 的定量比较会更加完整。

