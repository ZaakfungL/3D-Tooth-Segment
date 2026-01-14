# 项目记忆：3D牙齿分割（PyTorch/MONAI）

## 1. 项目概述
本项目实现了一个用于3D牙齿分割的深度学习框架，涵盖从基础方法到高级神经架构搜索（NAS）。
- **核心框架**：PyTorch, MONAI
- **任务**：3D体积分割（CBCT/MRI）
- **关键方法**：3D U-Net, DiNTS（可微神经网络拓扑搜索）, TMO（可信任动量优化）。

## 2. 环境与配置
- **Conda环境**：`tooth`
- **Python解释器路径**（建议直接调用）：
  - 本地：`/home/lzf/miniconda3/envs/tooth/bin/python`
- **项目根目录**：`/home/lzf/Code/3D-Tooth-Segment`
- **本地数据集路径**：`/home/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI`
- **服务器414数据集路径**：`/data/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI`
- **服务器708数据集路径**：`/home/data2/lzf/Code/dataset/nnUNet_raw/Dataset701_STS3D_ROI`
- **⚠️ 重要**：`DATA_DIR` 在Python脚本中是**硬编码**的，运行前必须验证。

## 3. 运行规范
- 所有命令应从项目根目录运行。
- 使用 `&&` 确保先激活环境再运行，若激活失败则不会执行 Python 命令

## 4. 代码结构
- **`src/models/`**：
  - `unet3D.py`：基线模型。
  - `dints.py`：用于NAS的动态网络定义。
- **`src/dataloaders/`**：
  - `basic_loader.py`：标准监督加载器。
  - `combo_loader.py`：用于SSL/TMO的复杂加载器（4流数据）。
- **`src/ssl/`**：TMO优化器和一致性损失工具。
- **`results/`**：存储搜索结果。
- **`weights/`**：存储模型检查文件（`.pth`）。
- **`lunwen/`**：项目的论文成果，latex格式文件。

## 5. 开发者备注
- **确定性**：随机种子一般设置为`2025`以确保可复现性。
- **警告**：脚本中抑制了`monai.inferers.utils`的用户警告。

## 6. 论文写作规范
### 6.1 文件结构
- **论文目录**：`/home/lzf/Code/3D-Tooth-Segment/lunwen/`
- **主文件**：`main.tex`
- **章节文件**：`sections/`
  - `abstract.tex` - 摘要
  - `intro.tex` - 引言
  - `related.tex` - 相关工作
  - `method.tex` - 方法
  - `experiments.tex` - 实验
  - `conclusion.tex` - 结论
- **图表目录**：`figures/`
- **参考文献**：`refs.bib`
### 6.2 LaTeX编译
```bash
# 从项目根目录编译
cd /home/lzf/Code/3D-Tooth-Segment/lunwen && xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
# 清理辅助文件
cd lunwen && rm -f *.aux *.bbl *.blg *.log *.out *.synctex.gz
```