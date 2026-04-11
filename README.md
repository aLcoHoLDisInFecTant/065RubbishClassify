# 垃圾分类多模态 AI 系统

本项目实现了“上传图片 → 视觉模型识别垃圾类别 → 大模型生成本地化回收指导与 Upcycling 建议 → 前端展示”的端到端流程。

## 功能概览

- 图片垃圾识别：ResNet50 迁移学习，输出 5 类（`plastic`/`glass`/`metal`/`paper`/`organic`）
- 置信度阈值降级：当置信度 < 0.5 返回 `Unknown` 并提示重拍
- 文本生成：通过 OpenAI SDK 兼容接口（已配置智增增 `base_url`）生成“回收步骤 + 创意升级改造”
- Web 前端：上传图片、预览、显示分类结果与建议

## 目录与关键文件

- `prepare_data.py`：将原始数据集映射/合并为统一 5 类数据集
- `train.py`：训练 ResNet50 并导出权重
- `best_model.pth`：训练得到的最佳权重（自动生成）
- `class_names.json`：类别顺序（自动生成）
- `backend.py`：FastAPI 后端（CV 推理 + LLM 文本生成）
- `index.html`：前端页面
- `unified_dataset/`：统一后的训练数据目录（自动生成）

## 数据集来源与标签映射

项目会读取以下目录的数据（已放在本机）：

- `d:\065创新\garbageClassification\garbage_classification`
- `d:\065创新\archive\dataset-resized`

合并为统一 5 类的数据集目录：`d:\065创新\unified_dataset`

映射规则（简化说明）：

- `plastic`：`plastic`
- `glass`：`brown-glass`/`green-glass`/`white-glass`/`glass`
- `metal`：`metal`
- `paper`：`paper`/`cardboard`
- `organic`：`biological`

## 环境要求

- Windows + Conda
- 推荐使用：`pytorch-gpu-11.8`（本项目已在该环境下训练/运行）

## 安装依赖（可选）

如果你的环境缺少依赖，可在 `pytorch-gpu-11.8` 中安装：

```powershell
conda activate pytorch-gpu-11.8
cd d:\065创新

# 如果 pip 出现 SSL 问题，可使用镜像源（示例：阿里源）
python -m pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## 配置大模型（智增增）

在项目根目录创建/修改 `.env`（不要提交到仓库）：

```ini
OPENAI_API_KEY=你的apikey
OPENAI_BASE_URL=https://api.zhizengzeng.com/v1
```

后端会使用 OpenAI Python SDK 的 `base_url` 来请求智增增接口。

## 一键运行（推荐）

### 1) 启动后端服务

```powershell
conda activate pytorch-gpu-11.8
cd d:\065创新
python -m uvicorn backend:app --host 0.0.0.0 --port 8000
```

启动成功会看到类似日志：`Uvicorn running on http://0.0.0.0:8000`

### 2) 启动前端静态服务

```powershell
cd d:\065创新
python -m http.server 8090 --bind 127.0.0.1
```

浏览器打开：

- `http://localhost:8090/index.html`

上传图片并点击 “Classify”。

## 训练流程（如需重新训练）

### 1) 生成统一数据集

```powershell
conda activate pytorch-gpu-11.8
cd d:\065创新
python prepare_data.py
```

### 2) 训练模型

```powershell
conda activate pytorch-gpu-11.8
cd d:\065创新
python train.py
```

训练完成会生成：

- `best_model.pth`
- `class_names.json`

## API 使用说明

### `POST /classify`

- 请求：`multipart/form-data`，字段名 `file`（图片文件）
- 响应：

```json
{
  "label": "plastic",
  "confidence": 0.93,
  "instructions": "...",
  "upcycling": "..."
}
```

PowerShell 测试示例（将路径换成你的图片路径）：

```powershell
$img = "C:\\path\\to\\test.jpg"
$form = @{ file = Get-Item $img }
Invoke-RestMethod -Uri "http://127.0.0.1:8000/classify" -Method Post -Form $form
```

## 常见问题

- 前端端口被占用/权限错误：换一个端口（如 `8090`、`8091`），并尽量使用 `--bind 127.0.0.1`
- 大模型无法生成建议：检查 `.env` 中的 `OPENAI_API_KEY` 与 `OPENAI_BASE_URL` 是否正确

