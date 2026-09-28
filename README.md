# Reins

*Let frontier models drive real robot arms, zero-shot.*

**Pengfei Ye** · MIT CSAIL · [pfy712@mit.edu](mailto:pfy712@mit.edu)

[English](#english) | [中文](#中文)

---

## English

A harness that lets a multimodal model (GPT-6 through the OpenAI or OpenRouter API) control one or two
**AirBot Play** arms zero-shot, from a task written in plain language. The model reasons and calls tools;
the harness measures: it gives the model metric perception (SAM3 segmentation + depth from RealSense
cameras), contact-guarded motion, the robot's own state, and a second model — the *narrator* — that
watches the head camera at 1 Hz and keeps an account of what happened while the acting model was busy.

### 1. Hardware

Our rig, as used for every demo:

| Qty | Item | Notes |
|---|---|---|
| 2 | AirBot Play 6-DoF arm with G2 parallel gripper | **firmware / SDK v5.1.6**. One arm is enough for single-arm tasks. |
| 3 | Intel RealSense D405 | 1 fixed head camera about 0.5 m above the table, looking down; 1 wrist camera on each arm. All on **USB 3**. |
| 2 | USB-CAN adapter | one per arm |
| 1 | PC | Ubuntu 20.04, NVIDIA GeForce RTX 4080 Laptop GPU (12 GB). ≥ 8 GB of GPU memory recommended; SAM3 needs about 4 GB. |

### 2. Install the AirBot Play software — version 5.1.6

> ⚠️ **Use v5.1.6. Do not follow the current "Software installation" page of the AirBot docs.**
> That page describes v5.2 (`airbot-arm` + `arm-sdk`), whose Python API is different. This harness uses
> the v5.1.6 Python package **`airbot_py`** and will not run with the v5.2 SDK.

Download both files from the **v5.1.6** entry of the changelog:
<https://docs.discover-robotics.com/airbot-play/changelog.html#v5.1.6>

- Driver software: `airbot-configure_5.1.6-1_all.deb`
- SDK package: `5.1.6.zip` (contains `airbot_py-5.1.6-py3-none-any.whl`)

The arm service runs in Docker, so install Docker first (<https://docs.docker.com/engine/install/ubuntu/>),
then:

```bash
sudo dpkg -i airbot-configure_5.1.6-1_all.deb     # installs airbot_fsm, the USB-CAN udev rules and tools
```

The Python package is installed into the conda environment in step 3. Bind the USB-CAN adapters as
described in the v5.1.6 product documentation (`产品使用文档-5.1.6.zip` on the same changelog entry).

Start one arm service per arm, each in its own terminal. The first run pulls the runtime image
`airbot-runtime:5.1.6`. The ports must match `config/play_config.json`
(`arm.port`, and `second_arm.port` for two arms):

```bash
airbot_fsm -i can0 -p 50050     # the world ("right") arm
airbot_fsm -i can2 -p 50052     # the second ("left") arm, only for --dual-arm
```

### 3. Create the conda environment

```bash
conda create -n reins python=3.12 -y
conda activate reins

# PyTorch with CUDA (pick the index URL that matches your driver)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# SAM3 (segmentation), from source
git clone https://github.com/facebookresearch/sam3.git ~/sam3-main
pip install -e ~/sam3-main

# the harness itself
pip install numpy==1.26.4 "pillow<12" opencv-python==4.11.0.86 pyrealsense2 scipy openai

# the AirBot SDK, v5.1.6, from 5.1.6.zip
pip install path/to/airbot_py-5.1.6-py3-none-any.whl
pip show airbot_py          # must say Version: 5.1.6
```

Keep `numpy==1.26.4`: SAM3 is built against it, and several packages will try to upgrade it.

### 4. SAM3 weights

The checkpoint `sam3.pt` goes into `$SAM3_HOME/checkpoint/` (`SAM3_HOME` defaults to `~/sam3-main`).
Download it from our Hugging Face mirror, or request access to the official repository
<https://huggingface.co/facebook/sam3> and download it from there:

```bash
pip install -U huggingface_hub
hf download <your-hf-repo> sam3.pt --local-dir ~/sam3-main/checkpoint     # TODO: fill in the repo id
```

If SAM3 lives somewhere else, `export SAM3_HOME=/path/to/sam3`. SAM3 is released by Meta under the
SAM License; its terms apply to the weights wherever you download them from.

### 5. Configure the model API

The harness reads its model settings from environment variables. Put them in your shell profile or in
`harness/.env` (one `KEY=value` per line; the file is git-ignored). **Never commit a key.**

OpenAI API:

```bash
export HARNESS_VLM_API_KEY=sk-...            # your OpenAI key
export HARNESS_VLM_MODEL=gpt-6-astra
export HARNESS_VLM_REASONING=medium
```

OpenRouter:

```bash
export HARNESS_VLM_API_KEY=sk-or-...         # your OpenRouter key
export HARNESS_VLM_BASE_URL=https://openrouter.ai/api/v1
export HARNESS_VLM_API=chat
export HARNESS_VLM_MODEL=openai/gpt-6-astra
export HARNESS_VLM_REASONING=medium
```

When switching back from OpenRouter to OpenAI, `unset HARNESS_VLM_BASE_URL HARNESS_VLM_API` — a
commented-out line in your profile does not remove a variable that is already exported. The first line
the harness prints shows what it uses, e.g. `[VLM] model=gpt-6-astra api=responses base_url=api.openai.com`.

Optional: `HARNESS_VLM_TIMEOUT` (seconds, default 60), `HARNESS_MAX_STEPS` (default 60),
`HARNESS_IMAGE_MAX_EDGE` (default 768).

Check the key, the model id, images and tool calls, without the robot:

```bash
python -m harness.scripts.check_api
```

### 6. Configure your rig

- `config/play_config.json`: arm ports, camera serial numbers (`cameras.head_serial`,
  `cameras.wrist_serial`, `second_arm.wrist_serial`), workspace bounds, zero and home joint poses.
- `calibration/play/`: camera and arm calibration. **The files shipped here are the calibration of our
  own rig. If you use different arms or cameras, or move any of them, recalibrate in your own setup.**
  See [calibration/play/README.md](calibration/play/README.md).
- `harness/grippers/airbot_g2.json`: the finger dimensions the model is told (stock G2 fingers).
  For other fingers, write your own file and pass `--gripper-facts path/to/file.json`.

### 7. Run

Run everything from the repository root, with the arm services running.

```bash
python -m harness.scripts.check_cameras      # all cameras present and on USB 3?
python -m harness.scripts.probe_contact      # the arm feels contact (moves the arm: clear the table first)

python -m harness.scripts.run_pickplace --narrator "put the black block in the green bowl"
python -m harness.scripts.run_pickplace --narrator --instruction-file my_task.txt
python -m harness.scripts.run_pickplace --narrator --dual-arm --max-steps 150 \
    --instruction-file my_two_arm_task.txt
```

A task is plain text in any language: say what to do and what counts as success. Long tasks belong in a
file passed with `--instruction-file`: quotes and `>` in a task typed on the command line are eaten by
the shell.

After every episode the arm opens its gripper and returns to its zero joint pose. On start it does not move; pass `--move-on-start` to send it to the zero pose first. Stopping: `q` or `Esc` in the live
window; `Ctrl+C` once stops after the current action and parks the arm, twice parks immediately, three
times quits and leaves the arm where it is.

Logs go to `harness/logs/<episode>/`: `session.jsonl` (the whole conversation with every tool result),
the images the model saw, the narrator's frames and account, and `outcome.json`. Only the latest episode
is kept unless you pass `--keep-logs 0`.

### 8. Troubleshooting

- **A camera is missing or slow**: `check_cameras` says which one is on USB 2. With three D405s, raise
  the USB buffer: add `usbcore.usbfs_memory_mb=1000` to the kernel command line.
- **A camera image stops changing**: restart the run; the camera pipeline is re-created on start.
- **`Not connected to the server`**: the arm service stopped; restart its `airbot_fsm` and the run.
- **`CUDA out of memory` when SAM3 loads**: another process holds the GPU (`nvidia-smi`).
- **The model API stalls**: requests time out after 60 s and are retried; the narrator pauses meanwhile.

### License

MIT License, copyright (c) 2026 Pengfei Ye; see [LICENSE](LICENSE). This project uses SAM3 (SAM License,
Meta) and the AirBot Play SDK (Discover Robotics); their own licenses apply to them.

---

## 中文

一套 harness，让多模态大模型（通过 OpenAI 或 OpenRouter 接口调用 GPT-6）根据自然语言写的任务，
零样本地控制一台或两台 **AirBot Play** 机械臂。模型负责推理和调用工具；harness 负责测量：
它为模型提供度量级的感知（SAM3 分割 + RealSense 深度）、带接触检测的运动、机器人自身的状态，
以及第二个模型——*叙述者*——它以 1 Hz 观察头部相机，把动作模型忙碌期间发生的事记录下来。

### 1. 硬件

我们的实验台（所有 demo 都在这套设备上完成）：

| 数量 | 设备 | 说明 |
|---|---|---|
| 2 | AirBot Play 六自由度机械臂 + G2 平行夹爪 | **固件 / SDK v5.1.6**。单臂任务只需要一台。 |
| 3 | Intel RealSense D405 | 1 台头部相机，固定在桌面上方约 0.5 m 处向下看；每条臂 1 台腕部相机。全部接 **USB 3**。 |
| 2 | USB-CAN 转换器 | 每条臂一个 |
| 1 | 电脑 | Ubuntu 20.04，NVIDIA GeForce RTX 4080 Laptop GPU（12 GB）。建议显存 ≥ 8 GB，SAM3 约占 4 GB。 |

### 2. 安装 AirBot Play 软件——版本 5.1.6

> ⚠️ **请使用 v5.1.6，不要照着 AirBot 官网当前的"软件安装"页面装。**
> 那个页面写的是 v5.2（`airbot-arm` + `arm-sdk`），Python 接口不一样。本 harness 使用的是
> v5.1.6 的 Python 包 **`airbot_py`**，用 v5.2 的 SDK 无法运行。

从更新日志的 **v5.1.6** 条目下载下面两个文件：
<https://docs.discover-robotics.com/airbot-play/changelog.html#v5.1.6>

- 驱动软件：`airbot-configure_5.1.6-1_all.deb`
- SDK 包：`5.1.6.zip`（里面是 `airbot_py-5.1.6-py3-none-any.whl`）

机械臂服务运行在 Docker 里，所以先安装 Docker（<https://docs.docker.com/engine/install/ubuntu/>），然后：

```bash
sudo dpkg -i airbot-configure_5.1.6-1_all.deb     # 安装 airbot_fsm、USB-CAN 的 udev 规则和相关工具
```

Python 包在第 3 步装进 conda 环境。USB-CAN 转换器的绑定方法见同一条目下的 v5.1.6 产品使用文档
（`产品使用文档-5.1.6.zip`）。

每条臂启动一个机械臂服务，各开一个终端。第一次运行会拉取运行镜像 `airbot-runtime:5.1.6`。
端口必须和 `config/play_config.json` 一致（`arm.port`，双臂时还有 `second_arm.port`）：

```bash
airbot_fsm -i can0 -p 50050     # 世界坐标系所在的臂（"right"）
airbot_fsm -i can2 -p 50052     # 第二条臂（"left"），只有 --dual-arm 时需要
```

### 3. 创建 conda 环境

```bash
conda create -n reins python=3.12 -y
conda activate reins

# 带 CUDA 的 PyTorch（按你的显卡驱动选择对应的 index URL）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# SAM3（分割模型），从源码安装
git clone https://github.com/facebookresearch/sam3.git ~/sam3-main
pip install -e ~/sam3-main

# harness 本身的依赖
pip install numpy==1.26.4 "pillow<12" opencv-python==4.11.0.86 pyrealsense2 scipy openai

# AirBot SDK v5.1.6，来自 5.1.6.zip
pip install path/to/airbot_py-5.1.6-py3-none-any.whl
pip show airbot_py          # 必须显示 Version: 5.1.6
```

请保持 `numpy==1.26.4`：SAM3 依赖这个版本，而好几个包会试图把它升级。

### 4. SAM3 权重

权重文件 `sam3.pt` 放在 `$SAM3_HOME/checkpoint/` 下（`SAM3_HOME` 默认是 `~/sam3-main`）。
可以从我们的 Hugging Face 镜像下载，也可以在官方仓库 <https://huggingface.co/facebook/sam3>
申请访问后从那里下载：

```bash
pip install -U huggingface_hub
hf download <your-hf-repo> sam3.pt --local-dir ~/sam3-main/checkpoint     # TODO：填入仓库名
```

如果 SAM3 装在别处，设置 `export SAM3_HOME=/path/to/sam3`。SAM3 由 Meta 以 SAM License 发布，
无论从哪里下载，权重都受该许可证约束。

### 5. 配置模型接口

harness 从环境变量读取模型设置。可以写进 shell 配置文件，也可以写进 `harness/.env`
（每行一个 `KEY=value`，该文件已被 git 忽略）。**永远不要把密钥提交到仓库。**

OpenAI 官方接口：

```bash
export HARNESS_VLM_API_KEY=sk-...            # 你的 OpenAI 密钥
export HARNESS_VLM_MODEL=gpt-6-astra
export HARNESS_VLM_REASONING=medium
```

OpenRouter：

```bash
export HARNESS_VLM_API_KEY=sk-or-...         # 你的 OpenRouter 密钥
export HARNESS_VLM_BASE_URL=https://openrouter.ai/api/v1
export HARNESS_VLM_API=chat
export HARNESS_VLM_MODEL=openai/gpt-6-astra
export HARNESS_VLM_REASONING=medium
```

从 OpenRouter 换回 OpenAI 时，要执行 `unset HARNESS_VLM_BASE_URL HARNESS_VLM_API`：
在配置文件里把某行注释掉，并不会清除已经导出的变量。harness 启动时打印的第一行会显示实际使用的配置，
例如 `[VLM] model=gpt-6-astra api=responses base_url=api.openai.com`。

可选：`HARNESS_VLM_TIMEOUT`（秒，默认 60）、`HARNESS_MAX_STEPS`（默认 60）、
`HARNESS_IMAGE_MAX_EDGE`（默认 768）。

不连机器人，先检查密钥、模型名、图像和工具调用是否正常：

```bash
python -m harness.scripts.check_api
```

### 6. 配置你的实验台

- `config/play_config.json`：机械臂端口、相机序列号（`cameras.head_serial`、`cameras.wrist_serial`、
  `second_arm.wrist_serial`）、工作空间范围、零位和 home 关节位姿。
- `calibration/play/`：相机和机械臂的标定。**这里附带的是我们自己实验台的标定结果。
  如果你换了机械臂或相机，或者移动了其中任何一个，请在你自己的环境里重新标定。**
  详见 [calibration/play/README.md](calibration/play/README.md)。
- `harness/grippers/airbot_g2.json`：告诉模型的手指尺寸（原装 G2 手指）。
  换了别的手指，就写一份自己的文件，用 `--gripper-facts path/to/file.json` 指定。

### 7. 运行

所有命令都在仓库根目录下运行，并且机械臂服务要已经启动。

```bash
python -m harness.scripts.check_cameras      # 相机是否都在、是否都在 USB 3 上
python -m harness.scripts.probe_contact      # 机械臂能否感知接触（手臂会动，先清空桌面）

python -m harness.scripts.run_pickplace --narrator "把黑色积木放进绿色的碗里"
python -m harness.scripts.run_pickplace --narrator --instruction-file my_task.txt
python -m harness.scripts.run_pickplace --narrator --dual-arm --max-steps 150 \
    --instruction-file my_two_arm_task.txt
```

任务就是一段纯文本，任何语言都可以：写清楚要做什么、做到什么程度算成功。长的任务描述请写进文件，
用 `--instruction-file` 传入：直接在命令行里输入时，引号和 `>` 会被 shell 吞掉。

每一局结束后，机械臂会张开夹爪并回到零位关节姿态。程序启动时手臂不动；加 `--move-on-start` 会先让它回到零位。停止方式：在实时画面窗口按 `q` 或 `Esc`；
按一次 `Ctrl+C` 会在当前动作结束后停下并让手臂回位，按两次立即回位，按三次直接退出、手臂停在原地。

日志保存在 `harness/logs/<episode>/`：`session.jsonl`（完整对话和每个工具的返回）、模型看过的图片、
叙述者的帧和记录，以及 `outcome.json`。默认只保留最新一局，想全部保留请加 `--keep-logs 0`。

### 8. 常见问题

- **相机缺失或很慢**：`check_cameras` 会指出哪台掉到了 USB 2。三台 D405 同时使用时，
  请在内核启动参数里加上 `usbcore.usbfs_memory_mb=1000`，调大 USB 缓冲区。
- **某台相机画面不再变化**：重新启动程序，相机会重新初始化。
- **`Not connected to the server`**：机械臂服务停了，重启对应的 `airbot_fsm` 和程序。
- **加载 SAM3 时 `CUDA out of memory`**：有其他进程占用了显卡（用 `nvidia-smi` 查看）。
- **模型接口卡住**：请求 60 秒超时后会自动重试，期间叙述者会暂停。

### 许可证

MIT 许可证，版权所有 (c) 2026 Pengfei Ye，见 [LICENSE](LICENSE)。本项目使用了 SAM3（Meta，SAM License）
和 AirBot Play SDK（Discover Robotics），它们各自的许可证适用于它们本身。
