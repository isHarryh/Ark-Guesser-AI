Ark-Guesser-AI
==========
Machine Learning Solution to Arknights Prediction Game  
《明日方舟》争锋频道竞猜玩法的机器学习解决方案

## 介绍 <sub>Intro</sub>

本项目旨在通过机器学习技术来预测《明日方舟》游戏中“争锋频道”（Duel Channel）竞猜玩法的对局结果。

在该游戏玩法中，玩家需要根据对战双方的阵营信息（出战人物及数量）来预测最终的获胜方。玩家选定了竞猜对象后，对战双方由计算机操纵进行自动对战，场上最终存活的一方即为获胜方。若玩家正确预测获胜方将获得奖励，否则可能受到惩罚。

### 赛季说明

> [!NOTE]
> 
> 由于不同的赛季会调整可出战人物的类别、数值及机制，甚至会修改地形要素和 UI，因此每个赛季都需要对代码库进行重新适配。为作区分，**本仓库中不同分支对应了不同的赛季**。不同分支中的模型架构、数据处理方式等内容可能存在较大差异，请您留意。

📍当前赛季分支：适用于 **CN 服务器第 3 赛季：绿藤城（IvyVine）**。

📃所有赛季分支：

|        赛季序号        |      赛季名称      |   开放日期    |   对应分支   |
| :--------------------: | :----------------: | :-----------: | :----------: |
| CN 服务器<br>第 2 赛季 | 蜜果城<br>Honeydew | 2025 年下半年 | [cn-season2] |
| CN 服务器<br>第 3 赛季 | 绿藤城<br>IvyVine  | 2026 年上半年 | [cn-season3] |

[cn-season2]: https://github.com/isHarryh/Ark-Guesser-AI/tree/cn-season2
[cn-season3]: https://github.com/isHarryh/Ark-Guesser-AI/tree/cn-season3

### 模型细节

模型输入为出战人物的离散类以及各个出战人物的数量（通过对游戏截图进行简单 CV 处理来提取）。模型输出是二分类。

模型架构概述如下：

1. 类别嵌入及数量编码融合层
2. 共享多层感知机层
3. 交叉注意力层
4. 池化及比较器层

<details>
<summary>🏗️模型架构变更日志（展开）</summary>

V1 版本（当前分支采用的版本）在上一版本的基础上：

- 使用注意力池化层代替了平均池化层；
- 使用 log1p 数量编码代替了线性数量编码。

</details>

### 性能表现

在二择一竞猜中（不允许“观望”的情景下），模型的平均准确率能够达到 75% 以上。人类玩家的平均水平为 60% 左右。

如果允许“观望”，并且规定模型的观望阈值等于人类玩家的观望阈值，那么在不同的竞猜难度下，模型的赌赢率能够达到人类玩家的 120\~150%，模型的赌输率仅为人类玩家的 20\~70%。

> [!TIP]
> 
> 有关详细的测评数据和图表，请参阅 [Releases](https://github.com/isHarryh/Ark-Guesser-AI/releases) 页面。

## 使用方法 <sub>Usage</sub>

下面介绍的是本项目的完整使用步骤。如果您不希望自己重新采集数据和训练模型，可以前往 [Releases](https://github.com/isHarryh/Ark-Guesser-AI/releases) 页面来下载已经处理好的数据集和模型权重。

### 1. 环境准备

本项目使用 Python 3.12 和 Torch 2.7 进行开发。具体需要安装的依赖库请参阅 [pyproject.toml](pyproject.toml) 文件，请务必确保您已安装了所有规定的依赖项。

### 2. 赛季信息录入

将当前赛季中的所有出战人物的头像图片保存到 `assets/avatars` 目录中。

修改 `src/dataset_generator.py` 文件中的 `DatasetGenerator.VERSION_NAME` 变量为当前赛季的名称，以便区分不同赛季的数据集。

### 3. 数据采集与分析

#### 3.1 截图采集

启动自动化 GUI 工具来采集游戏截图：

```bash
python main.py gui
```

> [!TIP]
>
> 有关自动化采集的详细说明，请参阅 [开发：自动化数据采集](#自动化数据采集) 章节。

#### 3.2 数据集生成

从采集的图片中生成训练数据集：

```bash
python main.py dataset_generate "dataset/images" "dataset/dataset.json" -p 8
```

从采集的图片中生成包含人类排名信息的评估数据集：

```bash
python main.py dataset_generate "dataset/images_eval" "dataset/dataset_eval.json" -p 8 --include-ranking
```

> [!TIP]
>
> 请勿使用和训练数据集相同的图片目录来生成评估数据集。

#### 3.3 数据可视化

生成数据集可视化报告：

```bash
python main.py dataset_visualize "dataset/dataset.json"
```

如果使用评估测试集进行可视化，报告中会包含人类玩家的预测准确率等信息：

```python
python main.py dataset_visualize "dataset/dataset_eval.json"
```

> [!TIP]
>
> 上述命令会启动一个 [Dash](https://dash.plotly.com/) 网页服务来展示可视化信息。

### 4. 模型训练、评估及应用

#### 4.1 模型训练

使用训练数据集来训练模型：

```bash
python main.py train "dataset/dataset.json" "ckpt/ark_guesser_model.pt"
```

#### 4.2 模型评估

使用评估数据集来测试模型性能：

```bash
python main.py eval "dataset/dataset_eval.json" "ckpt/ark_guesser_model.pt"
```

#### 4.3 模型推理应用

使用训练好的模型来对图片进行推理：

```bash
python main.py infer "dataset/dataset.json" "ckpt/ark_guesser_model.pt" "path/to/screenshot.png"
```

## 开发 <sub>Development</sub>

下面介绍的是本项目的具体的开发和实现细节。

### 开发环境

本项目积极采用 Poetry 作为 Python 包管理工具，其依赖项全部定义在 [pyproject.toml](pyproject.toml) 中。当然，您可以自由切换其他您熟悉的工具（例如 conda、uv、venv）来进行虚拟环境的管理。

### 自动化数据采集

为了自动操作游戏来采集游戏截图，我们使用了 MaaFramework + MXU 作为自动化方案。[MaaFramework](https://maafw.com/) 是一个基于图像识别的自动化框架，而 [MXU](https://github.com/MistEO/MXU) 是服务于 MaaFW 的一个前端 GUI 交互软件。

当您运行下述命令：

```bash
python main.py gui
```

这会自动下载 MaaFW 依赖库和 MXU 程序文件到 `gui` 目录中，随后启动 MXU 程序。

您只需启动《明日方舟》游戏窗口，打开玩法界面，然后按照 MXU 程序中的提示，添加采集任务即可完成游戏截图的采集。

### 数据集结构

数据集是单个 JSON 文件，包含以下结构：

- 顶层 JSON：`version`（赛季名）、`names`（角色字符串名称到整数索引的映射）、`data`（对局样本列表）。
- 每条对局样本：`groups` 为长度 2 的列表（左右阵营），元素为 `{"角色索引": 数量}`；`winner` 为 0/1（左侧/右侧获胜），从对局准备截图中识别。
- 评估集会包含额外字段：`game_round`、`human_correct`、`human_wrong`、`human_neutral` 均为整数，从排行榜截图中识别。

## 许可证 <sub>Licensing</sub>

本项目基于 **BSD-3 开源协议**。任何人都可以自由地使用和修改项目内的源代码，前提是要在源代码或版权声明中保留作者说明和原有协议，且不可以使用本项目名称或作者名称进行宣传推广。
