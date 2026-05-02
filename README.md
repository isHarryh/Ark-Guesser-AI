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

### 技术细节

模型输入为出战人物的离散类以及各个出战人物的数量（通过对游戏截图进行简单 CV 处理来提取）。模型输出是二分类。

模型架构概述如下：

1. 类别嵌入及数量编码融合层
2. 共享多层感知机层
3. 交叉注意力层
4. 池化及比较器层

在不同的竞猜难度下，模型性能表现均超过人类组。详情参阅 [Releases](https://github.com/isHarryh/Ark-Guesser-AI/releases) 页面。

## 使用方法 <sub>Usage</sub>

下面介绍的是本项目的完整使用步骤。如果您不希望自己重新采集数据和训练模型，可以前往 [Releases](https://github.com/isHarryh/Ark-Guesser-AI/releases) 页面来下载已经处理好的数据集和模型权重。

### 1. 环境准备

本项目使用 Python 3.12 和 Torch 2.7 进行开发。具体需要安装的依赖库请参阅 [pyproject.toml](pyproject.toml) 文件。

### 2. 赛季信息录入

将当前赛季中的所有出战人物的头像图片保存到 `assets/avatars` 目录中。

修改 `src/dataset_generator.py` 文件中的 `DatasetGenerator.VERSION_NAME` 变量为当前赛季的名称，以便区分不同赛季的数据集。

### 3. 原始数据采集

自动操作游戏来进行对局截图采集：

```bash
python main.py realtime -s "dataset/images" --auto-start
```

从采集的图片中生成训练数据集：

```bash
python main.py dataset_generate "dataset/images" "dataset/dataset.json" -p 8
```

从采集的图片中生成包含人类排名信息的评估数据集（请勿使用和训练数据集相同的图片目录）：

```bash
python main.py dataset_generate "dataset/images_eval" "dataset/dataset_eval.json" -p 8 --include-ranking
```

### 4. 模型训练、评估及应用

使用训练数据集来训练模型：

```bash
python main.py train "dataset/dataset.json" "ckpt/ark_guesser_model.pt"
```

使用评估数据集来测试模型性能：

```bash
python main.py eval "dataset/dataset_eval.json" "ckpt/ark_guesser_model.pt"
```

使用训练好的模型，在游戏中进行实时预测：

```bash
python main.py realtime -s "dataset/images_s3" -i --infer-dataset-path "dataset/dataset.json" --infer-model-path "ckpt/ark_guesser_model.pt"
```

## 许可证 <sub>Licensing</sub>

本项目基于 **BSD-3 开源协议**。任何人都可以自由地使用和修改项目内的源代码，前提是要在源代码或版权声明中保留作者说明和原有协议，且不可以使用本项目名称或作者名称进行宣传推广。
