# MNIST 多层感知机可视化器

![MNIST MLP 可视化器截图](https://raw.githubusercontent.com/caicaivic0322/Neural-Network-Visualisation/main/assets/screenshot.png)

一个交互式的Web可视化工具，用于展示在MNIST手写数字数据集上训练的多层感知机网络。绘制一个数字，观察激活如何在3D网络中传播，并实时查看预测概率。

## 项目简介

这是一个教育性质的可视化项目，旨在帮助理解神经网络的工作原理。通过直观的3D可视化，您可以看到：
- 手写数字的实时识别过程
- 神经网络各层的激活状态
- 权重连接的可视化表示
- 预测概率的动态变化

## 功能特色

- **中文界面支持**：完整的中文本地化，包括所有UI元素、提示信息和说明文本
- **交互式绘图**：在28×28网格上绘制数字（左键绘制，右键擦除）
- **3D网络可视化**：使用Three.js构建的立体网络结构展示
- **实时预测**：即时显示网络对输入数字的预测结果
- **时间轴控制**：可以查看网络训练过程中的权重变化
- **神经元详情**：点击查看任意神经元的详细信息，包括输入、权重和激活值

## 快速开始

### 1. 启动本地服务器

从项目根目录启动静态文件服务器（任何服务器都可以，这里使用Python作为示例）：

```bash
python3 -m http.server 8000
```

### 2. 访问应用

在浏览器中打开 `http://localhost:8000`，开始探索神经网络的世界！

## 训练新模型（可选）

如果您想训练自己的模型，可以使用提供的训练脚本：

### 安装Python依赖

```bash
python3 -m pip install torch torchvision
```

### 训练模型

```bash
python3 training/mlp_train.py \
  --epochs 5 \
  --hidden-dims 128 64 \
  --batch-size 256 \
  --export-path exports/mlp_weights.json
```

训练完成后，更新 `assets/main.js` 中的 `VISUALIZER_CONFIG.weightUrl` 配置，然后刷新浏览器加载新的权重。

## 向原作者致敬

本项目基于 [DFin/Neural-Network-Visualisation](https://github.com/DFin/Neural-Network-Visualisation) 进行中文本地化改造。感谢原作者Daniel Firth创建了如此优秀的教育工具，让我们能够直观地理解神经网络的内部工作原理。

原项目的特点：
- 创新的3D可视化方法
- 流畅的交互体验
- 详细的权重时间轴展示
- 优秀的性能优化

中文本地化改进：
- 完整的中文界面翻译
- 更符合中文用户习惯的术语表达
- 保持原有视觉效果和功能完整性

## 技术栈

- **前端**：Three.js, HTML5 Canvas, JavaScript
- **后端**：PyTorch（用于训练）
- **数据**：MNIST手写数字数据集
- **部署**：纯静态文件，支持任何Web服务器

## 使用说明

1. **绘制数字**：在左侧网格中绘制0-9的数字
2. **观察网络**：右侧3D视图显示神经网络结构
3. **查看预测**：顶部显示网络的预测结果和置信度
4. **探索神经元**：点击任意神经元查看详细信息
5. **时间轴控制**：使用底部滑块查看训练过程中的变化

## 部署

项目支持简单的部署脚本，可以方便地发布到生产环境：

```bash
./deploy.sh
```

部署脚本会将当前代码导出到 `releases/current/` 目录，并在 `releases/backups/` 中创建备份。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。如果您发现了翻译错误或者有更好的中文表达建议，请不吝赐教。

## 许可证

本项目遵循原项目的开源协议。具体许可证信息请参考原项目仓库。

---

*Made with ❤️ by the Chinese localization team*
*基于Daniel Firth的优秀工作*