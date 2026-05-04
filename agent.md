# Image Caption Generator Agent 任务总览

## 项目目标

本项目是 CS444 final project，主题是 image captioning 中的跨模态表示对齐。核心问题是：在 encoder-decoder multimodal architecture 中，如何把 pretrained vision encoder 和 pretrained language decoder 的表示空间有效对齐，从而生成更准确、更流畅的图片描述。

当前主线是使用 pretrained vision encoder 读取图片特征，使用 GPT-2 作为 text decoder，并比较不同 encoder 与 mapping module 对 caption quality 的影响。

## 当前状态

已经完成的基础能力：

- Flickr8k caption 读取、图片路径自动发现、image-level train/validation/test split。
- PyTorch `Dataset` 和 batch collation。
- ViT + GPT-2 baseline，基于 Hugging Face `VisionEncoderDecoderModel`。
- Frozen encoder 训练流程，包括 gradient accumulation、validation loss、checkpoint 保存。
- Caption generation，并把预测保存到 `outputs/<experiment_name>/predictions.json`。
- BLEU-4、METEOR、CIDEr 评估 helper。
- 基础可视化工具，包括 dataset sample、prediction sample、training history 和 metric bar plot。
- Notebook 已经组织出 baseline/debug/full run 的主要实验流程。

还没有完成或需要继续推进的部分：

- CLIP ViT encoder 条件。
- 可选的 MLP mapping module。
- 四个核心实验的统一运行逻辑。
- 完整实验结果表。
- 定性样例对比和最终分析结论。

## 核心实验矩阵

| Experiment | Encoder | Mapper | Decoder | Status |
| --- | --- | --- | --- | --- |
| `vit_no_mapper` | `google/vit-base-patch16-224-in21k` | No | `gpt2` | Baseline 已有代码，需跑正式结果 |
| `vit_mlp_mapper` | `google/vit-base-patch16-224-in21k` | 2-layer MLP | `gpt2` | 待实现 |
| `clip_no_mapper` | `openai/clip-vit-base-patch16` | No | `gpt2` | 待实现 |
| `clip_mlp_mapper` | `openai/clip-vit-base-patch16` | 2-layer MLP | `gpt2` | 待实现 |

最低目标：

- 在 Flickr8k 上训练并评估四个核心实验。
- 对比 BLEU-4、METEOR、CIDEr。
- 分析 CLIP pretrained image-text alignment 和显式 MLP mapper 是否提升 caption quality。

可选扩展目标：

- 给 GPT-2 decoder 加 LoRA fine-tuning。
- 做不同数据量的 ablation study。
- 加 attention visualization 或其他 qualitative visualization。

## 主要 TODO

1. 保持 `vit_no_mapper` 作为受控 baseline。
2. 在 config/modeling 层加入 encoder selection，让同一套流程可以切换 ViT 和 CLIP。
3. 实现可选 `MLPMapper`，用于把 encoder hidden states 映射到 decoder cross-attention 需要的表示空间。
4. 重构 notebook 中 baseline-specific 的代码，改成统一 experiment loop。
5. 依次运行 `vit_no_mapper`、`vit_mlp_mapper`、`clip_no_mapper`、`clip_mlp_mapper`。
6. 为每个实验保存：
   - `training_log.csv`
   - `checkpoint_last/`
   - `predictions.json`
   - BLEU-4、METEOR、CIDEr 结果行
7. 生成最终 comparison table，并用 metric bar plot 可视化。
8. 做 qualitative comparison：同一张图片展示四个模型 prediction 和 human references。
9. 写最终报告分析，回答：
   - CLIP pre-alignment 是否优于 vanilla ViT？
   - MLP mapper 是否改善 vanilla ViT 的对齐？
   - CLIP + MLP 是否有叠加收益？
   - 失败样例暴露了哪些 cross-modal alignment 问题？
10. 时间允许时再实现 LoRA、data-size ablation 和 attention visualization。

## 建议实现顺序

1. 先确认 baseline debug run 可以稳定跑通。
2. 跑 `vit_no_mapper` full baseline，拿到第一行正式结果。
3. 抽象 experiment config，例如 encoder name、是否启用 mapper、experiment name。
4. 实现 `clip_no_mapper`，优先验证 CLIP encoder 能和 GPT-2 decoder 连接并完成 forward/generate。
5. 实现 `MLPMapper`，先在 ViT 条件下验证 loss 和 generation。
6. 跑完四个核心实验。
7. 汇总 metrics、画图、整理 qualitative examples。
8. 最后补报告分析和 optional stretch goals。

## 交付物

最终需要交付：

- 可复现实验 notebook：`CS444_Final_Project_Image_Captioning_Alignment.ipynb`。
- 完整代码模块：`src/cs444_captioning/`。
- 四个核心实验的输出目录：`outputs/<experiment_name>/`。
- 汇总实验结果表，至少包含 Experiment、Encoder、Mapper、BLEU-4、METEOR、CIDEr。
- 预测样例对比图或 notebook section。
- 最终报告中的结果分析段落。

## 注意事项

- 当前任务清单只总结后续要做什么，不要求现在实现 CLIP、MLP 或训练代码。
- 不要把 `agent.md` 当成运行时配置文件；它只是给后续 agent/开发者看的项目执行清单。
- 优先保证四个核心实验可复现，再考虑 LoRA、ablation 和 attention visualization。
