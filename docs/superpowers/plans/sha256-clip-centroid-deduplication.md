# 基于 SHA-256 与 CLIP 特征质心仲裁的训练集去重方案

## 1. 目标

针对训练集中“同一张图片被放入多个类别目录”的显式标签矛盾，构建一套：

- 自动化；
- 可复现；
- 不修改原始数据；
- 可审计；
- 可共享；
- 可直接接入现有训练代码；

的数据去重与冲突仲裁流程。

当前已知数据统计：

| 统计项 | 数值 |
|---|---:|
| 训练图片总数 | 103,218 |
| 唯一 SHA-256 | 101,980 |
| 跨类重复组 | 1,032 |
| 涉及文件 | 2,095 |
| 最大候选类别数 | 4 |

本方案只处理**完全相同文件内容产生的确定性重复与跨类冲突**。普通错标、弱相关图片、近重复图片、相似图片不在本方案的直接处理范围内。

---

## 2. 核心决策

采用：

> SHA-256 精确分组 + 同类重复去重 + 鲁棒 CLIP 质心仲裁 + 低置信度拒绝 + 仲裁样本降权训练

具体规则：

1. **唯一图片**：正常保留；
2. **同 SHA、同类别重复**：只保留一个副本；
3. **同 SHA、跨类别冲突**：
   - 使用非冲突唯一图片计算类别质心；
   - 在该冲突组的候选类别中进行余弦相似度仲裁；
   - 高置信度时保留最优类别对应的一份文件；
   - 低置信度时整个冲突组不参与监督训练；
4. 所有决策通过 manifest 文件表达；
5. 不物理删除官方原始训练数据。

不采用以下策略：

- 随机保留某个目录标签；
- 根据文件出现次数简单投票；
- 对全部冲突组无条件强制仲裁；
- 直接修改或删除原始数据目录；
- 使用测试集参与质心计算或阈值选择。

---

## 3. 推荐工程目录

```text
project/
├── data/
│   └── train/
│       ├── 0000/
│       ├── 0001/
│       └── ...
├── artifacts/
│   └── dedup/
│       ├── dataset_index.csv
│       ├── duplicate_groups.csv
│       ├── conflict_groups.csv
│       ├── arbitration_results.csv
│       ├── clean_train_manifest.csv
│       ├── delete_list.txt
│       ├── dedup_stats.json
│       ├── centroids.pt
│       └── dataset_fingerprint.txt
├── configs/
│   └── dedup_clip_centroid.yaml
├── scripts/
│   ├── build_dataset_index.py
│   ├── build_duplicate_groups.py
│   ├── arbitrate_conflicts.py
│   ├── verify_dedup_artifacts.py
│   └── export_symlink_dataset.py
└── src/
    └── datasets/
        └── manifest_dataset.py
```

---

## 4. 输入要求

### 4.1 原始训练集

训练集按类别目录组织：

```text
train/
├── 0000/
│   ├── xxx.jpg
│   └── yyy.jpg
├── 0001/
└── ...
```

类别标签由父目录名确定。

### 4.2 CLIP 特征文件

已有：

```text
features.pt
shape = [103218, 512]
```

必须同时具备与特征逐行对应的路径列表，例如：

```text
feature_paths.txt
```

每行一个相对路径：

```text
0000/xxx.jpg
0000/yyy.jpg
0001/zzz.jpg
```

必须保证：

```python
features[i] <-> feature_paths[i]
```

如果只有 `features.pt` 而没有路径映射，禁止直接用于仲裁。必须回溯特征提取脚本，恢复特征与路径之间的稳定对应关系。

### 4.3 特征来源记录

在配置和统计文件中记录：

- 特征模型：OpenAI CLIP ViT-B/32；
- 特征层：图像编码器最终输出；
- 是否已归一化；
- 特征维度；
- 提取时的数据顺序；
- `features.pt` 的 SHA-256；
- `feature_paths.txt` 的 SHA-256。

---

## 5. 配置文件

建议创建 `configs/dedup_clip_centroid.yaml`：

```yaml
dataset:
  root: /path/to/train
  class_from_parent_dir: true
  extensions:
    - .jpg
    - .jpeg
    - .png
    - .bmp
    - .webp

features:
  path: /path/to/features.pt
  paths_file: /path/to/feature_paths.txt
  expected_num_samples: 103218
  expected_dim: 512
  normalize: true

hashing:
  algorithm: sha256
  chunk_size_bytes: 1048576

centroid:
  method: trimmed_mean
  trim_ratio: 0.10
  min_class_samples_for_trim: 20
  normalize_centroid: true
  exclude_cross_class_conflicts: true
  unique_sha_only: true

arbitration:
  candidate_labels_only: true
  require_global_top1_in_candidates: true
  margin_threshold: 0.02
  similarity_threshold_mode: percentile
  similarity_percentile: 10
  low_confidence_action: drop_entire_group

training:
  unique_weight: 1.0
  same_class_dedup_weight: 1.0
  resolved_conflict_weight: 0.5
  unresolved_conflict_weight: 0.0

output:
  dir: artifacts/dedup
  physically_delete_files: false
  save_centroids: true

runtime:
  seed: 42
  num_workers: 8
```

---

## 6. 完整处理流程

## 6.1 阶段 A：建立数据索引

遍历训练集，生成 `dataset_index.csv`。

字段：

```text
row_id
relative_path
absolute_path
label
file_size
sha256
feature_row
```

示例：

```csv
row_id,relative_path,label,file_size,sha256,feature_row
0,0001/a.jpg,0001,34567,abc123...,0
1,0237/b.jpg,0237,34567,abc123...,1
```

要求：

- `relative_path` 使用 POSIX 风格 `/`；
- 路径按固定规则排序；
- 所有图片都必须存在；
- 所有图片必须成功计算 SHA-256；
- 所有路径必须在 `feature_paths.txt` 中找到；
- `feature_row` 唯一；
- 总行数必须等于 103,218。

推荐排序规则：

```python
sorted(relative_paths)
```

但如果 `features.pt` 按其他顺序生成，应以 `feature_paths.txt` 为准，不得假设特征顺序等于目录排序顺序。

---

## 6.2 阶段 B：按 SHA-256 分组

对 `dataset_index.csv` 按 `sha256` 聚合。

每个组计算：

```text
group_size
unique_label_count
candidate_labels
paths
```

分为三种组。

### 类型 1：唯一组

```text
group_size = 1
unique_label_count = 1
```

处理：

```text
decision = keep_unique
```

### 类型 2：同类重复组

```text
group_size > 1
unique_label_count = 1
```

处理：

- 按相对路径字典序保留一个副本；
- 其他副本标记删除；
- 只保留的那一份参与质心计算；
- 不让重复图片在训练中被多次采样。

示例：

```text
0001/a.jpg -> keep_same_class_duplicate
0001/b.jpg -> delete_same_class_duplicate
```

### 类型 3：跨类冲突组

```text
group_size > 1
unique_label_count > 1
```

处理：

```text
decision = pending_arbitration
```

该组所有图片均不得参与初始类别质心计算。

输出 `duplicate_groups.csv`：

```csv
sha256,group_size,unique_label_count,candidate_labels,group_type
abc123...,2,2,0001|0237,cross_class_conflict
def456...,2,1,0012,same_class_duplicate
```

---

## 6.3 阶段 C：建立非冲突唯一特征集合

类别质心只使用：

- 唯一组；
- 同类重复组中保留的一个副本；
- 不属于任何跨类冲突组的图片。

记特征为：

\[
z_i \in \mathbb{R}^{512}
\]

先归一化：

\[
\tilde z_i = \frac{z_i}{\|z_i\|_2}
\]

代码要求：

```python
features = features.float()
features = torch.nn.functional.normalize(features, dim=1)
```

检查：

```python
torch.isfinite(features).all()
```

对零向量或非有限值：

- 记录异常；
- 对应样本不用于质心计算；
- 若冲突组图片特征异常，则该组直接拒绝。

---

## 6.4 阶段 D：计算鲁棒类别质心

### 6.4.1 初始质心

对于类别 \(c\) 的非冲突唯一特征集合 \(I_c\)：

\[
\mu_c^{(0)}
=
\operatorname{Normalize}
\left(
\frac{1}{|I_c|}
\sum_{i\in I_c}\tilde z_i
\right)
\]

### 6.4.2 类内截尾

计算每个样本与初始质心的余弦相似度：

\[
q_i = \tilde z_i^\top \mu_c^{(0)}
\]

当类别样本数满足：

```text
N_c >= min_class_samples_for_trim
```

时，删除该类别相似度最低的 `trim_ratio` 部分。

默认：

```text
trim_ratio = 0.10
```

保留集合：

\[
I_c^{core}
=
\{i \mid q_i \ge Q_{0.10}(q)\}
\]

重新计算：

\[
\mu_c
=
\operatorname{Normalize}
\left(
\frac{1}{|I_c^{core}|}
\sum_{i\in I_c^{core}}\tilde z_i
\right)
\]

当类别样本过少时，使用普通均值质心，不进行截尾。

### 6.4.3 保存质心元数据

`centroids.pt` 建议结构：

```python
{
    "centroids": Tensor[num_classes, 512],
    "class_names": List[str],
    "class_counts_raw": Dict[str, int],
    "class_counts_core": Dict[str, int],
    "trim_ratio": 0.10,
    "feature_dim": 512
}
```

---

## 6.5 阶段 E：冲突组质心仲裁

对每个跨类冲突 SHA 组：

```text
candidate_labels = {c1, c2, ..., ck}
```

同一 SHA 下所有文件内容相同，因此只需使用其中任意一个对应特征。

记归一化后的冲突图片特征为：

\[
\tilde z
\]

### 6.5.1 候选类别相似度

仅计算候选类别质心：

\[
s_c = \tilde z^\top \mu_c,\quad c\in S
\]

排序后：

```text
top1_label
top1_similarity
top2_label
top2_similarity
margin = top1_similarity - top2_similarity
```

### 6.5.2 全局类别检查

同时计算图片与全部类别质心的相似度：

\[
g = \arg\max_c \tilde z^\top \mu_c
\]

要求：

```text
global_top1_label in candidate_labels
```

如果全局 Top-1 不属于候选集合，说明候选目录标签可能全部不可靠，默认拒绝该组。

### 6.5.3 绝对相似度阈值

绝对相似度受特征模型、归一化和数据集影响，不建议固定拍脑袋数值。

推荐根据所有冲突组的 `top1_similarity` 分布决定：

```text
similarity_threshold = top1_similarity 的第 10 百分位数
```

也可在实验中比较：

- 无绝对阈值；
- p10；
- p25。

### 6.5.4 margin 阈值

默认：

```text
margin_threshold = 0.02
```

接受条件：

\[
\text{accept}=
(s_1 \ge \tau_s)
\land
(s_1-s_2 \ge \tau_m)
\land
(g \in S)
\]

### 6.5.5 决策

#### 高置信度仲裁

满足全部条件：

- 在 `top1_label` 对应路径中保留一份；
- 删除其他候选类别中的副本；
- 标记为 `keep_resolved_conflict`；
- 训练权重设为 `0.5`。

#### 低置信度仲裁

任一条件不满足：

- 整个 SHA 组不进入监督训练；
- 标记为 `drop_unresolved_conflict`；
- 不随机选择标签；
- 不强制使用最大相似度标签。

---

## 7. 输出文件定义

## 7.1 `arbitration_results.csv`

每个跨类冲突组一行：

```csv
sha256,group_size,candidate_labels,top1_label,top1_similarity,top2_label,top2_similarity,margin,global_top1_label,global_top1_similarity,similarity_threshold,margin_threshold,decision,reason
```

`reason` 可取：

```text
accepted
low_margin
low_similarity
global_top1_outside_candidates
missing_centroid
invalid_feature
```

---

## 7.2 `clean_train_manifest.csv`

最终训练索引，每个保留样本一行：

```csv
relative_path,label,sample_weight,source,sha256
```

`source` 取值：

```text
unique
same_class_dedup
centroid_resolved
```

示例：

```csv
relative_path,label,sample_weight,source,sha256
0001/a.jpg,0001,1.0,unique,abc...
0012/c.jpg,0012,1.0,same_class_dedup,def...
0237/b.jpg,0237,0.5,centroid_resolved,ghi...
```

训练代码只读取该文件，不扫描整个原始目录。

---

## 7.3 `delete_list.txt`

包含所有不进入训练的相对路径：

```text
0001/b.jpg
0178/c.jpg
0342/d.jpg
```

该文件仅用于审计或创建软链接目录，不建议据此删除原始文件。

---

## 7.4 `dedup_stats.json`

至少包含：

```json
{
  "total_files": 103218,
  "unique_sha256": 101980,
  "same_class_duplicate_groups": 0,
  "same_class_duplicate_files_removed": 0,
  "cross_class_conflict_groups": 1032,
  "cross_class_conflict_files": 2095,
  "resolved_conflict_groups": 0,
  "abstained_conflict_groups": 0,
  "final_manifest_size": 0,
  "feature_shape": [103218, 512],
  "feature_sha256": "",
  "paths_file_sha256": "",
  "centroid_method": "trimmed_mean",
  "trim_ratio": 0.1,
  "margin_threshold": 0.02,
  "similarity_threshold": 0.0,
  "resolved_sample_weight": 0.5,
  "seed": 42
}
```

实际运行后填入真实值。

---

## 8. 训练代码接入

## 8.1 Dataset

训练 Dataset 不再通过 `ImageFolder` 自动扫描所有图片，而是读取 manifest。

示例接口：

```python
class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, root, manifest_csv, transform=None):
        self.root = Path(root)
        self.df = pandas.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = PIL.Image.open(self.root / row["relative_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        weight = float(row["sample_weight"])

        return {
            "image": image,
            "label": label,
            "sample_weight": weight,
            "path": row["relative_path"]
        }
```

---

## 8.2 加权损失

使用逐样本交叉熵：

```python
per_sample_loss = torch.nn.functional.cross_entropy(
    logits,
    labels,
    reduction="none"
)

loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
```

默认权重：

| 样本类型 | 权重 |
|---|---:|
| 唯一非冲突样本 | 1.0 |
| 同类去重后保留样本 | 1.0 |
| 高置信度质心仲裁样本 | 0.5 |
| 未解决冲突样本 | 0.0，不进入 Dataset |

---

## 9. 本地验证防泄漏方案

本地验证时不能使用验证集特征计算训练集质心。

正确顺序：

1. 先按 SHA 分组；
2. 按 SHA 进行 train/val 划分；
3. 同一个 SHA 的所有副本必须在同一 split；
4. validation 只保留非冲突唯一图片；
5. 仅使用 train split 内的非冲突唯一图片计算质心；
6. 仅仲裁 train split 内的冲突组；
7. validation 不参与质心计算、阈值选择或冲突重标注。

最终提交训练时，可以对官方全部训练集重新执行一次完整仲裁，再训练最终模型。

---

## 10. 共享给队友

共享以下内容：

```text
configs/dedup_clip_centroid.yaml
scripts/*.py
artifacts/dedup/clean_train_manifest.csv
artifacts/dedup/conflict_groups.csv
artifacts/dedup/arbitration_results.csv
artifacts/dedup/delete_list.txt
artifacts/dedup/dedup_stats.json
artifacts/dedup/dataset_fingerprint.txt
README.md
```

不要上传：

- 官方原始图片；
- 清洗后的图片副本；
- 测试集；
- 任何能够直接泄露官方数据的压缩包。

如果 `features.pt` 太大：

- 放到团队服务器共享目录；
- 或使用受控云盘；
- 仓库中只记录下载位置、shape、dtype 和 SHA-256；
- 不建议放公开 GitHub。

队友只需保证：

- 官方训练集路径结构一致；
- 数据集 fingerprint 一致；
- `features.pt` SHA-256 一致；
- manifest 中所有路径均存在。

---

## 11. 数据集 fingerprint

为避免三个人的数据版本不一致，生成：

```text
relative_path,file_sha256
```

按 `relative_path` 排序后拼接，再计算整体 SHA-256。

伪代码：

```python
lines = [
    f"{relative_path},{file_sha256}"
    for relative_path, file_sha256 in sorted(records)
]

fingerprint = sha256("\n".join(lines).encode("utf-8")).hexdigest()
```

输出：

```text
dataset_fingerprint.txt
```

队友的 fingerprint 必须一致。

---

## 12. 命令行接口设计

## 12.1 建立索引

```bash
python scripts/build_dataset_index.py \
  --dataset-root /path/to/train \
  --features-path /path/to/features.pt \
  --feature-paths /path/to/feature_paths.txt \
  --output-dir artifacts/dedup
```

## 12.2 分析重复组

```bash
python scripts/build_duplicate_groups.py \
  --index artifacts/dedup/dataset_index.csv \
  --output-dir artifacts/dedup
```

## 12.3 质心仲裁

```bash
python scripts/arbitrate_conflicts.py \
  --config configs/dedup_clip_centroid.yaml
```

## 12.4 验证产物

```bash
python scripts/verify_dedup_artifacts.py \
  --dataset-root /path/to/train \
  --artifact-dir artifacts/dedup
```

## 12.5 训练

```bash
python train.py \
  --data-root /path/to/train \
  --train-manifest artifacts/dedup/clean_train_manifest.csv \
  --epochs 50
```

---

## 13. 验收标准

## 13.1 数据索引验收

必须满足：

```text
dataset_index 行数 = 103218
feature 行数 = 103218
所有 relative_path 唯一
所有 feature_row 唯一
所有文件存在
所有 SHA-256 非空
所有特征有限
```

## 13.2 分组验收

必须满足：

```text
unique_sha256 = 101980
cross_class_conflict_groups = 1032
cross_class_conflict_files = 2095
最大 unique_label_count = 4
```

如果与已知统计不一致，停止后续处理并检查：

- 数据路径是否错误；
- 数据是否被修改；
- 文件是否缺失；
- 特征路径映射是否错位；
- 是否遗漏扩展名。

## 13.3 manifest 验收

必须满足：

1. 每个 SHA 最多只有一条保留记录；
2. 不存在同一 SHA 以多个不同标签进入训练；
3. `sample_weight > 0`；
4. 所有保留路径存在；
5. 所有删除路径不出现在最终 manifest；
6. 仲裁样本标签必须属于原始候选标签集合；
7. 未解决组的所有副本均不在最终 manifest；
8. 最终 manifest 可被 Dataset 完整加载。

## 13.4 质心验收

必须满足：

```text
centroid shape = [num_classes, 512]
所有质心有限
所有有效质心 L2 范数约等于 1
每个类别记录 raw_count 与 core_count
```

## 13.5 训练验收

至少跑通：

- 1 个 epoch；
- 一个完整 validation；
- 加权损失无 NaN；
- 数据加载无缺失路径；
- 输出样本数与 manifest 一致。

---

## 14. 推荐实验矩阵

保持模型、随机种子、训练轮数、学习率和数据增强完全一致。

| 实验 | 数据处理 | 仲裁规则 | 仲裁权重 |
|---|---|---|---:|
| D0 | 原始数据 | 无 | 1.0 |
| D1 | 同类去重，跨类冲突全部删除 | 无 | — |
| D2 | 普通均值质心 | 全部强制仲裁 | 1.0 |
| D3 | 10% trimmed 质心 | margin + 全局一致性拒绝 | 1.0 |
| D4 | 10% trimmed 质心 | margin + 全局一致性拒绝 | 0.5 |

优先级：

```text
D0 -> D1 -> D3 -> D4
```

D2 只作为消融，不建议作为最终方案。

所有实验使用：

```text
epochs = 50
same split
same seed
same head
same backbone freeze/unfreeze policy
same augmentation
same optimizer
same scheduler
```

重点记录：

- validation Top-1；
- 最佳 epoch；
- 50 epoch 最终 Top-1；
- resolved conflict groups；
- abstained conflict groups；
- 不同类别的删除比例；
- 质心仲裁样本在训练中的平均损失；
- 普通样本与仲裁样本的置信度差异。

---

## 15. 类别级风险检查

总体冲突比例约 2%，但可能集中在少数类别。

输出：

```csv
label,original_files,unique_clean_files,conflict_files,resolved_files,dropped_files,removal_ratio
```

定义：

\[
removal\_ratio_c
=
\frac{dropped\_files_c}{original\_files_c}
\]

重点检查：

- `removal_ratio > 0.10` 的类别；
- 清洗后样本数过少的类别；
- 大量成对冲突的类别组合；
- 类别质心样本数太少的类别；
- 仲裁结果过度集中到某些类别的情况。

若某类别质心样本数过少：

- 不使用该类别参与自动仲裁；
- 涉及该类别的冲突组直接拒绝；
- 或退回 D1 的“整组删除”策略。

---

## 16. 日志要求

每次执行必须记录：

```text
运行时间
Git commit
配置文件内容
数据集 fingerprint
features.pt SHA-256
feature_paths.txt SHA-256
随机种子
PyTorch 版本
Python 版本
CUDA 版本
设备信息
统计结果
```

建议将运行配置复制到：

```text
artifacts/dedup/run_config_resolved.yaml
```

---

## 17. 实现任务拆分

### Task 1：数据索引与 SHA 扫描

输出：

- `dataset_index.csv`
- `dataset_fingerprint.txt`

验收：

- 103,218 行；
- SHA 统计与已知结果一致。

### Task 2：重复组分析

输出：

- `duplicate_groups.csv`
- `conflict_groups.csv`
- 初步统计 JSON。

验收：

- 101,980 个唯一 SHA；
- 1,032 个跨类组；
- 2,095 个冲突文件。

### Task 3：特征对齐验证

输出：

- 特征 shape；
- 路径映射校验；
- 特征异常报告。

验收：

- 一一对应；
- 无重复 feature row；
- 无缺失路径。

### Task 4：鲁棒质心构建

输出：

- `centroids.pt`
- 类别质心统计。

验收：

- shape 正确；
- 质心归一化；
- 类别样本计数正确。

### Task 5：冲突仲裁

输出：

- `arbitration_results.csv`
- `clean_train_manifest.csv`
- `delete_list.txt`
- 完整 `dedup_stats.json`。

验收：

- 无跨类同 SHA 同时进入训练；
- 未解决组完全移除；
- 仲裁标签仅来自候选集合。

### Task 6：训练接入

输出：

- `ManifestDataset`
- 加权 CE；
- 训练命令。

验收：

- 1 epoch smoke test；
- 无 NaN；
- 数据数量一致。

### Task 7：消融实验

按 D0、D1、D3、D4 运行 50 epochs。

输出：

- 指标表；
- 最佳模型；
- 结论说明。

---

## 18. 最终推荐参数

第一版直接采用：

```yaml
centroid:
  method: trimmed_mean
  trim_ratio: 0.10
  min_class_samples_for_trim: 20

arbitration:
  candidate_labels_only: true
  require_global_top1_in_candidates: true
  margin_threshold: 0.02
  similarity_threshold_mode: percentile
  similarity_percentile: 10
  low_confidence_action: drop_entire_group

training:
  resolved_conflict_weight: 0.5
```

如果 D4 不如 D1：

- 说明原始 CLIP 特征对该细粒度任务区分能力不足；
- 最终采用 D1：同类去重 + 跨类冲突整组删除。

如果 D3 优于 D1，但 D4 不如 D3：

- 仲裁标签整体较可靠；
- 不需要降权，使用权重 1.0。

如果 D4 优于 D3：

- 仲裁样本仍有一定噪声；
- 保留 0.5 权重。

---

## 19. 最终原则

1. SHA-256 相同代表文件内容完全一致，可作为确定性重复依据；
2. 同类重复只保留一个，避免重复加权；
3. 跨类冲突不应以多个硬标签同时训练；
4. 质心仲裁只在原始候选类别内进行；
5. 不确定时允许拒绝，不强制选标签；
6. 仲裁标签默认低于普通标签可信度；
7. 所有处理通过 manifest 表达；
8. 原始数据保持只读；
9. 本地验证必须避免质心计算泄漏；
10. 共享脚本、清单和校验值，不共享官方数据副本。
