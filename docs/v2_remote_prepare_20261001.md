# V2_REMOTE_PREPARE_20261001：新CUDA服务器准备与s1交接

用户于2026-10-01指定 `connect.bjb1.seetacloud.com:39385` 准备v2后续实验。
本段是数据、仓库和环境准备；没有启动训练、成本probe、推理或平台上传。
队友报告s1已完成，本会话尚未取得s1实际产物，后续训练仍待父权重交接与校验。

## 目录与实际环境

| 用途 | 新服务器路径 |
|---|---|
| 仓库，独立准备分支 | `/root/autodl-tmp/noise/repo` |
| 共享只读原始数据 | `/root/autodl-tmp/noise/data/{train,test}` |
| 官方CLIP权重 | `/root/autodl-tmp/noise/assets/clip/ViT-B-32.pt` |
| 当前阶段审计与固定划分 | `/root/autodl-tmp/noise/assets/stages/repechage/20260921` |
| 独立Python环境 | `/root/autodl-tmp/noise/venv` |
| 本段CPU预检与报告 | `/root/autodl-tmp/noise/runs/codex/v2_remote_prepare_20261001` |
| s1交接暂存目录 | `/root/autodl-tmp/noise/handoff/s1_pending` |

实测RTX4090，24,564MiB显存；准备时GPU利用率0%、无计算进程。
数据盘限额50GiB。环境为Python3.12.3、torch2.12.1+cu130、torchvision0.27.1+cu130，
CUDA可用并支持BF16。虚拟环境复用镜像内已有Torch，仅在自己的venv内补齐依赖。
这不是 `pyproject.toml` 锁定环境，也不是队友Windows/Torch2.6环境；本段只验证兼容性，
不声称跨后端逐位复现。实际版本及官方CLIP代码commit见
[环境清单](../results/v2_remote_prepare_20261001/environment.json)和
[实际依赖](../results/v2_remote_prepare_20261001/requirements.actual.txt)。

创建独立环境及安装依赖的实际命令：

```bash
/root/miniconda3/bin/python -m venv --system-site-packages /root/autodl-tmp/noise/venv
/root/autodl-tmp/noise/venv/bin/python -m pip install --no-cache-dir \
  ftfy==6.3.1 pytest==9.1.0 pandas==2.3.3 PyYAML==6.0.3 scikit-learn==1.7.2 \
  git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
```

OpenAI CLIP代码固定 `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`；
官方ViT-B/32权重SHA256为
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。
独立assets目录只接收本阶段审计和官方权重，recipe引用当前阶段数据。
仓库历史输出仍在Git中；旧阶段拟合产物不得作为本次训练输入。

## 仓库与数据准备

开工fetch并检查全部分支、近期main及工作目录；找到队友
`xjn/v2_windows_portability_20261001`，commit `831ca2183b342fae8ebd0eff9e00a3beb07ee14c`。
将其分隔符无关血缘校验、`sys.executable`修复及s1交接记录合并到独立方案分支；
不重复实现。原main工作区已有改动，保留原样，方案使用
`codex/v2_remote_prepare_20261001` 和本机独立worktree。

服务器GitHub直接克隆过慢，改用本机Git bundle克隆相同提交历史，再将origin设回GitHub。
最初网络克隆目录留档为 `/root/autodl-tmp/noise/repo_network_attempt_20261001`。
方案基点main为 `7b3bd4686c327e3dabab4e6a8a2b11a0c06a403f`；
包含队友修复的初始服务器commit为 `8b54be43ffa5006e7340c86f742321ed43367672`。
最终提交、push和main合并状态由本段交付消息及Git历史记录。

本机生成bundle并传入服务器后执行：

```bash
git bundle create /tmp/noise-v2-remote-prepare-20261001.bundle \
  codex/v2_remote_prepare_20261001 main refs/remotes/origin/xjn/v2_windows_portability_20261001
# 以下两条在服务器执行
git clone --branch codex/v2_remote_prepare_20261001 \
  /root/autodl-tmp/noise/assets/noise-v2-remote-prepare-20261001.bundle /root/autodl-tmp/noise/repo
git -C /root/autodl-tmp/noise/repo remote set-url origin \
  https://github.com/Luqhhh/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning.git
```

当前阶段官方训练集148,695张、750类，测试集37,444张，原始文件共32,492,515,836字节。
仅复制 `artifacts/stages/repechage/20260921` 的审计和清单，排除其 `features/`。
保留审计文件字节及固定133,815/14,880内容组隔离划分，不改标签或重分组。
审计metadata内原机器路径作为历史来源保留；服务器实际路径由独立recipe明确指定。

整目录rsync在大文件清单阶段反复断开，改为81批、每批原始文件不超过384MiB的tar归档。
每批用rsync复制单个归档，解压后删除本段临时归档；已完成批次按状态记录复用。
后续改为两路并行，各批临时文件独立；避免将约30GB ZIP和解压数据同时保留。
数据传输只产生当前阶段原始图，不修改本机源数据。
完整性以[逐文件资产校验](../results/v2_remote_prepare_20261001/assets_validation.json)为准。
该检查读取所有图的文件大小和SHA256，对照官方manifest，核对覆盖、类别数、
审计文件摘要、数据集fingerprint和官方权重；不重复解码或拟合测试图。
实测186,139张全部一致，缺失、额外文件及摘要失配均为0，750类和两个fingerprint通过；
全量校验用时37.60秒。数据、官方权重及阶段审计目录已递归设为只读。
81批的成功传输驱动累计63.10分钟，时间口径不含此前失败的整目录传输试验；
见[传输摘要](../results/v2_remote_prepare_20261001/transfer_summary.json)和
[校验原始日志](../results/v2_remote_prepare_20261001/assets_validation.log)。

## 已验证的工程结果与复现入口

本机及服务器相同CPU测试各67 passed；服务器原始测试输出见
[测试记录](../results/v2_remote_prepare_20261001/tests_server.log)。
prepare/verify通过，固定split为133,815/14,880，内容组交集0；所有runtime依赖可导入。
CPU官方模型前向通过：224原生与插值最大差0.0；384/448/576均为 `[1,750]`，
88,233,966个训练参数全为FP32。见[CPU前向报告](../results/v2_remote_prepare_20261001/cpu_forward.json)。
真实图片的双worker加载检查通过：5张训练图、448输入、含已知坏EXIF图，全部张量有限。
见[数据加载报告](../results/v2_remote_prepare_20261001/loader_smoke.json)。
这些结果不是识别分数、GPU训练吞吐或平台提升。

[服务器recipe](../configs/v2/remote_39385.recipe.json)只改四个资源路径；
micro batch2、workers2、所有训练超参数及 `execution_authorized=false` 保持原固定配置。
`preflight_template` 是禁用训练的环境检查模板，**不是接纳Windows s1的续训plan**。
其plan SHA256为 `ccb74ad8848b0b32897929cb1b3f50d0fea6bfbe6e18f8ea407d2712375ceee0`。
在服务器仓库根目录执行：

```bash
source scripts/activate_v2_remote_39385.sh
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_v2.py tests/test_v2_swa.py reproducibility/aegis_f1/tests/test_model.py -q
python -m v2.plan verify --plan /root/autodl-tmp/noise/runs/codex/v2_remote_prepare_20261001/preflight_template/plan.json
CUDA_VISIBLE_DEVICES='' python scripts/check_v2_cpu.py --recipe configs/v2/remote_39385.recipe.json --output <new_cpu_report.json>
python scripts/verify_remote_data_assets.py \
  --data-root /root/autodl-tmp/noise/data \
  --stage-artifacts /root/autodl-tmp/noise/assets/stages/repechage/20260921 \
  --official-checkpoint /root/autodl-tmp/noise/assets/clip/ViT-B-32.pt \
  --official-sha256 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af \
  --report <new_assets_report.json>
```

不要在已存在目录再次prepare；新检查使用新输出目录，保护不可变plan。
本段首次prepare的实际命令为
`python -m v2.plan prepare --recipe configs/v2/remote_39385.recipe.json --output /root/autodl-tmp/noise/runs/codex/v2_remote_prepare_20261001/preflight_template`。

## s1交接与停止边界

队友文档记录13,930次更新全部完成、best为epoch9 raw，val micro/macro约75.15%/74.23%；
来源是[队友实测记录](v2_local_windows_20261001.md)，本段未拿到checkpoint重载验证。
用户明确回复“暂时还没给我”。取得产物前不伪造binding、不重跑s1、不启动s2。

后续至少需要原workspace的 `plan.json`、recipe、四段生成配置、
`train_manifest.csv`、`split.json`，以及 `runs/s1_384/` 的
`best.pt`、`best.binding.json`、`status.json` 和原训练日志/授权；建议同时保留last及其sidecar。
保留原文件字节，先验证原stage完成态、模型/EMA、选择状态、清单和更新数，
再处理Windows路径与新目录之间的迁移。当前校验把绝对 `paths.project_root`
写进stage配置摘要，直接在新目录prepare会使父checkpoint失配；不能通过修改SHA或
删掉校验来绕过。实际迁移方案须依据拿到的原产物制定。

本段没有新增CSV/ZIP，不宣称v2新候选完成。现役full v1 SWA包为
`/home/lux1/noise/worktrees/v1_full_swa_20260930/outputs/codex/v1_full_swa_20260930/submission/{pred_results.csv,submission.zip}`。
本段复核ZIP SHA256仍为
`1a2bc8472f9c813e24781e798c233aba284df84380b8e76ef144f584e830505e`；
37,444行、9项通过及ZIP/CSV一致性的既有证据见
[现役包复核](../results/v1_full_swa_platform_20261001/artifact_verification.json)和
[正式提交校验](../results/v1_full_swa_platform_20261001/submission_check.log)。

用户本次授权覆盖新服务器准备；后续训练、GPU成本probe、推理和平台上传未执行。
用户随后明确要求准备完成后关闭服务器。本段将在数据校验、仓库同步及报告交付完成后
通过镜像提供的 `/usr/bin/shutdown` 关机，等待s1产物交接；实际关机结果记录于本段报告。
已只读确认该入口通过终止supervisord执行关闭，准备期间supervisord进程存在；
不使用本容器内无效的systemctl关机命令。
