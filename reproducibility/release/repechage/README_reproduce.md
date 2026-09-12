# repechage 复现材料状态

这是工程交付源目录，不是已验收发布包。原始官方图像和训练权重不在仓库内分发。
选定配方状态见 selected_recipe_manifest.json；尚未填入的真实训练节点不会自动执行。
正式阶段数据、划分/校准政策及结果复现口径未齐备时，不能声称已完成该阶段复现。

使用仓库现有 `aegis_clip.cli.reproduce_stage` 审计完整的运行 manifest；默认不启动训练。
当前正式运行仍受来源闭合检查限制。恢复实验另用 outputs/stage_readiness 下的明确记录。
