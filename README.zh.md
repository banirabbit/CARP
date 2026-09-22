# CARP

CARP 是面向固定候选 LLM 编排服务集合的查询感知路由方法。当前仓库仅保留正式主方法实现：冻结查询编码器的 uplift scorer、查询级 DynCost，以及验证集校准的 P3 Pareto-compromise 路由策略。

英文使用说明、标准 JSONL 数据格式和训练/推理命令见 [README.md](README.md)。数据、检查点、运行产物和改造前旧仓库都在 `.gitignore` 中；旧代码已保留在本地 `legacy_original/` 目录，便于追溯但不会进入后续提交。
