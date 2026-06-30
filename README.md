# GPA Optimization

这个仓库整理的是一个“固定成绩分布下，如何分配成绩使尽量多学生达到目标平均绩点”的优化问题。

当前重点是三专业场景：三个专业人数不同，共享一部分课程，也各自有专业课。精确 MILP 模型规模较大，CBC 求解器可能长时间卡在分支定界阶段，所以仓库里同时保留了精确模型和更实用的启发式搜索/局部 MILP 精修流程。

## 文件说明

- `PROBLEM.md`：实际问题定义，独立于代码，说明人数、课程池、成绩分布和优化目标。
- `gpa_ceiling_proof.md`：只基于总绩点资源得到的理论上界。
- `gpa-three-major.py`：三专业问题的全局精确 MILP。
- `gpa-three-major-heuristic.py`：启发式搜索，加按专业拆分的小 MILP 精修。
- `gpa.py`、`gpa-1.py`、`gpa-fragmented.py`：早期或简化实验代码，保留作参考。
- `requirements.txt`：Python 依赖。

## 环境

测试环境为 Python 3.11 和 PuLP 3.3.2。PuLP 通常会自带 CBC 可执行文件，不需要额外安装求解器。

```powershell
conda create -n gpa-opt python=3.11
conda activate gpa-opt
pip install -r requirements.txt
```

## 运行示例

快速启发式：

```powershell
python .\gpa-three-major-heuristic.py --thresholds 3.5 --mode greedy --search 20 --scale 1.2
```

启发式搜索加 MILP 精修：

```powershell
python .\gpa-three-major-heuristic.py --thresholds 3.5 --mode swap --search 120 --scale 1.8 --local-rounds 8 --local-beam 12 --local-mutations 24 --swap-size 4 --refine-top 10 --time-limit 30 --jobs 3
```

全局精确 MILP，主要用于验证或小时间预算实验：

```powershell
python .\gpa-three-major.py --threshold 3.5 --time-limit 120 --gap-rel 0.02 --threads 8
```

## 如何理解结果

当阈值为 `3.5` 时，只看总绩点资源得到的理论上界是 `75.61%`。这是乐观上界，不代表一定可达，因为它忽略了课程池共享结构、整数成绩人数、不同专业课程数差异等限制。

全局精确 MILP 可能长时间没有整数可行解。实际探索建议先跑 `gpa-three-major-heuristic.py`，找到较好的共享课程分配后，再对候选方案做小规模 MILP 精修。

如果 CBC 返回 `Not Solved`，结果需要谨慎解读。启发式脚本已经加入整数性检查，避免把 LP 松弛的分数解误当成真实可行整数解。
