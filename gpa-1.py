import itertools
from collections import Counter
import pulp
import math

# ============================================================
# 全局常量：成绩等级、GPA映射、单门课人数分布
# ============================================================
grades = ['C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
gpa = {
    'C-': 1.7, 'C': 2.0, 'C+': 2.3, 'B-': 2.7, 'B': 3.0,
    'B+': 3.3, 'A-': 3.7, 'A': 4.0, 'A+': 4.3
}

counts = {
    'C-': 4, 'C': 5, 'C+': 10, 'B-': 16, 'B': 20,
    'B+': 19, 'A-': 15, 'A': 8, 'A+': 3
}

# 单门课人数分布总和应为 100 (校验)
assert sum(counts.values()) == 100, "单门课人数分布总和必须为 100"

# 成绩组合缓存：key = 课程数量，value = profile 列表
_profiles_cache = {}


def _combo_count(n, k):
    """计算 C(n+k-1, k) 即 combinations_with_replacement 的数量。"""
    return math.comb(n + k - 1, k)


def _build_profiles(num_courses):
    """构建指定课程数量的所有成绩组合并缓存。

    参数
    ----
    num_courses : int
        每名学生修读的课程数量 K。

    返回
    ----
    list[dict] : 每个元素包含 combo, counts, avg_gpa
    """
    if num_courses in _profiles_cache:
        return _profiles_cache[num_courses]

    n_profiles = _combo_count(len(grades), num_courses)
    profiles = []
    for combo in itertools.combinations_with_replacement(grades, num_courses):
        sum_gpa = sum(gpa[g] for g in combo)
        avg_gpa = sum_gpa / num_courses
        profiles.append({
            'combo': combo,
            'counts': Counter(combo),
            'avg_gpa': avg_gpa,
        })
    _profiles_cache[num_courses] = profiles
    print(f"  [profile] 已生成 K={num_courses} 的成绩组合，共 {len(profiles)} 种 "
          f"(理论值 {n_profiles})")
    return profiles


# ============================================================
# 接口函数 a(min_avg_gpa, num_courses=5, total_students=100)
# ============================================================
def a(min_avg_gpa, num_courses=5, total_students=100):
    """
    以「平均 GPA >= min_avg_gpa」作为约束条件，求解最多有多少学生能达标
    （在所有课程的最优分配下）。

    参数
    ----
    min_avg_gpa : float
        平均绩点的最低要求（含），例如 3.5 表示均绩 >= 3.5 才算达标。
    num_courses : int
        每名学生修读的公选课数量 (默认 5)。
    total_students : int
        学生总人数 (默认 100)。成绩分布按比例缩放。

    返回
    ----
    result : dict
        {
            'status':           求解状态字符串 (如 'Optimal'),
            'threshold':        传入的 min_avg_gpa,
            'num_courses':      课程数量 K,
            'max_students':     达标的最大学生人数 (int),
            'max_ratio':        达标学生占总人数的比例 (float),
            'total_students':   学生总人数,
            'ceiling':          理论天花板比例 (仅与阈值和分布均值有关),
            'assignment':       最优分配方案 list[dict]，
                               每个元素包含: num_students, avg_gpa, combo, success
        }
    """
    # 按比例缩放成绩分布
    scale = total_students / 100.0
    scaled_counts = {g: int(round(counts[g] * scale)) for g in grades}
    # 修正舍入误差：微调最大档使总和等于 total_students
    diff = total_students - sum(scaled_counts.values())
    if diff != 0:
        max_grade = max(grades, key=lambda g: scaled_counts[g])
        scaled_counts[max_grade] += diff
    assert sum(scaled_counts.values()) == total_students, \
        f"缩放后成绩分布总和应为 {total_students}，实际 {sum(scaled_counts.values())}"

    profiles = _build_profiles(num_courses)
    total_slots = {g: num_courses * scaled_counts[g] for g in grades}

    # 根据阈值标注每条组合是否达标
    for p in profiles:
        p['success'] = 1 if p['avg_gpa'] >= min_avg_gpa else 0

    # ---- 创建整数规划模型 ----
    prob = pulp.LpProblem("Maximize_Successful_Students", pulp.LpMaximize)

    # 决策变量：选择每种组合的学生人数 (非负整数)
    x = pulp.LpVariable.dicts("profile", range(len(profiles)), lowBound=0, cat='Integer')

    # 目标函数：最大化达标学生人数
    prob += pulp.lpSum(x[i] * profiles[i]['success'] for i in range(len(profiles)))

    # 约束条件 1：总人数等于 total_students
    prob += pulp.lpSum(x[i] for i in range(len(profiles))) == total_students

    # 约束条件 2：每档成绩消耗总量必须刚好等于 K 门课的总和
    for g in grades:
        prob += pulp.lpSum(x[i] * profiles[i]['counts'][g] for i in range(len(profiles))) == total_slots[g]

    # ---- 求解 ----
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # ---- 收集结果 ----
    assignment = []
    for i in range(len(profiles)):
        num_students = int(x[i].varValue)
        if num_students > 0:
            info = profiles[i]
            assignment.append({
                'num_students': num_students,
                'avg_gpa': info['avg_gpa'],
                'combo': info['combo'],
                'success': bool(info['success']),
            })

    max_students = int(pulp.value(prob.objective))

    # 理论天花板 (与 N 无关)
    G_bar = 3.061
    G_min = 1.7
    if min_avg_gpa > G_min:
        ceiling = min(1.0, (G_bar - G_min) / (min_avg_gpa - G_min))
    else:
        ceiling = 1.0

    return {
        'status': pulp.LpStatus[prob.status],
        'threshold': min_avg_gpa,
        'num_courses': num_courses,
        'max_students': max_students,
        'max_ratio': max_students / total_students,
        'total_students': total_students,
        'ceiling': ceiling,
        'assignment': assignment,
    }


# ============================================================
# 扫描函数：分析达标比例随课程数量变化的趋势
# ============================================================
def sweep_courses(threshold, max_courses=8, min_courses=1, total_students=100):
    """
    对给定的阈值，扫描课程数量从 min_courses 到 max_courses，
    计算每个 K 下的最优达标比例。

    参数
    ----
    threshold : float
        平均 GPA 阈值 (如 3.5)
    max_courses : int
        最大课程数 (默认 8，注意 K 越大组合数指数增长)
    min_courses : int
        最小课程数 (默认 1)
    total_students : int
        学生总人数 (默认 100)

    返回
    ----
    list[dict] : 每个元素为 a() 的返回值
    """
    results = []
    for k in range(min_courses, max_courses + 1):
        print(f"\n>>> 求解 K={k} (组合数 ~{_combo_count(len(grades), k)}) ...")
        result = a(threshold, num_courses=k, total_students=total_students)
        results.append(result)
        print(f"    状态={result['status']}, "
              f"达标={result['max_students']}/{total_students} ({result['max_ratio']:.2%})"
              f"  [天花板={result['ceiling']:.2%}]")
    return results


def sweep_students(threshold, num_courses=5, student_sizes=None):
    """
    对给定的阈值和课程数，扫描学生总人数 N 的变化，
    观察整数粒度对达标比例的影响。

    参数
    ----
    threshold : float
        平均 GPA 阈值
    num_courses : int
        课程数量 K
    student_sizes : list[int] | None
        要扫描的学生人数列表，默认 [10, 20, 50, 100, 200, 500, 1000]

    返回
    ----
    list[dict] : 每个元素为 a() 的返回值
    """
    if student_sizes is None:
        student_sizes = [10, 20, 50, 100, 200, 500, 1000]
    results = []
    for n in student_sizes:
        print(f"\n>>> 求解 N={n} (K={num_courses}, 阈值={threshold}) ...")
        result = a(threshold, num_courses=num_courses, total_students=n)
        results.append(result)
        print(f"    状态={result['status']}, "
              f"达标={result['max_students']}/{n} ({result['max_ratio']:.2%})"
              f"  [天花板={result['ceiling']:.2%}]")
    return results


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    # ---- Part 1: 单点验证 ----
    print("=" * 60)
    print("Part 1: 单点验证 (N=100, K=5, 阈值=3.5)")
    print("=" * 60)
    r = a(3.5, num_courses=5, total_students=100)
    print(f"状态: {r['status']}, 达标: {r['max_students']}/100 ({r['max_ratio']:.2%})"
          f"  [天花板={r['ceiling']:.2%}]")

    # ---- Part 2: 扫描课程数 K=1..10, N=100 ----
    print("\n" + "=" * 60)
    print("Part 2: 扫描 K=1..10, N=100, 阈值=3.5")
    print("=" * 60)
    results_k = sweep_courses(threshold=3.5, max_courses=10, min_courses=1, total_students=100)

    print("\n" + "=" * 60)
    print("汇总: 达标比例随课程数 K 的变化 (N=100, 阈值=3.5)")
    print("=" * 60)
    print(f"{'K':>3s}  {'组合数':>8s}  {'达标':>6s}  {'比例':>8s}  {'天花板':>8s}  "
          f"{'距天花板':>8s}  {'状态':>10s}")
    print("-" * 55)
    for r in results_k:
        n_combos = _combo_count(len(grades), r['num_courses'])
        gap = r['ceiling'] - r['max_ratio']
        print(f"{r['num_courses']:3d}  {n_combos:8d}  "
              f"{r['max_students']:4d}/{r['total_students']:<3d}  "
              f"{r['max_ratio']:7.2%}  {r['ceiling']:7.2%}  {gap:7.2%}  "
              f"{r['status']:>10s}")

    # ---- Part 3: 扫描学生人数 N (固定 K=5, 阈值=3.5) ----
    print("\n\n" + "=" * 60)
    print("Part 3: 扫描学生人数 N (K=5, 阈值=3.5)")
    print("=" * 60)
    results_n = sweep_students(threshold=3.5, num_courses=5,
                               student_sizes=[10, 20, 50, 100, 200, 500, 1000])

    print("\n" + "=" * 60)
    print("汇总: 达标比例随学生人数 N 的变化 (K=5, 阈值=3.5)")
    print("=" * 60)
    print(f"{'N':>5s}  {'达标':>8s}  {'比例':>8s}  {'天花板':>8s}  "
          f"{'距天花板':>8s}  {'状态':>10s}")
    print("-" * 55)
    for r in results_n:
        gap = r['ceiling'] - r['max_ratio']
        print(f"{r['total_students']:5d}  "
              f"{r['max_students']:4d}/{r['total_students']:<4d}  "
              f"{r['max_ratio']:7.2%}  {r['ceiling']:7.2%}  {gap:7.2%}  "
              f"{r['status']:>10s}")