"""
gpa-three-major.py — 三专业多层交叉 MILP
总 300 人: G1(专业一)=150, G2(专业二)=100, G3(专业三)=50

Pool 结构:
  0: 公选 10门 × 300人
  1: 一二共修 4门 × 250人
  2: 二三共修 3门 × 150人
  3: 专业课一 10门 × 150人
  4: 专业课二 7门 × 100人
  5a: 专业课三-1 7门 × 50人  (原 Pool 5 k=15 拆分为 7+8)
  5b: 专业课三-2 8门 × 50人

每人修读: G1=24门(0+1+3), G2=24门(0+1+2+4), G3=28门(0+2+5a+5b)

求解方法: 层次联合变量 (hierarchical joint variables)
  SDP (sum discretization precision): 0.1 (所有 sum 四舍五入到一位小数)
"""

import itertools
import math
from collections import Counter, defaultdict
import pulp

# ── 常量 ──
grades = ['C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
gpa_map = {g: v for g, v in zip(grades, [1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 4.3])}
base = {'C-': 4, 'C': 5, 'C+': 10, 'B-': 16, 'B': 20, 'B+': 19, 'A-': 15, 'A': 8, 'A+': 3}
TOTAL_STUDENTS = 300

# ── Pool 定义 ──
POOLS = {
    0:  {'name': '公选',      'k': 10, 'n': 300, 'groups': [0, 1, 2]},
    1:  {'name': '一二共修',  'k': 4,  'n': 250, 'groups': [0, 1]},
    2:  {'name': '二三共修',  'k': 3,  'n': 150, 'groups': [1, 2]},
    3:  {'name': '专业课一',  'k': 10, 'n': 150, 'groups': [0]},
    4:  {'name': '专业课二',  'k': 7,  'n': 100, 'groups': [1]},
    5:  {'name': '专业课三a', 'k': 7,  'n': 50,  'groups': [2]},
    6:  {'name': '专业课三b', 'k': 8,  'n': 50,  'groups': [2]},
}

# ── 群体定义 ──
GROUPS = [
    {'name': 'G1(专业一)', 'size': 150, 'pools': [0, 1, 3],     'nc': 24},
    {'name': 'G2(专业二)', 'size': 100, 'pools': [0, 1, 2, 4],  'nc': 24},
    {'name': 'G3(专业三)', 'size': 50,  'pools': [0, 2, 5, 6],  'nc': 28},
]

SDP = 0.1  # sum discretization precision


def rnd(x):
    """四舍五入到 SDP 精度"""
    return round(round(x / SDP) * SDP, 10)


def sc(n):
    """按比例分配 n 人的成绩分布"""
    s = {g: int(round(base[g] * n / 100.0)) for g in grades}
    d = n - sum(s.values())
    if d:
        mg = max(grades, key=lambda g: s[g])
        s[mg] += d
    return s


# ── 预处理: 每个 pool 的 combo 枚举 ──
pool_combos = {}   # pid -> list of {idx, combo, sum_gpa, counts}
pool_sums = {}     # pid -> sorted list of unique rounded sums
pool_si = {}       # pid -> {sum_val: index}
pool_cbs = {}      # pid -> {sum_val: [combo_indices]}
pool_slots = {}    # pid -> {grade: count}

for pid, p in POOLS.items():
    k = p['k']
    slots = sc(p['n'])
    pool_slots[pid] = {g: k * slots[g] for g in grades}

    cb = []
    ss = set()
    sm = defaultdict(list)
    for combo in itertools.combinations_with_replacement(grades, k):
        sg = rnd(sum(gpa_map[g] for g in combo))
        idx = len(cb)
        cb.append({'idx': idx, 'combo': combo, 'sum_gpa': sg, 'counts': Counter(combo)})
        ss.add(sg)
        sm[sg].append(idx)
    pool_combos[pid] = cb
    sv = sorted(ss)
    pool_sums[pid] = sv
    pool_si[pid] = {s: i for i, s in enumerate(sv)}
    pool_cbs[pid] = dict(sm)

# 每组在各池的 pool_groups index
pool_groups = defaultdict(list)
for gid, g in enumerate(GROUPS):
    for pid in g['pools']:
        pool_groups[pid].append(gid)


def _build_layer_sums(prev_sums, new_pool_id):
    """给定上一层 sum 列表和新区 pool_id, 返回新 sum 列表和 (prev,new)→new_idx 映射"""
    new_sums = pool_sums[new_pool_id]
    combined = set()
    p2n = {}
    for ps in prev_sums:
        for ns in new_sums:
            cs = rnd(ps + ns)
            combined.add(cs)
            p2n[(ps, ns)] = cs
    cv = sorted(combined)
    csi = {s: i for i, s in enumerate(cv)}
    return cv, csi, p2n


def build_group_joint_vars(gid, prob):
    """
    为 group gid 构建层次联合变量。
    返回:
      y_layers: list of dict, 每层: {(prev_sum, cur_sum) -> var} 或 {key -> var}
      y_flat:   list of all y vars (for size constraint)
      y_info:   额外信息
    """
    g = GROUPS[gid]
    pids = g['pools']
    n_layers = len(pids)

    # 第一层: pool0 的 sum 值
    s0 = pool_sums[pids[0]]
    si0 = pool_si[pids[0]]

    if n_layers == 1:
        # Unlikely but handle
        y_flat = []
        y_layer0 = {}
        for a in s0:
            var = pulp.LpVariable(f"y_g{gid}_l0_{si0[a]}", 0, None, 'Integer')
            y_layer0[(a,)] = var
            y_flat.append(var)
        return [y_layer0], y_flat, {'pids': pids, 'sums': [s0], 'sis': [si0],
                                     'layer_sums': [s0], 'layer_si': [{s: i for i, s in enumerate(s0)}]}

    # 多层: 逐层构建
    y_layers = []
    all_layer_sums = [s0]  # 每层的 prev_sum 列表
    all_layer_si = [{s: i for i, s in enumerate(s0)}]
    y_all_flat = []

    # Layer 1: pool0 × pool1
    s1 = pool_sums[pids[1]]
    si1 = pool_si[pids[1]]
    y_l1 = {}
    for a in s0:
        for b in s1:
            var = pulp.LpVariable(f"y_g{gid}_l1_{si0[a]}_{si1[b]}", 0, None, 'Integer')
            y_l1[(a, b)] = var
            y_all_flat.append(var)
    y_layers.append(y_l1)

    if n_layers == 2:
        # y_l1 is the final joint
        return y_layers, y_all_flat, {'pids': pids, 'sums': [s0, s1], 'sis': [si0, si1],
                                       'layer_sums': [s0, s1], 'final_layer': 0}

    # Build prev_sum for layer 2
    prev_sums_l2 = sorted(set(rnd(a + b) for a in s0 for b in s1))
    prev_si_l2 = {s: i for i, s in enumerate(prev_sums_l2)}
    all_layer_sums.append(prev_sums_l2)
    all_layer_si.append(prev_si_l2)

    # Map prev_sum → list of (a,b) that produce it
    prev_to_ab = defaultdict(list)
    for a in s0:
        for b in s1:
            prev_to_ab[rnd(a + b)].append((a, b))

    # Layer 2 onward
    cur_prev_sums = prev_sums_l2
    cur_prev_si = prev_si_l2
    prev_y_dict = y_l1  # reference to previous layer for linking
    prev_pool_idx_start = 2  # which pool index the previous sum represents (i.e., sum of pools [0..prev_pool_idx_start-1])

    for li in range(2, n_layers):
        cur_pid = pids[li]
        sn = pool_sums[cur_pid]
        sin = pool_si[cur_pid]

        y_cur = {}
        p2n_map = {}  # (prev_sum, cur_sum) → new total

        new_sums_set = set()
        for ps in cur_prev_sums:
            for ns in sn:
                cs = rnd(ps + ns)
                new_sums_set.add(cs)
                p2n_map[(ps, ns)] = cs
        new_sums_sorted = sorted(new_sums_set)
        new_si = {s: i for i, s in enumerate(new_sums_sorted)}

        # Build y_cur vars and link to prev layer
        for ps in cur_prev_sums:
            for ns in sn:
                var = pulp.LpVariable(
                    f"y_g{gid}_l{li}_{cur_prev_si[ps]}_{sin[ns]}", 0, None, 'Integer')
                y_cur[(ps, ns)] = var

        # Link: sum_{ns} y_cur[(ps, ns)] = sum_{a+b=ps (or prev key = ps)} prev_layer entries
        if li == 2:
            # prev layer is y_l1 (pool0+pool1), keyed by (a,b)
            for ps in cur_prev_sums:
                lhs_terms = [y_cur[(ps, ns)] for ns in sn if (ps, ns) in y_cur]
                rhs_terms = [prev_y_dict[(a, b)] for (a, b) in prev_to_ab.get(ps, [])]
                if lhs_terms:
                    if rhs_terms:
                        prob += pulp.lpSum(lhs_terms) == pulp.lpSum(rhs_terms), \
                            f"lk_g{gid}_l{li}_ps{cur_prev_si[ps]}"
                    else:
                        prob += pulp.lpSum(lhs_terms) == 0, \
                            f"lkz_g{gid}_l{li}_ps{cur_prev_si[ps]}"
                elif rhs_terms:
                    prob += pulp.lpSum(rhs_terms) == 0, \
                        f"lkz2_g{gid}_l{li}_ps{cur_prev_si[ps]}"
        else:
            # prev layer is y_prev (keyed by (prev_prev_sum, prev_cur_sum))
            # need to map prev_prev_sum → list of keys in prev_y_prev
            prev_prev_to_keys = defaultdict(list)
            for (pps, pcs), var in prev_y_dict.items():
                prev_prev_to_keys[pps].append((pps, pcs))

            for ps in cur_prev_sums:
                lhs_terms = [y_cur[(ps, ns)] for ns in sn if (ps, ns) in y_cur]
                # ps here is the TOTAL sum of all previous pools.
                # In prev layer, the prev sum was (prev_prev_sum=A) and cur_sum=B, total=A+B.
                # We need to find all prev_y_dict entries where A+B = ps
                # This requires summing over all A where (A, ps-A) exists
                rhs_parts = []
                for pps in all_layer_sums[li - 1]:
                    pcs = rnd(ps - pps)
                    if (pps, pcs) in prev_y_dict:
                        rhs_parts.append(prev_y_dict[(pps, pcs)])

                if lhs_terms:
                    if rhs_parts:
                        prob += pulp.lpSum(lhs_terms) == pulp.lpSum(rhs_parts), \
                            f"lk_g{gid}_l{li}_ps{cur_prev_si[ps]}"
                    else:
                        prob += pulp.lpSum(lhs_terms) == 0, \
                            f"lkz_g{gid}_l{li}_ps{cur_prev_si[ps]}"
                elif rhs_parts:
                    prob += pulp.lpSum(rhs_parts) == 0, \
                        f"lkz2_g{gid}_l{li}_ps{cur_prev_si[ps]}"

        y_layers.append(y_cur)
        prev_y_dict = y_cur

        if li < n_layers - 1:
            cur_prev_sums = new_sums_sorted
            cur_prev_si = new_si
            all_layer_sums.append(new_sums_sorted)
            all_layer_si.append(new_si)

    return y_layers, y_all_flat, {
        'pids': pids,
        'sums': [pool_sums[pid] for pid in pids],
        'sis': [pool_si[pid] for pid in pids],
        'layer_sums': all_layer_sums,
        'layer_si': all_layer_si,
        'final_layer': len(y_layers) - 1,
    }


def solve(threshold=3.5, time_limit=600, verbose=True):
    prob = pulp.LpProblem("ThreeMajor", pulp.LpMaximize)

    # ── w 变量 ──
    w = {}
    for pid in POOLS:
        w[pid] = {}
        for gid in pool_groups[pid]:
            n_cb = len(pool_combos[pid])
            w[pid][gid] = [
                pulp.LpVariable(f"w_p{pid}_g{gid}_c{i}", 0, None, 'Integer')
                for i in range(n_cb)
            ]

    # ── y 层次联合变量 ──
    all_y_info = {}
    all_y_flat = {}
    for gid in range(len(GROUPS)):
        y_layers, y_flat, info = build_group_joint_vars(gid, prob)
        all_y_info[gid] = (y_layers, info)
        all_y_flat[gid] = y_flat

    # ── 每组总人数 ──
    for gid, g in enumerate(GROUPS):
        prob += pulp.lpSum(all_y_flat[gid]) == g['size'], f"sz_g{gid}"

    # ── w ↔ y 边际耦合 ──
    for gid, g in enumerate(GROUPS):
        pids = g['pools']
        y_layers, info = all_y_info[gid]
        final_li = info['final_layer']
        final_layer = y_layers[final_li]

        # Build mapping from pool sum → list of y entries
        # For each pool index in this group, find which y layer and dimension contains it
        n_pools = len(pids)

        # For each pool, find all entries in final_layer that have the right sum
        for pi, pid in enumerate(pids):
            s_list = pool_sums[pid]
            s_idx_map = pool_si[pid]
            cbs = pool_cbs[pid]

            for s_val, s_idx in s_idx_map.items():
                c_indices = cbs.get(s_val, [])
                lhs = pulp.lpSum(w[pid][gid][ci] for ci in c_indices) if c_indices else 0

                # Find y entries where this pool's contribution = s_val
                rhs_terms = []

                if n_pools == 1:
                    # Single pool, y is in layer 0
                    y_layer = y_layers[0]
                    for key, var in y_layer.items():
                        if abs(key[0] - s_val) < 0.001:
                            rhs_terms.append(var)
                elif n_pools == 2:
                    # Two pools, y_l1 keyed by (a, b)
                    y_layer = y_layers[0]
                    for (a, b), var in y_layer.items():
                        if pi == 0 and abs(a - s_val) < 0.001:
                            rhs_terms.append(var)
                        elif pi == 1 and abs(b - s_val) < 0.001:
                            rhs_terms.append(var)
                elif n_pools == 3:
                    # Three pools: y_l1 + y_l2
                    y_l1 = y_layers[0]
                    y_l2 = y_layers[1]
                    if pi == 0:
                        # pool 0: sum over y_l1 entries where a = s_val
                        for (a, b), var in y_l1.items():
                            if abs(a - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 1:
                        # pool 1: sum over y_l1 entries where b = s_val
                        for (a, b), var in y_l1.items():
                            if abs(b - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 2:
                        # pool 2 (or pool 3 for G1): in y_l2, keyed by (prev_sum, cur_sum)
                        for (ps, ns), var in y_l2.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)
                elif n_pools == 4:
                    # Four pools (G2): y_l1 + y_l2 + y_l3
                    y_l1 = y_layers[0]
                    y_l2 = y_layers[1]
                    y_l3 = y_layers[2]
                    if pi == 0:
                        for (a, b), var in y_l1.items():
                            if abs(a - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 1:
                        for (a, b), var in y_l1.items():
                            if abs(b - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 2:
                        # pool 2: in y_l2, keyed by (prev_sum, cur_sum), where cur_sum = pool2
                        for (ps, ns), var in y_l2.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 3:
                        # pool 4: in y_l3, keyed by (prev_sum, cur_sum), where cur_sum = pool4
                        for (ps, ns), var in y_l3.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)

                rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
                if lhs != 0 or rhs_terms:
                    prob += lhs == rhs, f"wy_g{gid}_p{pid}_si{s_idx}"

    # ── 槽位约束 ──
    for pid in POOLS:
        sl = pool_slots[pid]
        for gi, gr in enumerate(grades):
            terms = []
            for gid in pool_groups[pid]:
                for c in pool_combos[pid]:
                    cnt = c['counts'].get(gr, 0)
                    if cnt > 0:
                        terms.append(cnt * w[pid][gid][c['idx']])
            if terms:
                prob += pulp.lpSum(terms) <= sl[gr], f"sl_p{pid}_gi{gi}"

    # ── 目标函数 ──
    obj_terms = []
    for gid, g in enumerate(GROUPS):
        nc = g['nc']
        y_layers, info = all_y_info[gid]
        final_li = info['final_layer']
        final_layer = y_layers[final_li]

        for key, var in final_layer.items():
            # key is a tuple of (prev_sum, cur_sum) for the last two dimensions
            # or just (a,) for single-pool groups
            # We need the total sum across all pools
            if isinstance(key, tuple):
                if len(key) == 2:
                    total = rnd(key[0] + key[1])
                else:
                    total = key[0]
            else:
                total = key

            avg = rnd(total / nc)
            if avg >= threshold:
                obj_terms.append(var)

    prob += pulp.lpSum(obj_terms)

    # ── 统计 ──
    nw = sum(len(w[p][g]) for p in POOLS for g in pool_groups[p])
    ny = sum(len(all_y_flat[g]) for g in range(len(GROUPS)))
    if verbose:
        print(f"变量: w={nw}  y={ny}  总计={nw + ny}  约束={len(prob.constraints)}")
        print(f"求解中 (threshold={threshold}, time_limit={time_limit}s)...")

    prob.solve(pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit))

    st = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    max_s = int(round(obj_val)) if obj_val is not None else 0

    Gbar, Gmin = 3.061, 1.7
    ceil = min(1.0, (Gbar - Gmin) / (threshold - Gmin)) if threshold > Gmin else 1.0

    # ── 收集结果 ──
    gd = {}
    for gid, g in enumerate(GROUPS):
        nc = g['nc']
        y_layers, info = all_y_info[gid]
        final_li = info['final_layer']
        final_layer = y_layers[final_li]

        items = []
        ok = 0
        total_students = 0
        for key, var in final_layer.items():
            n = int(round(pulp.value(var) or 0))
            if n > 0:
                if isinstance(key, tuple) and len(key) == 2:
                    total = rnd(key[0] + key[1])
                else:
                    total = key[0] if isinstance(key, tuple) else key
                avg = rnd(total / nc)
                succ = avg >= threshold
                if succ:
                    ok += n
                total_students += n
                items.append({'n': n, 'avg': avg, 'total': total, 'key': key, 'ok': succ})

        gd[gid] = {'size': g['size'], 'ok': ok, 'total': total_students,
                   'items': sorted(items, key=lambda x: -x['n'])}

    return {
        'status': st,
        'threshold': threshold,
        'max_s': max_s,
        'ratio': max_s / TOTAL_STUDENTS,
        'ceiling': ceil,
        'groups': gd,
        'w': w,
    }


def show(r):
    print(f"\n{'=' * 60}")
    print(f"结果 | 状态={r['status']} | 阈值={r['threshold']}")
    print(f"达标: {r['max_s']}/{TOTAL_STUDENTS} ({r['ratio']:.2%})")
    print(f"天花板={r['ceiling']:.2%}  差距={r['ceiling'] - r['ratio']:+.2%}")

    for gid, g in enumerate(GROUPS):
        gd = r['groups'][gid]
        print(f"\n{'─' * 40}")
        print(f"{g['name']}({g['size']}人 {g['nc']}门): "
              f"达标{gd['ok']}/{g['size']} ({gd['ok'] / g['size']:.1%})")
        for item in gd['items'][:5]:
            st = "OK" if item['ok'] else "--"
            print(f"  [{st}] {item['n']:3d}人 均={item['avg']:.3f}  "
                  f"total_sum={item['total']:.1f}  key={item['key']}")


if __name__ == '__main__':
    # 打印槽位信息
    print("=" * 60)
    print("池槽位统计")
    print("=" * 60)
    for pid, p in POOLS.items():
        sl = pool_slots[pid]
        total_slots = sum(sl.values())
        print(f"  Pool {pid} {p['name']}({p['k']}门×{p['n']}人): "
              f"总槽位={total_slots}  "
              + " ".join(f"{g}={sl[g]}" for g in grades))

    print(f"\n组合数: " + " ".join(f"P{pid}={len(pool_combos[pid])}" for pid in POOLS))
    print(f"Sum 值数: " + " ".join(f"P{pid}={len(pool_sums[pid])}" for pid in POOLS))

    # 求解
    r = solve(threshold=3.5, time_limit=6000, verbose=True)
    show(r)