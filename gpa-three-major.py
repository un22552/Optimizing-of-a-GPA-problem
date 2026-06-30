"""
gpa-three-major.py — 三专业多层交叉 MILP（拆分大池版）
总 300 人: G1=150, G2=100, G3=50

Pool 结构 (k=10 拆为 5+5):
  0a: 公选a 5门  (原公选前半)
  0b: 公选b 5门  (原公选后半)
  1:  一二共修 4门
  2:  二三共修 3门
  3a: 专业课一a 5门
  3b: 专业课一b 5门
  4:  专业课二 7门
  5:  专业课三a 7门 (原 k=15 的一半)
  6:  专业课三b 8门 (原 k=15 的另一半)

每人修读:
  G1: 0a+0b+1+3a+3b = 24门
  G2: 0a+0b+1+2+4   = 24门
  G3: 0a+0b+2+5+6   = 28门

SDP = 0.5
每个池精确 per-combo w 变量（子池 ≤1287 种组合）
"""

import itertools
import math
import argparse
from collections import Counter, defaultdict
import pulp

grades = ['C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
gpa_map = {g: v for g, v in zip(grades, [1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 4.3])}
base = {'C-': 4, 'C': 5, 'C+': 10, 'B-': 16, 'B': 20, 'B+': 19, 'A-': 15, 'A': 8, 'A+': 3}
TOTAL_STUDENTS = 300

# ── Pool 定义 (k=10 拆为 5+5) ──
POOLS = {
    '0a': {'name': '公选a',     'k': 5, 'n': 300, 'groups': [0, 1, 2]},
    '0b': {'name': '公选b',     'k': 5, 'n': 300, 'groups': [0, 1, 2]},
    '1':  {'name': '一二共修',  'k': 4, 'n': 250, 'groups': [0, 1]},
    '2':  {'name': '二三共修',  'k': 3, 'n': 150, 'groups': [1, 2]},
    '3a': {'name': '专业课一a', 'k': 5, 'n': 150, 'groups': [0]},
    '3b': {'name': '专业课一b', 'k': 5, 'n': 150, 'groups': [0]},
    '4':  {'name': '专业课二',  'k': 7, 'n': 100, 'groups': [1]},
    '5':  {'name': '专业课三a', 'k': 7, 'n': 50,  'groups': [2]},
    '6':  {'name': '专业课三b', 'k': 8, 'n': 50,  'groups': [2]},
}

GROUPS = [
    {'name': 'G1', 'size': 150, 'pools': ['0a', '0b', '1', '3a', '3b'], 'nc': 24},
    {'name': 'G2', 'size': 100, 'pools': ['0a', '0b', '1', '2', '4'],    'nc': 24},
    {'name': 'G3', 'size': 50,  'pools': ['0a', '0b', '2', '5', '6'],    'nc': 28},
]

SDP = 1.0


def rnd(x):
    return round(round(x / SDP) * SDP, 10)


def sc(n):
    s = {g: int(round(base[g] * n / 100.0)) for g in grades}
    d = n - sum(s.values())
    if d:
        mg = max(grades, key=lambda g: s[g])
        s[mg] += d
    return s


# ── 预处理 ──
pool_combos = {}   # pid -> list of {idx, combo, sum_gpa, counts}
pool_sums = {}     # pid -> sorted unique sums
pool_si = {}       # pid -> {sum: index}
pool_cbs = {}      # pid -> {sum: [combo_indices]}
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

pool_groups = defaultdict(list)
for gid, g in enumerate(GROUPS):
    for pid in g['pools']:
        pool_groups[pid].append(gid)


def build_group_joint_vars(gid, prob):
    g = GROUPS[gid]
    pids = g['pools']
    n_pools = len(pids)

    # 所有池的 sum 值和索引
    sums = [pool_sums[pid] for pid in pids]
    sis = [pool_si[pid] for pid in pids]

    y_layers = []
    y_all_flat = []

    # 层级累加
    layer_sums_list = [sums[0]]  # 每层累积 sum 列表
    layer_si_list = [{s: i for i, s in enumerate(sums[0])}]
    prev_to_keys_map = []  # 每层 prev_sum → 上层 key 列表

    # Layer 1: sum(pool0) × sum(pool1)
    y_l1 = {}
    for a in sums[0]:
        for b in sums[1]:
            var = pulp.LpVariable(f"y_g{gid}_l1_{sis[0][a]}_{sis[1][b]}", 0, None, 'Integer')
            y_l1[(a, b)] = var
            y_all_flat.append(var)
    y_layers.append(y_l1)

    # 构建 layer 2 的 prev_sum 和映射
    prev_sums = sorted(set(rnd(a + b) for a in sums[0] for b in sums[1]))
    prev_si = {s: i for i, s in enumerate(prev_sums)}
    layer_sums_list.append(prev_sums)
    layer_si_list.append(prev_si)

    # Map prev_sum → list of (a,b) keys
    prev_to_ab = defaultdict(list)
    for a in sums[0]:
        for b in sums[1]:
            prev_to_ab[rnd(a + b)].append((a, b))
    prev_to_keys_map.append(dict(prev_to_ab))

    cur_prev_sums = prev_sums
    cur_prev_si = prev_si
    prev_y_dict = y_l1

    for li in range(2, n_pools):
        cur_pid = pids[li]
        sn = sums[li]
        sin = sis[li]
        y_cur = {}

        # 新累积 sum
        new_sums_set = set()
        p_to_keys_cur = defaultdict(list)
        for ps in cur_prev_sums:
            for ns in sn:
                cs = rnd(ps + ns)
                new_sums_set.add(cs)
                p_to_keys_cur[ps].append((ps, ns))
        new_sums = sorted(new_sums_set)
        new_si = {s: i for i, s in enumerate(new_sums)}

        for ps in cur_prev_sums:
            for ns in sn:
                var = pulp.LpVariable(
                    f"y_g{gid}_l{li}_{cur_prev_si[ps]}_{sin[ns]}", 0, None, 'Integer')
                y_cur[(ps, ns)] = var

        # Link to previous layer
        if li == 2:
            pmap = prev_to_keys_map[0]  # prev_sum → [(a,b)]
            for ps in cur_prev_sums:
                lhs_terms = [y_cur[(ps, ns)] for ns in sn if (ps, ns) in y_cur]
                rhs_terms = [prev_y_dict[(a, b)] for (a, b) in pmap.get(ps, [])]
                if lhs_terms:
                    if rhs_terms:
                        prob += pulp.lpSum(lhs_terms) == pulp.lpSum(rhs_terms), \
                            f"lk_g{gid}_l{li}_ps{cur_prev_si[ps]}"
                    else:
                        prob += pulp.lpSum(lhs_terms) == 0, f"lkz_g{gid}_l{li}_ps{cur_prev_si[ps]}"
                elif rhs_terms:
                    prob += pulp.lpSum(rhs_terms) == 0, f"lkz2_g{gid}_l{li}_ps{cur_prev_si[ps]}"
        else:
            # prev_y_dict keys are (prev_prev_sum, prev_cur_sum)
            # Build reverse map: ps_out → [(pps, pcs)] where pps+pcs=ps_out
            rev_map = defaultdict(list)
            for (pps, pcs), var in prev_y_dict.items():
                rev_map[rnd(pps + pcs)].append((pps, pcs))

            for ps in cur_prev_sums:
                lhs_terms = [y_cur[(ps, ns)] for ns in sn if (ps, ns) in y_cur]
                rhs_parts = [prev_y_dict[key] for key in rev_map.get(ps, [])]
                if lhs_terms:
                    if rhs_parts:
                        prob += pulp.lpSum(lhs_terms) == pulp.lpSum(rhs_parts), \
                            f"lk_g{gid}_l{li}_ps{cur_prev_si[ps]}"
                    else:
                        prob += pulp.lpSum(lhs_terms) == 0, f"lkz_g{gid}_l{li}_ps{cur_prev_si[ps]}"
                elif rhs_parts:
                    prob += pulp.lpSum(rhs_parts) == 0, f"lkz2_g{gid}_l{li}_ps{cur_prev_si[ps]}"

        y_layers.append(y_cur)
        prev_y_dict = y_cur

        if li < n_pools - 1:
            cur_prev_sums = new_sums
            cur_prev_si = new_si
            layer_sums_list.append(new_sums)
            layer_si_list.append(new_si)
            prev_to_keys_map.append(dict(p_to_keys_cur))

    return y_layers, y_all_flat, {
        'pids': pids,
        'sums': sums,
        'sis': sis,
        'final_layer': len(y_layers) - 1,
        'layer_sums': layer_sums_list,
        'layer_si': layer_si_list,
    }


def solve(threshold=3.5, time_limit=600, gap_rel=None, threads=None, max_nodes=None, verbose=True):
    prob = pulp.LpProblem("ThreeMajor", pulp.LpMaximize)

    # ── w 变量: per-combo ──
    w = {}
    for pid in POOLS:
        w[pid] = {}
        n_cb = len(pool_combos[pid])
        for gid in pool_groups[pid]:
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
        n_pools = len(pids)
        y_layers, info = all_y_info[gid]
        sums = info['sums']
        sis = info['sis']

        for pi, pid in enumerate(pids):
            s_list = sums[pi]
            s_idx_map = sis[pi]
            cbs = pool_cbs[pid]

            for s_val, s_idx in s_idx_map.items():
                c_indices = cbs.get(s_val, [])
                lhs = pulp.lpSum(w[pid][gid][ci] for ci in c_indices) if c_indices else 0

                rhs_terms = []
                if n_pools == 1:
                    y_layer = y_layers[0]
                    for key, var in y_layer.items():
                        if abs(key[0] - s_val) < 0.001:
                            rhs_terms.append(var)
                elif n_pools == 2:
                    y_layer = y_layers[0]
                    for (a, b), var in y_layer.items():
                        if pi == 0 and abs(a - s_val) < 0.001:
                            rhs_terms.append(var)
                        elif pi == 1 and abs(b - s_val) < 0.001:
                            rhs_terms.append(var)
                elif n_pools == 3:
                    y_l1 = y_layers[0]
                    y_l2 = y_layers[1]
                    if pi == 0:
                        for (a, b), var in y_l1.items():
                            if abs(a - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 1:
                        for (a, b), var in y_l1.items():
                            if abs(b - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 2:
                        for (ps, ns), var in y_l2.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)
                elif n_pools == 4:
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
                        for (ps, ns), var in y_l2.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 3:
                        for (ps, ns), var in y_l3.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)
                elif n_pools == 5:
                    y_l1 = y_layers[0]
                    y_l2 = y_layers[1]
                    y_l3 = y_layers[2]
                    y_l4 = y_layers[3]
                    if pi == 0:
                        for (a, b), var in y_l1.items():
                            if abs(a - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 1:
                        for (a, b), var in y_l1.items():
                            if abs(b - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 2:
                        for (ps, ns), var in y_l2.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 3:
                        for (ps, ns), var in y_l3.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)
                    elif pi == 4:
                        for (ps, ns), var in y_l4.items():
                            if abs(ns - s_val) < 0.001:
                                rhs_terms.append(var)

                rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
                if lhs != 0 or rhs_terms:
                    prob += lhs == rhs, f"wy_g{gid}_p{pid}_si{s_idx}"

    # ── 槽位约束 (精确 per-combo) ──
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
            if isinstance(key, tuple) and len(key) == 2:
                total = rnd(key[0] + key[1])
            else:
                total = key[0] if isinstance(key, tuple) else key
            avg = total / nc
            if avg >= threshold - 1e-9:
                obj_terms.append(var)

    prob += pulp.lpSum(obj_terms)

    # ── 统计 ──
    nw = sum(len(w[p][g]) for p in POOLS for g in pool_groups[p])
    ny = sum(len(all_y_flat[g]) for g in range(len(GROUPS)))
    if verbose:
        print(f"变量: w={nw}  y={ny}  总计={nw + ny}  约束={len(prob.constraints)}")
        print(f"求解中 (threshold={threshold}, time_limit={time_limit}s)...")

    solver = pulp.PULP_CBC_CMD(
        msg=verbose,
        timeLimit=time_limit,
        gapRel=gap_rel,
        threads=threads,
        maxNodes=max_nodes,
    )
    prob.solve(solver)

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
        for key, var in final_layer.items():
            n = int(round(pulp.value(var) or 0))
            if n > 0:
                if isinstance(key, tuple) and len(key) == 2:
                    total = rnd(key[0] + key[1])
                else:
                    total = key
                avg = total / nc
                succ = avg >= threshold - 1e-9
                if succ:
                    ok += n
                items.append({'n': n, 'avg': avg, 'total': total, 'key': key, 'ok': succ})
        gd[gid] = {'size': g['size'], 'ok': ok,
                   'items': sorted(items, key=lambda x: -x['n'])}

    return {
        'status': st,
        'threshold': threshold,
        'max_s': max_s,
        'ratio': max_s / TOTAL_STUDENTS,
        'ceiling': ceil,
        'groups': gd,
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
        for item in gd['items'][:6]:
            st = "OK" if item['ok'] else "--"
            print(f"  [{st}] {item['n']:3d}人 均={item['avg']:.3f}  "
                  f"total={item['total']:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--gap-rel", type=float, default=None, help="relative MIP gap for early stopping, e.g. 0.01")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("池槽位 + 组合数")
    print("=" * 60)
    for pid, p in POOLS.items():
        sl = pool_slots[pid]
        print(f"  {pid}: {p['name']}({p['k']}门x{p['n']}人) "
              f"总槽位={sum(sl.values())}  combos={len(pool_combos[pid])}  "
              f"sums={len(pool_sums[pid])}")

    r = solve(
        threshold=args.threshold,
        time_limit=args.time_limit,
        gap_rel=args.gap_rel,
        threads=args.threads,
        max_nodes=args.max_nodes,
        verbose=not args.quiet,
    )
    show(r)
