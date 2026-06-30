"""
gpa-three-major-heuristic.py — 三专业分解启发式求解
策略: 将共享池按比例分给各组, 然后对各组独立使用 gpa-1.py 的 IP 求解
这给出一个 feasible 下界
"""

import itertools
import math
import argparse
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pulp

grades = ['C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
gpa_map = {g: v for g, v in zip(grades, [1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 4.3])}
base = {'C-': 4, 'C': 5, 'C+': 10, 'B-': 16, 'B': 20, 'B+': 19, 'A-': 15, 'A': 8, 'A+': 3}
TOTAL_STUDENTS = 300
PROFILE_CACHE = {}
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


# ── 原始 Pool 定义 ──
# Pool 0: 公选 10门×300人 (拆为 0a+0b, 各 5门)
# Pool 1: 一二共修 4门×250人
# Pool 2: 二三共修 3门×150人
# Pool 3: 专业课一 10门×150人 (拆为 3a+3b, 各 5门)
# Pool 4: 专业课二 7门×100人
# Pool 5+6: 专业课三 15门×50人 (拆为 5(7门)+6(8门))

# 拆分为子池
SUB_POOLS = [
    {'name': '公选a',     'k': 5, 'n': 300, 'shared': [0, 1, 2], 'orig': 0},
    {'name': '公选b',     'k': 5, 'n': 300, 'shared': [0, 1, 2], 'orig': 0},
    {'name': '一二共修',  'k': 4, 'n': 250, 'shared': [0, 1],    'orig': 1},
    {'name': '二三共修',  'k': 3, 'n': 150, 'shared': [1, 2],    'orig': 2},
    {'name': '专一a',     'k': 5, 'n': 150, 'shared': [0],       'orig': 3},
    {'name': '专一b',     'k': 5, 'n': 150, 'shared': [0],       'orig': 3},
    {'name': '专二',      'k': 7, 'n': 100, 'shared': [1],       'orig': 4},
    {'name': '专三a',     'k': 7, 'n': 50,  'shared': [2],       'orig': 5},
    {'name': '专三b',     'k': 8, 'n': 50,  'shared': [2],       'orig': 5},
]

GROUPS = [
    {'name': 'G1',  'size': 150, 'pools': [0, 1, 2, 4, 5],  'nc': 24},
    {'name': 'G2',  'size': 100, 'pools': [0, 1, 2, 3, 6],  'nc': 24},
    {'name': 'G3',  'size': 50,  'pools': [0, 1, 3, 7, 8],  'nc': 28},
]


def build_profiles(k):
    if k in PROFILE_CACHE:
        return PROFILE_CACHE[k]

    cb = []
    ss = set()
    sm = defaultdict(list)
    for combo in itertools.combinations_with_replacement(grades, k):
        sg = rnd(sum(gpa_map[g] for g in combo))
        cnt = Counter(combo)
        idx = len(cb)
        cb.append({'idx': idx, 'combo': combo, 'sum_gpa': sg, 'counts': cnt,
                   'avg': sg / k})
        ss.add(sg)
        sm[sg].append(idx)
    result = (cb, sorted(ss), sm)
    PROFILE_CACHE[k] = result
    return result


def values_are_integral(vars_, tol=1e-5):
    for var in vars_:
        value = pulp.value(var)
        if value is None or abs(value - round(value)) > tol:
            return False
    return True


def solve_group(gid, allocated_slots, threshold, time_limit=120, verbose=False):
    """
    为单个 group 求解独立 GPA 分配问题。
    allocated_slots: {pid: {grade: count}} 分配给该组的每个池的槽位
    """
    g = GROUPS[gid]
    nc = g['nc']
    n_students = g['size']
    pids = g['pools']

    # 每个子池的 combo 枚举
    pool_cb = {}
    pool_sums = {}
    pool_si = {}
    pool_cbs = {}
    for pi, pid in enumerate(pids):
        sp = SUB_POOLS[pid]
        cb, sv, sm = build_profiles(sp['k'])
        pool_cb[pid] = cb
        pool_sums[pid] = sv
        pool_si[pid] = {s: i for i, s in enumerate(sv)}
        pool_cbs[pid] = dict(sm)

    prob = pulp.LpProblem(f"Group{gid}", pulp.LpMaximize)

    # w[pid][c] per-combo
    w = {}
    for pi, pid in enumerate(pids):
        n_cb = len(pool_cb[pid])
        w[pid] = [pulp.LpVariable(f"w_p{pid}_c{i}", 0, None, 'Integer')
                  for i in range(n_cb)]

    # 每组总人数
    for pid in pids:
        prob += pulp.lpSum(w[pid]) == n_students, f"sz_p{pid}"

    # 槽位约束
    for pi, pid in enumerate(pids):
        sl = allocated_slots[pid]
        for gi, gr in enumerate(grades):
            terms = []
            for c in pool_cb[pid]:
                cnt = c['counts'].get(gr, 0)
                if cnt > 0:
                    terms.append(cnt * w[pid][c['idx']])
            if terms:
                prob += pulp.lpSum(terms) <= sl.get(gr, 0), f"sl_p{pid}_gi{gi}"

    # 目标: 使用层次联合变量
    # 两层构建: pool0×pool1 → L1, L1×pool2 → L2, ...
    y_layers = []
    all_y_flat = []
    sums = [pool_sums[pid] for pid in pids]
    sis = [pool_si[pid] for pid in pids]

    # Layer 1
    s0, s1 = sums[0], sums[1]
    si0, si1 = sis[0], sis[1]
    y_l1 = {}
    for a in s0:
        for b in s1:
            var = pulp.LpVariable(f"y_l1_{si0[a]}_{si1[b]}", 0, None, 'Integer')
            y_l1[(a, b)] = var
            all_y_flat.append(var)
    y_layers.append(y_l1)

    # w ↔ y Layer1 边际
    for s_val, s_idx in sis[0].items():
        c_indices = pool_cbs[pids[0]].get(s_val, [])
        lhs = pulp.lpSum(w[pids[0]][ci] for ci in c_indices) if c_indices else 0
        rhs_terms = [y_l1[(a, b)] for (a, b) in y_l1 if abs(a - s_val) < 0.001]
        rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
        if lhs != 0 or rhs_terms:
            prob += lhs == rhs, f"wy0_si{s_idx}"

    for s_val, s_idx in sis[1].items():
        c_indices = pool_cbs[pids[1]].get(s_val, [])
        lhs = pulp.lpSum(w[pids[1]][ci] for ci in c_indices) if c_indices else 0
        rhs_terms = [y_l1[(a, b)] for (a, b) in y_l1 if abs(b - s_val) < 0.001]
        rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
        if lhs != 0 or rhs_terms:
            prob += lhs == rhs, f"wy1_si{s_idx}"

    prev_y = y_l1
    prev_sums_list = sorted(set(rnd(a + b) for a in s0 for b in s1))
    prev_si = {s: i for i, s in enumerate(prev_sums_list)}

    # Build layers 2..N-1
    prev_to_keys = defaultdict(list)
    for (a, b), var in y_l1.items():
        prev_to_keys[rnd(a + b)].append(((a, b), var))

    for li in range(2, len(pids)):
        cur_pid = pids[li]
        sn = sums[li]
        sin = sis[li]
        y_cur = {}

        new_sums = sorted(set(rnd(ps + ns) for ps in prev_sums_list for ns in sn))
        new_si = {s: i for i, s in enumerate(new_sums)}

        for ps in prev_sums_list:
            for ns in sn:
                var = pulp.LpVariable(f"y_l{li}_{prev_si[ps]}_{sin[ns]}",
                                      0, None, 'Integer')
                y_cur[(ps, ns)] = var

        # Link
        for ps in prev_sums_list:
            lhs_terms = [y_cur[(ps, ns)] for ns in sn if (ps, ns) in y_cur]
            rhs_terms = [var for key, var in prev_to_keys.get(ps, [])
                        if key in prev_y]
            if lhs_terms:
                if rhs_terms:
                    prob += pulp.lpSum(lhs_terms) == pulp.lpSum(rhs_terms), \
                        f"lk_l{li}_ps{prev_si[ps]}"
                else:
                    prob += pulp.lpSum(lhs_terms) == 0, f"lkz_l{li}_ps{prev_si[ps]}"
            elif rhs_terms:
                prob += pulp.lpSum(rhs_terms) == 0, f"lkz2_l{li}_ps{prev_si[ps]}"

        # w ↔ cur pool marginal
        for s_val, s_idx in sin.items():
            c_indices = pool_cbs[cur_pid].get(s_val, [])
            lhs = pulp.lpSum(w[cur_pid][ci] for ci in c_indices) if c_indices else 0
            rhs_terms = [y_cur[(ps, ns)] for (ps, ns) in y_cur if abs(ns - s_val) < 0.001]
            rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
            if lhs != 0 or rhs_terms:
                prob += lhs == rhs, f"wy{li}_si{s_idx}"

        y_layers.append(y_cur)
        prev_y = y_cur
        prev_sums_list = new_sums
        prev_si = new_si

        # Rebuild prev_to_keys for next iteration
        prev_to_keys = defaultdict(list)
        for (ps, ns), var in y_cur.items():
            prev_to_keys[rnd(ps + ns)].append(((ps, ns), var))

    # 目标: 最终层的 var, avg >= threshold
    final_layer = y_layers[-1]
    obj_terms = []
    for key, var in final_layer.items():
        total = rnd(key[0] + key[1])
        avg = total / nc
        if avg >= threshold - 1e-9:
            obj_terms.append(var)
    prob += pulp.lpSum(obj_terms)

    # 人数约束
    prob += pulp.lpSum(all_y_flat) == n_students, "total_size"

    prob.solve(pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit))

    st = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    max_s = int(round(obj_val)) if obj_val is not None else 0
    w_flat = [var for pid_vars in w.values() for var in pid_vars]
    has_integer_solution = st == 'Optimal' or (
        values_are_integral(all_y_flat) and values_are_integral(w_flat)
    )

    # 收集结果
    items = []
    ok = 0
    if not has_integer_solution:
        return {
            'status': f'{st} (fractional)',
            'ok': 0,
            'max_s': 0,
            'items': [{'n': n_students, 'avg': 0.0, 'total': 0.0, 'ok': False}],
            'w': w,
        }
    for key, var in final_layer.items():
        n = int(round(pulp.value(var) or 0))
        if n > 0:
            total = key[0] + key[1]
            avg = total / nc
            succ = avg >= threshold - 1e-9
            if succ:
                ok += n
            items.append({'n': n, 'avg': avg, 'total': total, 'ok': succ})
    items.sort(key=lambda x: -x['n'])

    return {'status': st, 'ok': ok, 'max_s': max_s, 'items': items, 'w': w}


def proportional_allocation(pid, gid):
    """
    将共享池 pid 按组内人数比例分配给组 gid
    共享池的总共 slot = k * sc(n), 分配比例 = g['size'] / sum(group sizes for shared groups)
    """
    sp = SUB_POOLS[pid]
    k = sp['k']
    total_n = sum(GROUPS[g]['size'] for g in sp['shared'])
    group_n = GROUPS[gid]['size']
    ratio = group_n / total_n

    # 总 slot
    total_slots = sc(sp['n'])
    allocated = {}
    # 比例分配, 整数化
    leftover = {}
    for g_name in grades:
        raw = total_slots[g_name] * ratio
        int_part = int(math.floor(raw))
        allocated[g_name] = int_part
        leftover[g_name] = raw - int_part

    # 分配余数给最大分数的
    total_left = sp['k'] * group_n - sum(allocated.values())
    if total_left > 0:
        sorted_g = sorted(grades, key=lambda x: leftover[x], reverse=True)
        for i in range(total_left):
            allocated[sorted_g[i % len(sorted_g)]] += 1
    elif total_left < 0:
        sorted_g = sorted(grades, key=lambda x: -leftover[x], reverse=True)
        for i in range(-total_left):
            allocated[sorted_g[i % len(sorted_g)]] -= 1

    # 乘以 k 门课
    return {g: k * allocated[g] for g in grades}


def solve_all(threshold=3.5):
    """分配共享池 → 独立求解各组 → 汇总"""
    results = {}
    all_ok = 0

    for gid, g in enumerate(GROUPS):
        # 为该组分配槽位
        allocated = {}
        for pi, pid in enumerate(g['pools']):
            sp = SUB_POOLS[pid]
            if len(sp['shared']) > 1:
                # 共享池, 按比例分配
                allocated[pid] = proportional_allocation(pid, gid)
            else:
                # 专属池, 全部分配
                sl = sc(sp['n'])
                allocated[pid] = {gr: sp['k'] * sl[gr] for gr in grades}

        print(f"\n求解 {g['name']} ({g['size']}人, {g['nc']}门)...")
        r = solve_group(gid, allocated, threshold, time_limit=120)
        results[gid] = r
        all_ok += r['ok']
        print(f"  状态={r['status']}  达标={r['ok']}/{g['size']} ({r['ok']/g['size']:.1%})")

    return results, all_ok


if False and __name__ == '__main__':
    Gbar, Gmin = 3.061, 1.7

    for T in [3.5, 3.7, 4.0]:
        ceil = min(1.0, (Gbar - Gmin) / (T - Gmin)) if T > Gmin else 1.0
        print(f"\n{'='*60}")
        print(f"阈值 T={T}  天花板={ceil:.2%}")
        print(f"{'='*60}")

        res, total_ok = solve_all(T)

        print(f"\n总计达标: {total_ok}/{TOTAL_STUDENTS} ({total_ok/TOTAL_STUDENTS:.2%})")
        print(f"天花板: {ceil:.2%}  差距: {ceil - total_ok/TOTAL_STUDENTS:+.2%}")

        for gid, g in enumerate(GROUPS):
            r = res[gid]
            print(f"\n  {g['name']}: 达标{r['ok']}/{g['size']} ({r['ok']/g['size']:.1%})")
            for item in r['items'][:4]:
                st = "OK" if item['ok'] else "--"
                print(f"    [{st}] {item['n']:3d}人 均={item['avg']:.3f}")

# Fast feasible heuristic -------------------------------------------------
# The original proportional allocator above solves each group independently,
# but it rounds shared pools from each group's point of view.  The replacement
# below allocates every shared pool once, so grade slots and group capacities
# both balance exactly.


def grade_score(gr):
    return gpa_map[gr] - 3.061


def distribute_counts(total_by_grade, shared_gids, capacities, biases):
    """Allocate one shared pool while preserving grade totals and group capacities."""
    allocation = {gid: {gr: 0 for gr in grades} for gid in shared_gids}
    remaining = dict(capacities)

    for gr in sorted(grades, key=gpa_map.get, reverse=True):
        total = total_by_grade[gr]
        if total <= 0:
            continue

        active = [gid for gid in shared_gids if remaining[gid] > 0]
        weights = {}
        for gid in active:
            raw = GROUPS[gid]['size'] * math.exp(biases.get(gid, 0.0) * grade_score(gr))
            weights[gid] = max(raw, 1e-9)

        weight_sum = sum(weights.values())
        base_take = {}
        fractions = []
        assigned = 0
        for gid in active:
            raw = total * weights[gid] / weight_sum
            take = min(remaining[gid], int(math.floor(raw)))
            base_take[gid] = take
            assigned += take
            fractions.append((raw - math.floor(raw), gid))

        left = total - assigned
        fractions.sort(reverse=True)
        while left > 0:
            moved = False
            for _, gid in fractions:
                if left <= 0:
                    break
                if base_take[gid] < remaining[gid]:
                    base_take[gid] += 1
                    left -= 1
                    moved = True
            if not moved:
                raise RuntimeError("shared-pool allocation ran out of capacity")

        for gid, take in base_take.items():
            allocation[gid][gr] += take
            remaining[gid] -= take

    if any(v != 0 for v in remaining.values()):
        raise RuntimeError(f"capacity mismatch after allocation: {remaining}")

    return allocation


def make_allocations(pool_biases=None):
    pool_biases = pool_biases or {}
    allocated_by_group = {gid: {} for gid in range(len(GROUPS))}

    for pid, sp in enumerate(SUB_POOLS):
        total_slots = {gr: sp['k'] * sc(sp['n'])[gr] for gr in grades}

        if len(sp['shared']) == 1:
            gid = sp['shared'][0]
            allocated_by_group[gid][pid] = total_slots
            continue

        capacities = {gid: sp['k'] * GROUPS[gid]['size'] for gid in sp['shared']}
        split = distribute_counts(
            total_slots,
            sp['shared'],
            capacities,
            pool_biases.get(pid, {}),
        )
        for gid, slots in split.items():
            allocated_by_group[gid][pid] = slots

    return allocated_by_group


def make_allocation_with_splits(splits=None, pool_biases=None):
    splits = splits or {}
    pool_biases = pool_biases or {}
    allocated_by_group = {gid: {} for gid in range(len(GROUPS))}

    for pid, sp in enumerate(SUB_POOLS):
        total_slots = {gr: sp['k'] * sc(sp['n'])[gr] for gr in grades}

        if len(sp['shared']) == 1:
            gid = sp['shared'][0]
            allocated_by_group[gid][pid] = total_slots
            continue

        if pid in splits:
            split = splits[pid]
        else:
            capacities = {gid: sp['k'] * GROUPS[gid]['size'] for gid in sp['shared']}
            split = distribute_counts(
                total_slots,
                sp['shared'],
                capacities,
                pool_biases.get(pid, {}),
            )
        for gid, slots in split.items():
            allocated_by_group[gid][pid] = dict(slots)

    return allocated_by_group


def make_shared_splits(pool_biases=None):
    allocation = make_allocation_with_splits(pool_biases=pool_biases)
    splits = {}
    for pid, sp in enumerate(SUB_POOLS):
        if len(sp['shared']) <= 1:
            continue
        splits[pid] = {gid: dict(allocation[gid][pid]) for gid in sp['shared']}
    return splits


def solve_all_fast_from_allocations(threshold, allocations, group_time_limit=60, verbose=True):
    results = {}
    all_ok = 0

    for gid, g in enumerate(GROUPS):
        if verbose:
            print(f"\nSolving {g['name']} ({g['size']} students, {g['nc']} courses)...")
        r = solve_group(gid, allocations[gid], threshold, time_limit=group_time_limit)
        results[gid] = r
        all_ok += r['ok']
        if verbose:
            print(f"  status={r['status']}  ok={r['ok']}/{g['size']} ({r['ok']/g['size']:.1%})")

    return results, all_ok


def solve_all_fast(threshold=3.5, pool_biases=None, group_time_limit=60, verbose=True):
    results = {}
    all_ok = 0
    allocations = make_allocations(pool_biases)

    for gid, g in enumerate(GROUPS):
        if verbose:
            print(f"\nSolving {g['name']} ({g['size']} students, {g['nc']} courses)...")
        r = solve_group(gid, allocations[gid], threshold, time_limit=group_time_limit)
        results[gid] = r
        all_ok += r['ok']
        if verbose:
            print(f"  status={r['status']}  ok={r['ok']}/{g['size']} ({r['ok']/g['size']:.1%})")

    return results, all_ok


def expand_slots(slot_counts):
    values = []
    for gr, count in slot_counts.items():
        values.extend([gpa_map[gr]] * count)
    values.sort(reverse=True)
    return values


def pool_chunks(slot_counts, k, n_students, order):
    values = expand_slots(slot_counts)
    if len(values) != k * n_students:
        raise RuntimeError(f"slot count mismatch: got {len(values)}, expected {k * n_students}")

    chunks = {sid: [] for sid in range(n_students)}
    pos = 0
    for sid in order:
        chunks[sid] = values[pos:pos + k]
        pos += k
    return {sid: sum(vals) for sid, vals in chunks.items()}


def solve_group_greedy(gid, allocated_slots, threshold, strategy="concentrate"):
    g = GROUPS[gid]
    n_students = g['size']
    target = threshold * g['nc']
    totals = [0.0] * n_students

    for pid in g['pools']:
        sp = SUB_POOLS[pid]
        if strategy == "spread":
            values = expand_slots(allocated_slots[pid])
            order = sorted(range(n_students), key=lambda sid: (-totals[sid], sid))
            additions = [0.0] * n_students
            for i, value in enumerate(values):
                additions[order[i % n_students]] += value
            for sid, add in enumerate(additions):
                totals[sid] += add
            continue

        if strategy == "fill":
            values = expand_slots(allocated_slots[pid])
            used = [0] * n_students
            for value in values:
                candidates = [sid for sid in range(n_students) if used[sid] < sp['k']]
                below = [sid for sid in candidates if totals[sid] < target - 1e-9]
                if below:
                    sid = max(below, key=lambda x: (totals[x], -used[x], -x))
                else:
                    sid = min(candidates, key=lambda x: (totals[x], used[x], x))
                totals[sid] += value
                used[sid] += 1
            continue

        if strategy == "balance":
            order = sorted(range(n_students), key=lambda sid: (totals[sid], sid))
        elif strategy == "near_target":
            order = sorted(
                range(n_students),
                key=lambda sid: (totals[sid] >= target, -totals[sid], sid),
            )
        else:
            order = sorted(range(n_students), key=lambda sid: (-totals[sid], sid))

        additions = pool_chunks(allocated_slots[pid], sp['k'], n_students, order)
        for sid, add in additions.items():
            totals[sid] += add

    items_by_avg = Counter(round(total / g['nc'], 10) for total in totals)
    items = []
    ok = 0
    for avg, n in items_by_avg.items():
        succ = avg >= threshold - 1e-9
        if succ:
            ok += n
        items.append({'n': n, 'avg': avg, 'total': avg * g['nc'], 'ok': succ})
    items.sort(key=lambda x: (-x['ok'], -x['avg'], -x['n']))
    return {'status': 'Greedy', 'ok': ok, 'max_s': ok, 'items': items}


def solve_all_greedy(threshold=3.5, pool_biases=None, strategy="concentrate", verbose=True):
    results = {}
    all_ok = 0
    allocations = make_allocations(pool_biases)

    for gid, g in enumerate(GROUPS):
        r = solve_group_greedy(gid, allocations[gid], threshold, strategy=strategy)
        results[gid] = r
        all_ok += r['ok']
        if verbose:
            print(f"  {g['name']}: status={r['status']} ok={r['ok']}/{g['size']} ({r['ok']/g['size']:.1%})")

    return results, all_ok


def solve_all_greedy_from_allocations(threshold, allocations, strategy="concentrate", verbose=True):
    results = {}
    all_ok = 0

    for gid, g in enumerate(GROUPS):
        r = solve_group_greedy(gid, allocations[gid], threshold, strategy=strategy)
        results[gid] = r
        all_ok += r['ok']
        if verbose:
            print(f"  {g['name']}: status={r['status']} ok={r['ok']}/{g['size']} ({r['ok']/g['size']:.1%})")

    return results, all_ok


def solve_all_greedy_best(threshold=3.5, pool_biases=None, verbose=True):
    best = None
    for strategy in ["spread", "fill", "concentrate", "near_target", "balance"]:
        if verbose:
            print(f"\nGreedy strategy: {strategy}")
        results, total_ok = solve_all_greedy(
            threshold=threshold,
            pool_biases=pool_biases,
            strategy=strategy,
            verbose=verbose,
        )
        if best is None or total_ok > best['total_ok']:
            best = {'strategy': strategy, 'results': results, 'total_ok': total_ok}
    return best['results'], best['total_ok'], best['strategy']


def solve_all_greedy_best_from_allocations(threshold, allocations, verbose=True):
    best = None
    for strategy in ["spread", "fill", "concentrate", "near_target", "balance"]:
        if verbose:
            print(f"\nGreedy strategy: {strategy}")
        results, total_ok = solve_all_greedy_from_allocations(
            threshold=threshold,
            allocations=allocations,
            strategy=strategy,
            verbose=verbose,
        )
        if best is None or total_ok > best['total_ok']:
            best = {'strategy': strategy, 'results': results, 'total_ok': total_ok}
    return best['results'], best['total_ok'], best['strategy']


def random_biases(rng, scale):
    biases = {}
    for pid, sp in enumerate(SUB_POOLS):
        if len(sp['shared']) <= 1:
            continue
        vals = {gid: rng.uniform(-scale, scale) for gid in sp['shared']}
        avg = sum(vals.values()) / len(vals)
        biases[pid] = {gid: vals[gid] - avg for gid in sp['shared']}
    return biases


def normalize_biases(biases):
    normalized = {}
    for pid, vals in biases.items():
        if not vals:
            continue
        avg = sum(vals.values()) / len(vals)
        normalized[pid] = {gid: val - avg for gid, val in vals.items()}
    return normalized


def mutate_biases(biases, rng, step):
    mutated = {pid: dict(vals) for pid, vals in biases.items()}
    shared_pids = [pid for pid, sp in enumerate(SUB_POOLS) if len(sp['shared']) > 1]
    pid = rng.choice(shared_pids)
    gids = SUB_POOLS[pid]['shared']
    for gid in gids:
        mutated.setdefault(pid, {})[gid] = mutated.get(pid, {}).get(gid, 0.0) + rng.gauss(0.0, step)
    return normalize_biases(mutated)


def bias_key(biases, ndigits=3):
    parts = []
    for pid in sorted(biases):
        vals = tuple((gid, round(biases[pid].get(gid, 0.0), ndigits)) for gid in sorted(biases[pid]))
        parts.append((pid, vals))
    return tuple(parts)


def bias_summary(biases):
    chunks = []
    for pid in sorted(biases):
        vals = ", ".join(f"G{gid + 1}:{biases[pid][gid]:+.2f}" for gid in sorted(biases[pid]))
        chunks.append(f"p{pid}({vals})")
    return " ".join(chunks)


def score_greedy_candidate(threshold, biases):
    results, total_ok, strategy = solve_all_greedy_best(
        threshold=threshold,
        pool_biases=biases,
        verbose=False,
    )
    return total_ok, strategy, results


def balance_score(total_ok, results):
    ratios = [results[gid]['ok'] / GROUPS[gid]['size'] for gid in range(len(GROUPS))]
    return total_ok + 25.0 * min(ratios) + 10.0 * sum(ratios)


def group_resource_estimate(threshold, gid, allocations):
    g = GROUPS[gid]
    nc = g['nc']
    size = g['size']
    slot_counts = {gr: 0 for gr in grades}

    for pid in g['pools']:
        for gr, count in allocations[gid][pid].items():
            slot_counts[gr] += count

    total_points = sum(gpa_map[gr] * count for gr, count in slot_counts.items())
    min_points = size * nc * 1.7
    pass_extra = nc * (threshold - 1.7)
    point_cap = int((total_points - min_points) // pass_extra) if pass_extra > 0 else size

    bplus_need = max(1e-9, nc * max(0.0, threshold - 3.3))
    bplus_excess = sum(max(0.0, gpa_map[gr] - 3.3) * count for gr, count in slot_counts.items())
    bplus_cap = int(bplus_excess // bplus_need) if threshold > 3.3 else size

    b_need = max(1e-9, nc * max(0.0, threshold - 3.0))
    b_excess = sum(max(0.0, gpa_map[gr] - 3.0) * count for gr, count in slot_counts.items())
    b_cap = int(b_excess // b_need) if threshold > 3.0 else size

    return max(0, min(size, point_cap, bplus_cap, b_cap))


def allocation_surrogate_score(threshold, allocations):
    estimates = [group_resource_estimate(threshold, gid, allocations) for gid in range(len(GROUPS))]
    ratios = [estimates[gid] / GROUPS[gid]['size'] for gid in range(len(GROUPS))]
    return sum(estimates) + 20.0 * min(ratios) + 5.0 * sum(ratios)


def split_surrogate_score(threshold, splits):
    return allocation_surrogate_score(threshold, make_allocation_with_splits(splits=splits))


def root_label(label):
    return label.split(".", 1)[0]


def diverse_top_items(scored_values, keyers, limit, item_key):
    rankings = [
        sorted(scored_values, key=keyer, reverse=True)
        for keyer in keyers
    ]
    selected = []
    seen = set()
    seen_roots = set()

    def try_add(item, require_new_root):
        key = item_key(item)
        if key in seen:
            return False
        root = root_label(item['label'])
        if require_new_root and root in seen_roots:
            return False
        selected.append(item)
        seen.add(key)
        seen_roots.add(root)
        return True

    max_len = max((len(ranking) for ranking in rankings), default=0)
    for require_new_root in [True, False]:
        for idx in range(max_len):
            for ranking in rankings:
                if idx < len(ranking):
                    try_add(ranking[idx], require_new_root=require_new_root)
                    if len(selected) >= limit:
                        return selected
    return selected


def refine_bias_item(item, threshold, group_time_limit):
    results, total_ok = solve_all_fast(
        threshold=threshold,
        pool_biases=item['biases'],
        group_time_limit=group_time_limit,
        verbose=False,
    )
    return {
        'label': item['label'],
        'biases': item['biases'],
        'results': results,
        'total_ok': total_ok,
    }


def refine_split_item(item, threshold, group_time_limit):
    allocations = make_allocation_with_splits(splits=item['splits'])
    results, total_ok = solve_all_fast_from_allocations(
        threshold=threshold,
        allocations=allocations,
        group_time_limit=group_time_limit,
        verbose=False,
    )
    return {
        'label': item['label'],
        'splits': item['splits'],
        'results': results,
        'total_ok': total_ok,
    }


def refine_items(items, threshold, group_time_limit, jobs, kind):
    if jobs <= 1:
        refined = []
        for item in items:
            print(
                f"\n=== Refining {item['label']} with group MILP "
                f"(greedy={item.get('greedy_ok', '?')}/{TOTAL_STUDENTS}) ==="
            )
            if kind == "split":
                result = refine_split_item(item, threshold, group_time_limit)
            else:
                if 'biases' in item:
                    print(f"Biases: {bias_summary(item['biases'])}")
                result = refine_bias_item(item, threshold, group_time_limit)
            ratio = result['total_ok'] / TOTAL_STUDENTS
            print(f"Refined {item['label']}: total_ok={result['total_ok']}/{TOTAL_STUDENTS} ({ratio:.2%})")
            refined.append(result)
        return refined

    print(f"\nRefining {len(items)} candidates with {jobs} parallel workers...")
    refined = []
    worker = refine_split_item if kind == "split" else refine_bias_item
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(worker, item, threshold, group_time_limit): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            result = future.result()
            ratio = result['total_ok'] / TOTAL_STUDENTS
            print(f"Refined {item['label']}: total_ok={result['total_ok']}/{TOTAL_STUDENTS} ({ratio:.2%})")
            refined.append(result)
    return refined


def clone_splits(splits):
    return {
        pid: {gid: dict(slots) for gid, slots in split.items()}
        for pid, split in splits.items()
    }


def splits_key(splits):
    parts = []
    for pid in sorted(splits):
        pid_parts = []
        for gid in sorted(splits[pid]):
            pid_parts.append((gid, tuple((gr, splits[pid][gid].get(gr, 0)) for gr in grades)))
        parts.append((pid, tuple(pid_parts)))
    return tuple(parts)


def mutate_splits(splits, rng, swaps=1):
    mutated = clone_splits(splits)
    shared_pids = list(mutated.keys())

    for _ in range(swaps):
        pid = rng.choice(shared_pids)
        gids = list(mutated[pid].keys())
        if len(gids) < 2:
            continue

        src, dst = rng.sample(gids, 2)
        high_grades = sorted(grades, key=gpa_map.get, reverse=True)
        low_grades = sorted(grades, key=gpa_map.get)

        high_choices = [gr for gr in high_grades if mutated[pid][src].get(gr, 0) > 0]
        low_choices = [gr for gr in low_grades if mutated[pid][dst].get(gr, 0) > 0]
        if not high_choices or not low_choices:
            continue

        high = rng.choice(high_choices[:max(1, min(4, len(high_choices)))])
        low = rng.choice(low_choices[:max(1, min(4, len(low_choices)))])
        if high == low:
            continue

        mutated[pid][src][high] -= 1
        mutated[pid][dst][high] = mutated[pid][dst].get(high, 0) + 1
        mutated[pid][dst][low] -= 1
        mutated[pid][src][low] = mutated[pid][src].get(low, 0) + 1

    return mutated


def apply_single_swap(splits, pid, src, dst, high, low, amount=1):
    if high == low:
        return None
    if splits[pid][src].get(high, 0) < amount:
        return None
    if splits[pid][dst].get(low, 0) < amount:
        return None

    mutated = clone_splits(splits)
    mutated[pid][src][high] -= amount
    mutated[pid][dst][high] = mutated[pid][dst].get(high, 0) + amount
    mutated[pid][dst][low] -= amount
    mutated[pid][src][low] = mutated[pid][src].get(low, 0) + amount
    return mutated


def iter_directed_swaps(splits, limit_per_pair=18):
    high_grades = sorted(grades, key=gpa_map.get, reverse=True)
    low_grades = sorted(grades, key=gpa_map.get)

    for pid, split in splits.items():
        gids = list(split.keys())
        for src in gids:
            for dst in gids:
                if src == dst:
                    continue
                moves = []
                for high in high_grades:
                    if split[src].get(high, 0) <= 0:
                        continue
                    for low in low_grades:
                        if gpa_map[high] <= gpa_map[low]:
                            continue
                        if split[dst].get(low, 0) <= 0:
                            continue
                        moves.append((pid, src, dst, high, low))
                for move in moves[:limit_per_pair]:
                    yield move


def score_split_candidate(threshold, splits):
    allocations = make_allocation_with_splits(splits=splits)
    results, total_ok, strategy = solve_all_greedy_best_from_allocations(
        threshold=threshold,
        allocations=allocations,
        verbose=False,
    )
    return total_ok, strategy, results


def swap_search_allocations(
    threshold,
    seed=1,
    scale=1.8,
    initial=100,
    rounds=8,
    beam=12,
    mutations=24,
    swap_size=3,
    refine_top=8,
    group_time_limit=30,
    jobs=1,
):
    rng = random.Random(seed)
    candidates = [(make_shared_splits({}), "proportional")]
    for i in range(initial):
        candidates.append((make_shared_splits(random_biases(rng, scale)), f"random-{i + 1}"))

    scored = {}
    for splits, label in candidates:
        total_ok, strategy, results = score_split_candidate(threshold, splits)
        scored[splits_key(splits)] = {
            'label': label,
            'splits': splits,
            'greedy_ok': total_ok,
            'surrogate': split_surrogate_score(threshold, splits),
            'strategy': strategy,
            'results': results,
        }

    for round_i in range(rounds):
        beam_items = diverse_top_items(
            scored.values(),
            keyers=[
                lambda x: x.get('surrogate', 0),
                lambda x: balance_score(x['greedy_ok'], x['results']),
                lambda x: x['greedy_ok'],
                lambda x: x['results'][0]['ok'],
                lambda x: x['results'][1]['ok'],
                lambda x: x['results'][2]['ok'],
            ],
            limit=beam,
            item_key=lambda x: splits_key(x['splits']),
        )
        best = max(beam_items, key=lambda x: x['greedy_ok'])
        best_surrogate = max(beam_items, key=lambda x: x.get('surrogate', 0))
        print(
            f"Swap round {round_i + 1}: best greedy={best['greedy_ok']}/{TOTAL_STUDENTS} "
            f"label={best['label']} strategy={best['strategy']} "
            f"best_surrogate={best_surrogate.get('surrogate', 0):.1f}"
        )

        for item in beam_items:
            for mutation_i in range(mutations):
                mutated = mutate_splits(item['splits'], rng, swaps=swap_size)
                key = splits_key(mutated)
                if key in scored:
                    continue
                total_ok, strategy, results = score_split_candidate(threshold, mutated)
                scored[key] = {
                    'label': f"{item['label']}.s{round_i + 1}.{mutation_i + 1}",
                    'splits': mutated,
                    'greedy_ok': total_ok,
                    'surrogate': split_surrogate_score(threshold, mutated),
                    'strategy': strategy,
                    'results': results,
                }

    top = diverse_top_items(
        scored.values(),
        keyers=[
            lambda x: x.get('surrogate', 0),
            lambda x: x['greedy_ok'],
            lambda x: balance_score(x['greedy_ok'], x['results']),
            lambda x: x['results'][0]['ok'],
            lambda x: x['results'][1]['ok'],
            lambda x: x['results'][2]['ok'],
        ],
        limit=refine_top,
        item_key=lambda x: splits_key(x['splits']),
    )

    best_refined = None
    for result in refine_items(top, threshold, group_time_limit, jobs, kind="split"):
        if best_refined is None or result['total_ok'] > best_refined['total_ok']:
            best_refined = {
                'label': f"{result['label']}+milp",
                'splits': result['splits'],
                'results': result['results'],
                'total_ok': result['total_ok'],
            }
            print(f"New best: {result['label']}+milp")

    return best_refined


def tabu_search_allocations(
    threshold,
    seed=1,
    scale=1.8,
    initial=80,
    rounds=8,
    beam=8,
    neighborhood=12,
    move_limit=18,
    tabu_tenure=250,
    refine_top=10,
    group_time_limit=30,
    jobs=1,
):
    rng = random.Random(seed)
    starts = [(make_shared_splits({}), "proportional")]
    starts.extend((make_shared_splits(random_biases(rng, scale)), f"random-{i + 1}") for i in range(initial))

    scored = {}
    frontier = []
    for splits, label in starts:
        total_ok, strategy, results = score_split_candidate(threshold, splits)
        item = {
            'label': label,
            'splits': splits,
            'greedy_ok': total_ok,
            'surrogate': split_surrogate_score(threshold, splits),
            'strategy': strategy,
            'results': results,
        }
        key = splits_key(splits)
        scored[key] = item
        frontier.append(item)

    tabu = []
    tabu_set = set()

    for round_i in range(rounds):
        frontier = diverse_top_items(
            frontier + list(scored.values()),
            keyers=[
                lambda x: x.get('surrogate', 0),
                lambda x: balance_score(x['greedy_ok'], x['results']),
                lambda x: x['greedy_ok'],
                lambda x: x['results'][0]['ok'],
                lambda x: x['results'][1]['ok'],
                lambda x: x['results'][2]['ok'],
            ],
            limit=beam,
            item_key=lambda x: splits_key(x['splits']),
        )
        best = max(scored.values(), key=lambda x: x['greedy_ok'])
        best_surrogate = max(scored.values(), key=lambda x: x.get('surrogate', 0))
        print(
            f"Tabu round {round_i + 1}: best greedy={best['greedy_ok']}/{TOTAL_STUDENTS} "
            f"label={best['label']} strategy={best['strategy']} "
            f"best_surrogate={best_surrogate.get('surrogate', 0):.1f}"
        )

        next_items = []
        for item in frontier:
            moves = list(iter_directed_swaps(item['splits'], limit_per_pair=move_limit))
            rng.shuffle(moves)
            evaluated = []
            for move in moves:
                pid, src, dst, high, low = move
                move_key = (pid, src, dst, high, low)
                if move_key in tabu_set:
                    continue
                mutated = apply_single_swap(item['splits'], pid, src, dst, high, low)
                if mutated is None:
                    continue
                key = splits_key(mutated)
                if key in scored:
                    continue
                total_ok, strategy, results = score_split_candidate(threshold, mutated)
                child = {
                    'label': f"{item['label']}.t{round_i + 1}",
                    'splits': mutated,
                    'greedy_ok': total_ok,
                    'surrogate': split_surrogate_score(threshold, mutated),
                    'strategy': strategy,
                    'results': results,
                }
                scored[key] = child
                evaluated.append(child)

                tabu.append((pid, dst, src, high, low))
                tabu_set.add((pid, dst, src, high, low))
                if len(tabu) > tabu_tenure:
                    old = tabu.pop(0)
                    tabu_set.discard(old)
                if len(evaluated) >= neighborhood:
                    break

            next_items.extend(evaluated)

        if next_items:
            frontier = next_items

    top = diverse_top_items(
        scored.values(),
        keyers=[
            lambda x: x.get('surrogate', 0),
            lambda x: x['greedy_ok'],
            lambda x: balance_score(x['greedy_ok'], x['results']),
            lambda x: x['results'][0]['ok'],
            lambda x: x['results'][1]['ok'],
            lambda x: x['results'][2]['ok'],
        ],
        limit=refine_top,
        item_key=lambda x: splits_key(x['splits']),
    )

    best_refined = None
    for result in refine_items(top, threshold, group_time_limit, jobs, kind="split"):
        if best_refined is None or result['total_ok'] > best_refined['total_ok']:
            best_refined = {
                'label': f"{result['label']}+milp",
                'splits': result['splits'],
                'results': result['results'],
                'total_ok': result['total_ok'],
            }
            print(f"New best: {result['label']}+milp")

    return best_refined


def local_search_allocations(
    threshold,
    seed=1,
    scale=1.2,
    initial=80,
    rounds=6,
    beam=8,
    mutations=12,
    refine_top=5,
    group_time_limit=30,
    jobs=1,
):
    rng = random.Random(seed)
    candidates = [({}, "proportional")]
    candidates.extend((random_biases(rng, scale), f"random-{i + 1}") for i in range(initial))

    scored = {}
    for biases, label in candidates:
        total_ok, strategy, results = score_greedy_candidate(threshold, biases)
        scored[bias_key(biases)] = {
            'label': label,
            'biases': biases,
            'greedy_ok': total_ok,
            'strategy': strategy,
            'results': results,
        }

    step = scale / 2.0
    for round_i in range(rounds):
        beam_items = sorted(scored.values(), key=lambda x: x['greedy_ok'], reverse=True)[:beam]
        best = beam_items[0]
        print(
            f"Local round {round_i + 1}: best greedy={best['greedy_ok']}/{TOTAL_STUDENTS} "
            f"label={best['label']} strategy={best['strategy']}"
        )

        for item in beam_items:
            for mutation_i in range(mutations):
                mutated = mutate_biases(item['biases'], rng, step)
                key = bias_key(mutated)
                if key in scored:
                    continue
                total_ok, strategy, results = score_greedy_candidate(threshold, mutated)
                scored[key] = {
                    'label': f"{item['label']}.m{round_i + 1}.{mutation_i + 1}",
                    'biases': mutated,
                    'greedy_ok': total_ok,
                    'strategy': strategy,
                    'results': results,
                }
        step *= 0.65

    top = diverse_top_items(
        scored.values(),
        keyers=[
            lambda x: x['greedy_ok'],
            lambda x: balance_score(x['greedy_ok'], x['results']),
            lambda x: x['results'][0]['ok'],
            lambda x: x['results'][1]['ok'],
            lambda x: x['results'][2]['ok'],
        ],
        limit=refine_top,
        item_key=lambda x: bias_key(x['biases']),
    )
    best_refined = None
    for result in refine_items(top, threshold, group_time_limit, jobs, kind="bias"):
        if best_refined is None or result['total_ok'] > best_refined['total_ok']:
            best_refined = {
                'label': f"{result['label']}+milp",
                'biases': result['biases'],
                'results': result['results'],
                'total_ok': result['total_ok'],
            }
            print(f"New best: {result['label']}+milp")

    return best_refined


def search_allocations(
    threshold,
    iterations=12,
    seed=1,
    scale=1.0,
    group_time_limit=45,
    mode="greedy",
    refine_top=3,
    jobs=1,
):
    rng = random.Random(seed)
    candidates = [({}, "proportional")]
    candidates.extend((random_biases(rng, scale), f"random-{i + 1}") for i in range(iterations))

    best = None
    scored = []
    for biases, label in candidates:
        print(f"\n=== Candidate {label} ===")
        if mode == "milp":
            results, total_ok = solve_all_fast(
                threshold=threshold,
                pool_biases=biases,
                group_time_limit=group_time_limit,
                verbose=True,
            )
            extra = ""
        elif mode == "hybrid":
            results, total_ok, strategy = solve_all_greedy_best(
                threshold=threshold,
                pool_biases=biases,
                verbose=True,
            )
            scored.append((total_ok, label, biases))
            extra = f" greedy_strategy={strategy}"
        else:
            results, total_ok, strategy = solve_all_greedy_best(
                threshold=threshold,
                pool_biases=biases,
                verbose=True,
            )
            extra = f" strategy={strategy}"
        ratio = total_ok / TOTAL_STUDENTS
        print(f"Candidate {label}: total_ok={total_ok}/{TOTAL_STUDENTS} ({ratio:.2%}){extra}")
        if best is None or total_ok > best['total_ok']:
            best = {'label': label, 'biases': biases, 'results': results, 'total_ok': total_ok}
            print(f"New best: {label}")

    if mode == "hybrid":
        scored.sort(reverse=True, key=lambda item: item[0])
        top = [
            {'label': label, 'biases': biases, 'greedy_ok': total_ok}
            for total_ok, label, biases in scored[:refine_top]
        ]
        for result in refine_items(top, threshold, group_time_limit, jobs, kind="bias"):
            if best is None or result['total_ok'] > best['total_ok']:
                best = {
                    'label': f"{result['label']}+milp",
                    'biases': result['biases'],
                    'results': result['results'],
                    'total_ok': result['total_ok'],
                }
                print(f"New best: {result['label']}+milp")

    return best


def print_summary(threshold, results, total_ok):
    Gbar, Gmin = 3.061, 1.7
    ceil = min(1.0, (Gbar - Gmin) / (threshold - Gmin)) if threshold > Gmin else 1.0
    print(f"\nTotal ok: {total_ok}/{TOTAL_STUDENTS} ({total_ok/TOTAL_STUDENTS:.2%})")
    print(f"Ceiling: {ceil:.2%}  gap: {ceil - total_ok/TOTAL_STUDENTS:+.2%}")

    for gid, g in enumerate(GROUPS):
        r = results[gid]
        print(f"\n  {g['name']}: ok={r['ok']}/{g['size']} ({r['ok']/g['size']:.1%})")
        for item in r['items'][:4]:
            st = "OK" if item['ok'] else "--"
            print(f"    [{st}] {item['n']:3d} students avg={item['avg']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=float, default=[3.5])
    parser.add_argument("--search", type=int, default=0, help="number of random shared-pool allocations to try")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--time-limit", type=int, default=60, help="CBC seconds per group solve")
    parser.add_argument("--mode", choices=["greedy", "milp", "hybrid", "local", "swap", "tabu"], default="greedy")
    parser.add_argument("--refine-top", type=int, default=3, help="hybrid mode: greedy candidates to refine by MILP")
    parser.add_argument("--local-initial", type=int, default=80)
    parser.add_argument("--local-rounds", type=int, default=6)
    parser.add_argument("--local-beam", type=int, default=8)
    parser.add_argument("--local-mutations", type=int, default=12)
    parser.add_argument("--swap-size", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1, help="parallel MILP refinements; keep small to avoid CPU oversubscription")
    parser.add_argument("--neighborhood", type=int, default=12)
    parser.add_argument("--move-limit", type=int, default=18)
    parser.add_argument("--tabu-tenure", type=int, default=250)
    args = parser.parse_args()

    for T in args.thresholds:
        print(f"\n{'=' * 60}")
        print(f"Threshold T={T}")
        print(f"{'=' * 60}")

        if args.mode == "tabu":
            best = tabu_search_allocations(
                threshold=T,
                seed=args.seed,
                scale=args.scale,
                initial=max(args.search, args.local_initial),
                rounds=args.local_rounds,
                beam=args.local_beam,
                neighborhood=args.neighborhood,
                move_limit=args.move_limit,
                tabu_tenure=args.tabu_tenure,
                refine_top=args.refine_top,
                group_time_limit=args.time_limit,
                jobs=args.jobs,
            )
            print(f"\nBest candidate: {best['label']}")
            print_summary(T, best['results'], best['total_ok'])
        elif args.mode == "swap":
            best = swap_search_allocations(
                threshold=T,
                seed=args.seed,
                scale=args.scale,
                initial=max(args.search, args.local_initial),
                rounds=args.local_rounds,
                beam=args.local_beam,
                mutations=args.local_mutations,
                swap_size=args.swap_size,
                refine_top=args.refine_top,
                group_time_limit=args.time_limit,
                jobs=args.jobs,
            )
            print(f"\nBest candidate: {best['label']}")
            print_summary(T, best['results'], best['total_ok'])
        elif args.mode == "local":
            best = local_search_allocations(
                threshold=T,
                seed=args.seed,
                scale=args.scale,
                initial=max(args.search, args.local_initial),
                rounds=args.local_rounds,
                beam=args.local_beam,
                mutations=args.local_mutations,
                refine_top=args.refine_top,
                group_time_limit=args.time_limit,
                jobs=args.jobs,
            )
            print(f"\nBest candidate: {best['label']}")
            print_summary(T, best['results'], best['total_ok'])
        elif args.search > 0:
            best = search_allocations(
                threshold=T,
                iterations=args.search,
                seed=args.seed,
                scale=args.scale,
                group_time_limit=args.time_limit,
                mode=args.mode,
                refine_top=args.refine_top,
                jobs=args.jobs,
            )
            print(f"\nBest candidate: {best['label']}")
            print_summary(T, best['results'], best['total_ok'])
        else:
            if args.mode == "milp":
                res, total_ok = solve_all_fast(T, group_time_limit=args.time_limit)
            else:
                res, total_ok, strategy = solve_all_greedy_best(T)
                print(f"\nBest greedy strategy: {strategy}")
            print_summary(T, res, total_ok)
