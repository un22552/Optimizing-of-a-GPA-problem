"""
gpa-phase-transition.py — 二分搜索精确锁定相变临界权重

已知:
  25:75 (w35=1, w33=3) → 精英模式 (cnt_35=175)
  20:80 (w35=1, w33=4) → 普惠模式 (cnt_35=65)

临界点 r* = w35/(w35+w33) 在 20%~25% 之间。
二分搜索找到精英<->普惠跳变的精确权重比。
"""
import itertools
from collections import Counter, defaultdict
import pulp
import sys

# ── 全局常量 ──
grades = ['C-','C','C+','B-','B','B+','A-','A','A+']
gpa_map = {g:v for g,v in zip(grades,[1.7,2.0,2.3,2.7,3.0,3.3,3.7,4.0,4.3])}
base = {'C-':4,'C':5,'C+':10,'B-':16,'B':20,'B+':19,'A-':15,'A':8,'A+':3}
TOTAL = 300; TH = 3.5; TL = 3.3; SDP = 1.0

def rnd(x): return round(round(x/SDP)*SDP,10)
def sc(n):
    s = {g:int(round(base[g]*n/100.0)) for g in grades}
    d = n-sum(s.values())
    if d: mg = max(grades,key=lambda g:s[g]); s[mg] += d
    return s

POOLS = {
    '0a':{'k':5,'n':300,'groups':[0,1,2]},'0b':{'k':5,'n':300,'groups':[0,1,2]},
    '1':{'k':4,'n':250,'groups':[0,1]},'2':{'k':3,'n':150,'groups':[1,2]},
    '3a':{'k':5,'n':150,'groups':[0]},'3b':{'k':5,'n':150,'groups':[0]},
    '4':{'k':7,'n':100,'groups':[1]},'5':{'k':7,'n':50,'groups':[2]},'6':{'k':8,'n':50,'groups':[2]},
}
GROUPS = [
    {'name':'G1','size':150,'pools':['0a','0b','1','3a','3b'],'nc':24},
    {'name':'G2','size':100,'pools':['0a','0b','1','2','4'],'nc':24},
    {'name':'G3','size':50,'pools':['0a','0b','2','5','6'],'nc':28},
]

# ── 预处理 (只做一次) ──
print("预处理课程组合...", end=' ', flush=True)
pool_combos = {}; pool_sums = {}; pool_si = {}; pool_cbs = {}; pool_slots = {}
for pid,p in POOLS.items():
    k = p['k']; slots = sc(p['n']); pool_slots[pid] = {g:k*slots[g] for g in grades}
    cb = []; ss = set(); sm = defaultdict(list)
    for combo in itertools.combinations_with_replacement(grades,k):
        sg = rnd(sum(gpa_map[g] for g in combo)); idx = len(cb)
        cb.append({'idx':idx,'combo':combo,'sum_gpa':sg,'counts':Counter(combo)})
        ss.add(sg); sm[sg].append(idx)
    pool_combos[pid] = cb
    sv = sorted(ss); pool_sums[pid] = sv
    pool_si[pid] = {s:i for i,s in enumerate(sv)}; pool_cbs[pid] = dict(sm)
pool_groups = defaultdict(list)
for gid,g in enumerate(GROUPS):
    for pid in g['pools']: pool_groups[pid].append(gid)
print(f"完成 ({sum(len(v) for v in pool_combos.values())} combos)")

# ── 模型构建 (只做一次, 返回 prob + 目标项引用) ──
def build_model():
    prob = pulp.LpProblem('M', pulp.LpMaximize)
    w = {}
    for pid in POOLS:
        w[pid] = {}
        for gid in pool_groups[pid]:
            w[pid][gid] = [pulp.LpVariable(f'w{pid}_g{gid}_c{i}',0,None,'Integer')
                           for i in range(len(pool_combos[pid]))]
    all_y_info = {}; all_y_flat = {}
    for gid in range(len(GROUPS)):
        g = GROUPS[gid]; pids = g['pools']; n_pools = len(pids)
        sums = [pool_sums[pid] for pid in pids]; sis = [pool_si[pid] for pid in pids]
        y_layers = []; y_all_flat = []
        y_l1 = {}
        for a in sums[0]:
            for b in sums[1]:
                var = pulp.LpVariable(f'y_g{gid}_l1_{sis[0][a]}_{sis[1][b]}',0,None,'Integer')
                y_l1[(a,b)] = var; y_all_flat.append(var)
        y_layers.append(y_l1)
        prev_sums = sorted(set(rnd(a+b) for a in sums[0] for b in sums[1]))
        prev_si = {s:i for i,s in enumerate(prev_sums)}
        prev_to_ab = defaultdict(list)
        for a in sums[0]:
            for b in sums[1]: prev_to_ab[rnd(a+b)].append((a,b))
        cur_prev_sums, cur_prev_si, prev_y_dict = prev_sums, prev_si, y_l1
        for li in range(2, n_pools):
            sn, sin = sums[li], sis[li]; y_cur = {}
            new_sums_set = set()
            for ps in cur_prev_sums:
                for ns in sn: new_sums_set.add(rnd(ps+ns))
            new_sums = sorted(new_sums_set); new_si = {s:i for i,s in enumerate(new_sums)}
            for ps in cur_prev_sums:
                for ns in sn:
                    var = pulp.LpVariable(f'y_g{gid}_l{li}_{cur_prev_si[ps]}_{sin[ns]}',0,None,'Integer')
                    y_cur[(ps,ns)] = var
            if li == 2:
                for ps in cur_prev_sums:
                    lhs = [y_cur[(ps,ns)] for ns in sn if (ps,ns) in y_cur]
                    rhs = [prev_y_dict[(a,b)] for (a,b) in prev_to_ab.get(ps,[])]
                    if lhs: prob += pulp.lpSum(lhs) == (pulp.lpSum(rhs) if rhs else 0), f'lk_g{gid}_l{li}_ps{cur_prev_si[ps]}'
                    elif rhs: prob += pulp.lpSum(rhs) == 0, f'lkz2_g{gid}_l{li}_ps{cur_prev_si[ps]}'
            else:
                rev_map = defaultdict(list)
                for (pps,pcs),var in prev_y_dict.items(): rev_map[rnd(pps+pcs)].append((pps,pcs))
                for ps in cur_prev_sums:
                    lhs = [y_cur[(ps,ns)] for ns in sn if (ps,ns) in y_cur]
                    rhs_parts = [prev_y_dict[key] for key in rev_map.get(ps,[])]
                    if lhs: prob += pulp.lpSum(lhs) == (pulp.lpSum(rhs_parts) if rhs_parts else 0), f'lk_g{gid}_l{li}_ps{cur_prev_si[ps]}'
                    elif rhs_parts: prob += pulp.lpSum(rhs_parts) == 0, f'lkz2_g{gid}_l{li}_ps{cur_prev_si[ps]}'
            y_layers.append(y_cur); prev_y_dict = y_cur
            if li < n_pools-1: cur_prev_sums, cur_prev_si = new_sums, new_si
        all_y_info[gid] = (y_layers, {'pids':pids,'sums':sums,'sis':sis,'final_layer':len(y_layers)-1})
        all_y_flat[gid] = y_all_flat
    for gid,g in enumerate(GROUPS): prob += pulp.lpSum(all_y_flat[gid]) == g['size'], f'sz_g{gid}'
    for gid,g in enumerate(GROUPS):
        pids = g['pools']; n_pools = len(pids)
        y_layers, info = all_y_info[gid]; sums, sis = info['sums'], info['sis']
        for pi, pid in enumerate(pids):
            s_idx_map = sis[pi]; cbs = pool_cbs[pid]
            for s_val, s_idx in s_idx_map.items():
                c_indices = cbs.get(s_val, [])
                lhs = pulp.lpSum(w[pid][gid][ci] for ci in c_indices) if c_indices else 0
                rhs_terms = []
                yl = y_layers
                if n_pools <= 2:
                    for (a,b),var in yl[0].items():
                        if pi == 0 and abs(a-s_val) < 0.001: rhs_terms.append(var)
                        elif n_pools == 2 and pi == 1 and abs(b-s_val) < 0.001: rhs_terms.append(var)
                else:
                    for (a,b),var in yl[0].items():
                        if pi <= 1 and abs((a if pi==0 else b)-s_val) < 0.001: rhs_terms.append(var)
                    for li2 in range(1, n_pools-1):
                        for (ps,ns),var in yl[li2].items():
                            if pi == li2+1 and abs(ns-s_val) < 0.001: rhs_terms.append(var)
                rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
                if lhs != 0 or rhs_terms: prob += lhs == rhs, f'wy_g{gid}_p{pid}_si{s_idx}'
    for pid in POOLS:
        sl = pool_slots[pid]
        for gi, gr in enumerate(grades):
            terms = []
            for gid in pool_groups[pid]:
                for c in pool_combos[pid]:
                    cnt = c['counts'].get(gr,0)
                    if cnt > 0: terms.append(cnt * w[pid][gid][c['idx']])
            if terms: prob += pulp.lpSum(terms) <= sl[gr], f'sl_p{pid}_gi{gi}'
    obj_35 = []; obj_33 = []
    for gid,g in enumerate(GROUPS):
        nc = g['nc']; y_layers, info = all_y_info[gid]
        final_layer = y_layers[info['final_layer']]
        for key, var in final_layer.items():
            total = rnd(key[0]+key[1]) if isinstance(key,tuple) and len(key)==2 else key
            avg = total / nc
            if avg >= TH-1e-9: obj_35.append(var)
            if avg >= TL-1e-9: obj_33.append(var)
    obj_35_sum = pulp.lpSum(obj_35); obj_33_sum = pulp.lpSum(obj_33)
    return prob, obj_35_sum, obj_33_sum, obj_35, obj_33

print("模型将在每次求解时重建 (每次 ~20s 构建 + ~2-4min 求解)\n")

def solve_weight(w35, w33, time_limit=300):
    """每次重建模型并求解——避免 PuLP 目标函数修改的副作用"""
    import time
    prob2, o35s, o33s, o35v, o33v = build_model()
    prob2 += w35 * o35s + w33 * o33s
    t0 = time.time()
    prob2.solve(pulp.HiGHS(msg=False, time_limit=time_limit))
    elapsed = time.time() - t0
    c35 = int(round(sum(pulp.value(v) or 0 for v in o35v)))
    c33 = int(round(sum(pulp.value(v) or 0 for v in o33v)))
    return c35, c33, pulp.LpStatus[prob2.status], elapsed

def classify(c35):
    """精英模式: c35 > 120, 普惠模式: c35 < 80, 中间: 过渡"""
    if c35 > 120: return 'ELITE'
    elif c35 < 80: return 'UNIVERSAL'
    else: return 'TRANSITION'

# ── 二分搜索 ──
# 在有理数空间搜索: 固定分母范围, 二分分子
# 权重比 r = w35/(w35+w33), 精英模式在 r=25% (1:3), 普惠在 r=20% (1:4)
# 用 Farey 序列思路: 在 (1/5, 1/2) 区间二分

print("\n" + "=" * 80)
print("相变临界权重二分搜索")
print("=" * 80)
print(f"{'w35:w33':>8s} {'ratio':>7s} {'cnt_35':>6s} {'cnt_33':>6s} {'[3.3,3.5)':>8s} {'模式':>12s} {'时间':>6s}")
print("-" * 80)

# 初始边界: 已知精英(1:3)和普惠(1:4)
# 为了更精细的搜索, 在 [1/5, 1/2] = [20%, 33%] 区间搜索
# lo = 1/5 边界(普惠), hi = 1/3 边界(精英)
# 用两个有理数表示: lo_num/lo_den, hi_num/hi_den
lo_num, lo_den = 1, 5   # 1/5 = 20% (普惠): w35=1, w33=4
hi_num, hi_den = 1, 3   # 1/3 ≈ 33% (精英): w35=1, w33=2

results = []

# 先跑几个锚点确认已知结果
anchors = [(1,4,'1:4 20%'), (1,3,'1:3 25%'), (2,7,'2:7 22.2%'), (3,7,'3:7 30%'), (3,11,'3:11 21.4%')]
for w35, w33, label in anchors:
    c35, c33, st, t = solve_weight(w35, w33)
    mode = classify(c35)
    r = w35 / (w35 + w33)
    results.append((w35, w33, r, c35, c33, mode, t))
    print(f"  {w35:>2d}:{w33:<3d} {r:>6.1%} {c35:>5d}人 {c33:>5d}人 {c33-c35:>7d}人 {mode:>12s} {t:>5.0f}s")
    # 更新二分边界
    if mode == 'ELITE' and r < (hi_num/hi_den):
        hi_num, hi_den = w35, w35 + w33
    if mode == 'UNIVERSAL' and r > (lo_num/lo_den):
        lo_num, lo_den = w35, w35 + w33

# 正式二分: 在 (lo, hi) 之间搜索, 最多 6 步
print(f"\n二分搜索: lo={lo_num}/{lo_den}={lo_num/lo_den:.1%}, hi={hi_num}/{hi_den}={hi_num/hi_den:.1%}")
print("-" * 80)

for iteration in range(6):
    # Farey mediant
    mid_num = lo_num + hi_num
    mid_den = lo_den + hi_den
    w35, w33 = mid_num, mid_den - mid_num

    if (w35, w33) in [(r[0], r[1]) for r in results]:
        # 已经算过, 微调
        w35 += 1; w33 = mid_den - mid_num + (mid_num - w35)

    c35, c33, st, t = solve_weight(w35, w33)
    mode = classify(c35)
    r = w35 / (w35 + w33)
    results.append((w35, w33, r, c35, c33, mode, t))
    print(f"  {w35:>2d}:{w33:<3d} {r:>6.1%} {c35:>5d}人 {c33:>5d}人 {c33-c35:>7d}人 {mode:>12s} {t:>5.0f}s")

    if mode == 'ELITE':
        hi_num, hi_den = mid_num, mid_den
    elif mode == 'UNIVERSAL':
        lo_num, lo_den = mid_num, mid_den
    else:
        # 过渡模式: 缩小范围
        hi_num, hi_den = mid_num, mid_den

    gap = hi_num/hi_den - lo_num/lo_den
    if gap < 0.005:  # 精度 < 0.5%
        break

# ── 汇总 ──
print("\n" + "=" * 80)
print("相变总结")
print("=" * 80)
results.sort(key=lambda x: x[2])  # 按 ratio 排序
prev_mode = None
for w35, w33, r, c35, c33, mode, t in results:
    marker = ""
    if prev_mode and prev_mode != mode:
        marker = " <-- 相变临界区间"
    print(f"  {w35:>2d}:{w33:<3d} {r:>6.1%} {c35:>5d}人 {c33:>5d}人 {mode:>12s}{marker}")
    prev_mode = mode

# 找出临界区间
elite_pts = [r for r in results if r[5]=='ELITE']
univ_pts = [r for r in results if r[5]=='UNIVERSAL']
trans_pts = [r for r in results if r[5]=='TRANSITION']

if elite_pts and univ_pts:
    r_elite_min = min(r[2] for r in elite_pts)
    r_univ_max = max(r[2] for r in univ_pts)
    print(f"\n  精英模式最低权重: {r_elite_min:.1%}")
    print(f"  普惠模式最高权重: {r_univ_max:.1%}")
    print(f"  临界区间: ({r_univ_max:.1%}, {r_elite_min:.1%}]")
    print(f"  临界点估计: r* ≈ {(r_univ_max + r_elite_min)/2:.1%}")
    print(f"\n  解读: 当 3.5 权重 ≤ {r_univ_max:.0%} 时, 求解器选择 '普惠' 策略 (牺牲 3.5 保 3.3)")
    print(f"        当 3.5 权重 ≥ {r_elite_min:.0%} 时, 求解器选择 '精英' 策略 (保 3.5)")
