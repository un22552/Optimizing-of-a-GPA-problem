"""
gpa-fragmented.py — 多池整数规划（分层联合 + 正确边际约束）

层次结构:
  G2 (2池): y_p0_p4[a][b] 全联合, 170×170 = 28,900 变量
  G1a/G1b (3池):
    Level 1: y01[a][b]    联合池0和池1  (170×53 = 9,010)
    Level 2: y012[t][c]   联合(0+1总sum)和池2  (~222×25 = 5,550)
    
  边际一致性:
    z[t] = sum_{a+b=t} y01[a][b]        (内联计算)
    sum_c y012[t][c] == z[t]             (y012 → y01 耦合)
    sum_b y01[a][b] == x_pool0[a]        (pool0 边际)
    sum_a y01[a][b] == x_pool1[b]        (pool1 边际)
    sum_t y012[t][c] == x_pool2[c]       (pool2 边际)
"""

import itertools
from collections import Counter, defaultdict
import pulp

grades = ['C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
gpa_map = {g: v for g, v in zip(grades, [1.7,2.0,2.3,2.7,3.0,3.3,3.7,4.0,4.3])}
base = {'C-':4,'C':5,'C+':10,'B-':16,'B':20,'B+':19,'A-':15,'A':8,'A+':3}

POOLS = {
    0: {'name': '公选',  'k': 5, 'n': 100},
    1: {'name': '专业A', 'k': 3, 'n': 80},
    2: {'name': '专业B', 'k': 2, 'n': 50},
    3: {'name': '专业C', 'k': 2, 'n': 30},
    4: {'name': '专业D', 'k': 5, 'n': 20},
}

GROUPS = [
    {'name': 'G1a', 'size': 50, 'pools': [0, 1, 2], 'nc': 10},
    {'name': 'G1b', 'size': 30, 'pools': [0, 1, 3], 'nc': 10},
    {'name': 'G2',  'size': 20, 'pools': [0, 4],     'nc': 10},
]

def sc(n):
    s = {g: int(round(base[g]*n/100.0)) for g in grades}
    d = n - sum(s.values())
    if d:
        mg = max(grades, key=lambda g: s[g])
        s[mg] += d
    return s

# ---- 池数据 ----
pool_slots = {}; pool_combos = {}; pool_sums = {}; pool_si = {}; pool_cbs = {}
pool_groups = defaultdict(list)

for pid, p in POOLS.items():
    sl = sc(p['n'])
    pool_slots[pid] = {g: p['k']*sl[g] for g in grades}
    cb = []; ss = set(); sm = defaultdict(list)
    for combo in itertools.combinations_with_replacement(grades, p['k']):
        sg = sum(gpa_map[g] for g in combo)
        idx = len(cb)
        cb.append({'idx':idx,'combo':combo,'sum_gpa':sg,'counts':Counter(combo)})
        ss.add(sg); sm[sg].append(idx)
    pool_combos[pid] = cb
    sv = sorted(ss)
    pool_sums[pid] = sv
    pool_si[pid] = {s:i for i,s in enumerate(sv)}
    pool_cbs[pid] = dict(sm)

for gid, g in enumerate(GROUPS):
    for pid in g['pools']:
        pool_groups[pid].append(gid)


def solve(threshold=3.5, time_limit=300, verbose=True):
    prob = pulp.LpProblem("MF", pulp.LpMaximize)
    
    # w[pid][gid][c]
    w = {}
    for pid in POOLS:
        w[pid] = {}
        for gid in pool_groups[pid]:
            w[pid][gid] = [pulp.LpVariable(f"w_p{pid}_g{gid}_c{i}",0,None,'Integer')
                           for i in range(len(pool_combos[pid]))]
    
    # ---- 联合变量 ----
    y_all = {}   # gid -> flat list of y vars
    y_info = {}  # gid -> dict: key → (var, total_sum, per_pool_sums_tuple, avg)
    
    for gid, g in enumerate(GROUPS):
        pids = g['pools']
        
        if len(pids) == 2:
            # G2: 全联合 y04[a][b]
            s0, s1 = pool_sums[pids[0]], pool_sums[pids[1]]
            y_map = {}
            y_flat = []
            for a in s0:
                for b in s1:
                    key = (pool_si[pids[0]][a], pool_si[pids[1]][b])
                    t = a + b
                    avg = t / g['nc']
                    var = pulp.LpVariable(f"y_g{gid}_{key[0]}_{key[1]}",0,None,'Integer')
                    y_map[key] = (var, t, (a,b), avg)
                    y_flat.append(var)
            y_all[gid] = y_flat
            y_info[gid] = y_map
        else:
            # G1a/G1b: 分层
            s0, s1, s2 = pool_sums[pids[0]], pool_sums[pids[1]], pool_sums[pids[2]]
            
            # Level 1: y01
            y01 = {}
            sum01_to_keys = defaultdict(list)  # t_ab → [(key, var)]
            for a in s0:
                for b in s1:
                    key = (pool_si[pids[0]][a], pool_si[pids[1]][b])
                    t = a + b
                    var = pulp.LpVariable(f"y01_g{gid}_{key[0]}_{key[1]}",0,None,'Integer')
                    y01[key] = var
                    sum01_to_keys[t].append((key, var))
            
            # Level 2: y012[t][c]
            y012 = {}
            y_map = {}  # unified key → (var, total, sums, avg)
            y_flat = []
            # index t values
            t_vals = sorted(set(a+b for a,b in itertools.product(s0,s1)))
            t_si = {t:i for i,t in enumerate(t_vals)}
            
            for t in t_vals:
                for c in s2:
                    key012 = (t_si[t], pool_si[pids[2]][c])
                    total = t + c
                    avg = total / g['nc']
                    var = pulp.LpVariable(f"y012_g{gid}_{key012[0]}_{key012[1]}",0,None,'Integer')
                    y012[(t,c)] = var
                    y_map[key012] = (var, total, (None,None,c), avg)  # fills in later
                    y_flat.append(var)
            
            # Link y01 ↔ y012:
            # For each t = a+b: sum_c y012[t][c] == sum_{a+b=t} y01[a][b]
            for t in t_vals:
                y012_terms = [y012[(t,c)] for c in s2]
                y01_terms = [var for _, var in sum01_to_keys.get(t, [])]
                
                if y012_terms and y01_terms:
                    prob += pulp.lpSum(y012_terms) == pulp.lpSum(y01_terms), \
                            f"lk01_g{gid}_t{t_si[t]}"
                elif y012_terms:
                    prob += pulp.lpSum(y012_terms) == 0, f"lk01z_g{gid}_t{t_si[t]}"
                elif y01_terms:
                    prob += pulp.lpSum(y01_terms) == 0, f"lk01y_g{gid}_t{t_si[t]}"
            
            # Build unified y_info
            full_y_map = {}
            # For y01 entries: we need to link them to the marginal sum
            # but these are not directly used in w linking
            # w linking goes through y012 for pool2, y01 for pools 0 and 1
            
            # Store y01 separately
            y_info[gid] = {
                'type': 'hier3',
                'y01': y01,
                'y012': y012,
                's0': s0, 's1': s1, 's2': s2,
                'si0': pool_si[pids[0]], 'si1': pool_si[pids[1]], 'si2': pool_si[pids[2]],
                't_vals': t_vals, 't_si': t_si,
            }
            
            # y_all for size constraint: y01 vars (each student appears exactly once)
            y_all[gid] = list(y01.values())
    
    # ---- 目标 ----
    obj = []
    for gid, g in enumerate(GROUPS):
        pids = g['pools']
        if len(pids) == 2:
            for key, (var, t, sums, avg) in y_info[gid].items():
                if avg >= threshold:
                    obj.append(var)
        else:
            yi = y_info[gid]
            y01 = yi['y01']
            y012 = yi['y012']
            s0, s1, s2 = yi['s0'], yi['s1'], yi['s2']
            # For 3-pool: rebuild avg from pairs
            for (a,b), var in y01.items():
                total_sum = a + b
                for c in s2:
                    t = total_sum + c
                    avg = t / g['nc']
                    if avg >= threshold:
                        # This is an issue: y01 doesn't know about c.
                        # Need to use y012 for objective, or sum over decompositions
                        pass
            # Use y012 for objective
            for (t_val, c), var in y012.items():
                total = t_val + c
                avg = total / g['nc']
                if avg >= threshold:
                    obj.append(var)
    prob += pulp.lpSum(obj)
    
    # ---- 每组总人数 ----
    for gid, g in enumerate(GROUPS):
        prob += pulp.lpSum(y_all[gid]) == g['size'], f"sz_g{gid}"
    
    # ---- w ↔ y 边际耦合 ----
    for gid, g in enumerate(GROUPS):
        pids = g['pools']
        
        if len(pids) == 2:
            # G2
            yi = y_info[gid]
            for pi, pid in enumerate(pids):
                s_list = pool_sums[pid]
                s_idx_map = pool_si[pid]
                cbs = pool_cbs[pid]
                
                for s_val, s_idx in s_idx_map.items():
                    c_indices = cbs.get(s_val, [])
                    lhs = pulp.lpSum(w[pid][gid][ci] for ci in c_indices) if c_indices else 0
                    
                    rhs_terms = []
                    for key, (var, _, sums, _) in yi.items():
                        if abs(sums[pi] - s_val) < 0.001:
                            rhs_terms.append(var)
                    
                    if c_indices or rhs_terms:
                        rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
                        prob += lhs == rhs, f"wy_g{gid}_p{pid}_si{s_idx}"
        else:
            # G1a/G1b
            yi = y_info[gid]
            y01 = yi['y01']
            y012 = yi['y012']
            s0, s1, s2 = yi['s0'], yi['s1'], yi['s2']
            si0, si1, si2 = yi['si0'], yi['si1'], yi['si2']
            
            # Pool 0: sum_b y01[a][b] = x_pool0[a]
            for s_val, s_idx in si0.items():
                c_indices = pool_cbs[pids[0]].get(s_val, [])
                lhs = pulp.lpSum(w[pids[0]][gid][ci] for ci in c_indices) if c_indices else 0
                rhs_terms = []
                for b in s1:
                    key = (s_idx, si1[b])
                    if key in y01:
                        rhs_terms.append(y01[key])
                rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
                if lhs != 0 or rhs_terms:
                    prob += lhs == rhs, f"wy_g{gid}_p0_si{s_idx}"
            
            # Pool 1: sum_a y01[a][b] = x_pool1[b]
            for s_val, s_idx in si1.items():
                c_indices = pool_cbs[pids[1]].get(s_val, [])
                lhs = pulp.lpSum(w[pids[1]][gid][ci] for ci in c_indices) if c_indices else 0
                rhs_terms = []
                for a in s0:
                    key = (si0[a], s_idx)
                    if key in y01:
                        rhs_terms.append(y01[key])
                rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
                if lhs != 0 or rhs_terms:
                    prob += lhs == rhs, f"wy_g{gid}_p1_si{s_idx}"
            
            # Pool 2/3: sum_t y012[t][c] = x_pool2[c]
            for s_val, s_idx in si2.items():
                c_indices = pool_cbs[pids[2]].get(s_val, [])
                lhs = pulp.lpSum(w[pids[2]][gid][ci] for ci in c_indices) if c_indices else 0
                rhs_terms = []
                for t_val in yi['t_vals']:
                    key = (t_val, s_val)
                    if key in y012:
                        rhs_terms.append(y012[key])
                rhs = pulp.lpSum(rhs_terms) if rhs_terms else 0
                if lhs != 0 or rhs_terms:
                    prob += lhs == rhs, f"wy_g{gid}_p{pid}_si{s_idx}"
    
    # ---- 槽位约束 ----
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
    
    nw = sum(len(w[p][g]) for p in POOLS for g in pool_groups[p])
    ny = sum(len(y_all[g]) for g in range(len(GROUPS)))
    if verbose:
        print(f"变量: w={nw} y={ny}, 总计={nw+ny}, 约束={len(prob.constraints)}")
        print("求解中...")
    
    prob.solve(pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit))
    
    st = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    max_s = int(round(obj_val)) if obj_val is not None else 0
    
    Gbar, Gmin = 3.061, 1.7
    ceil = min(1.0, (Gbar-Gmin)/(threshold-Gmin)) if threshold > Gmin else 1.0
    
    # 收集
    gd = {}
    for gid, g in enumerate(GROUPS):
        pids = g['pools']
        items = []; ok = 0
        
        if len(pids) == 2:
            yi = y_info[gid]
            for key, (var, t, sums, avg) in yi.items():
                n = int(round(pulp.value(var)))
                if n > 0:
                    succ = avg >= threshold
                    if succ: ok += n
                    items.append({'n':n,'avg':avg,'total':t,'sums':sums,'ok':succ})
        else:
            yi = y_info[gid]
            y012 = yi['y012']
            for (t_val, c), var in y012.items():
                n = int(round(pulp.value(var)))
                if n > 0:
                    total = t_val + c
                    avg = total / g['nc']
                    # Need per-pool sums; 0+1 combined = t_val, pool2 = c
                    # For display, show (combined=t_val, pool2=c)
                    succ = avg >= threshold
                    if succ: ok += n
                    sums_display = (t_val, c)
                    items.append({'n':n,'avg':avg,'total':total,'sums':sums_display,'ok':succ})
        
        gd[gid] = {'size':g['size'],'ok':ok,'items':sorted(items,key=lambda x:-x['n'])}
    
    return {'status':st,'threshold':threshold,'max_s':max_s,'ratio':max_s/100.0,
            'ceiling':ceil,'groups':gd,'w':w}


def show(r):
    print(f"\n{'='*60}\n结果 | 状态={r['status']} | 阈值={r['threshold']}")
    print(f"达标: {r['max_s']}/100 ({r['ratio']:.2%})  天花板={r['ceiling']:.2%}  差距={r['ceiling']-r['ratio']:.2%}")
    for gid, g in enumerate(GROUPS):
        gd = r['groups'][gid]
        print(f"\n{'─'*40}\n组{g['name']}({g['size']}人 {g['nc']}门): 达标{gd['ok']}/{g['size']} ({gd['ok']/g['size']:.1%})")
        for item in gd['items'][:7]:
            st = "OK" if item['ok'] else "--"
            print(f"  [{st}] {item['n']:2d}人 均={item['avg']:.3f}  sum={item['total']:.1f}  {item['sums']}")
    # combo
    w = r['w']
    for gid, g in enumerate(GROUPS):
        print(f"\n  组{g['name']} combo:")
        for pid in g['pools']:
            p=POOLS[pid]; k=p['k']
            a=[(int(round(pulp.value(w[pid][gid][c['idx']]))),c) for c in pool_combos[pid]]
            a=[x for x in a if x[0]>0]; a.sort(key=lambda x:-x[0])
            print(f"  池{pid} {p['name']}({k}门):")
            for n,c in a[:4]: print(f"      {n:2d} sum={c['sum_gpa']:.1f} 均={c['sum_gpa']/k:.2f} [{','.join(c['combo'])}]")
            if len(a)>4: print(f"      ... 还有{len(a)-4}种")


if __name__ == '__main__':
    print("="*50+"\n多池槽位\n"+"="*50)
    for pid,p in POOLS.items():
        sl=pool_slots[pid]
        print(f"  {p['name']}({p['k']}门×{p['n']}人): 总={sum(sl.values())}  "+" ".join(f"{g}={sl[g]}" for g in grades))
    
    r = solve(threshold=3.5, time_limit=300, verbose=True)
    show(r)
    
    print(f"\n{'='*50} 对比 {'='*50}")
    try:
        import importlib; m=importlib.import_module('gpa-1')
        r10=m.a(3.5,num_courses=10,total_students=100)
        r5=m.a(3.5,num_courses=5,total_students=100)
        print(f"  无差异 M=10: {r10['max_students']}人 ({r10['max_ratio']:.2%})")
        print(f"  无差异 M=5:  {r5['max_students']}人 ({r5['max_ratio']:.2%})")
        print(f"  碎片化:      {r['max_s']}人 ({r['ratio']:.2%})")
        print(f"  代价:        {r10['max_ratio']-r['ratio']:.2%}")
    except Exception as e: print(f"  无法对比: {e}")
