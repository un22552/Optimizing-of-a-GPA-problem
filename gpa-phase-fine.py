"""
gpa-phase-fine.py — 细扫 22.2%–25.0% 区间, 描出完整的相变过渡曲线
"""
import itertools, time
from collections import Counter, defaultdict
import pulp

grades = ['C-','C','C+','B-','B','B+','A-','A','A+']
gpa_map = {g:v for g,v in zip(grades,[1.7,2.0,2.3,2.7,3.0,3.3,3.7,4.0,4.3])}
base = {'C-':4,'C':5,'C+':10,'B-':16,'B':20,'B+':19,'A-':15,'A':8,'A+':3}
TOTAL=300; TH=3.5; TL=3.3; SDP=1.0
def rnd(x): return round(round(x/SDP)*SDP,10)
def sc(n):
    s={g:int(round(base[g]*n/100.0)) for g in grades}
    d=n-sum(s.values())
    if d: mg=max(grades,key=lambda g:s[g]); s[mg]+=d
    return s

POOLS={
    '0a':{'k':5,'n':300,'groups':[0,1,2]},'0b':{'k':5,'n':300,'groups':[0,1,2]},
    '1':{'k':4,'n':250,'groups':[0,1]},'2':{'k':3,'n':150,'groups':[1,2]},
    '3a':{'k':5,'n':150,'groups':[0]},'3b':{'k':5,'n':150,'groups':[0]},
    '4':{'k':7,'n':100,'groups':[1]},'5':{'k':7,'n':50,'groups':[2]},'6':{'k':8,'n':50,'groups':[2]},
}
GROUPS=[
    {'name':'G1','size':150,'pools':['0a','0b','1','3a','3b'],'nc':24},
    {'name':'G2','size':100,'pools':['0a','0b','1','2','4'],'nc':24},
    {'name':'G3','size':50,'pools':['0a','0b','2','5','6'],'nc':28},
]

print("预处理...", end=' ', flush=True)
pool_combos={}; pool_sums={}; pool_si={}; pool_cbs={}; pool_slots={}
for pid,p in POOLS.items():
    k=p['k']; slots=sc(p['n']); pool_slots[pid]={g:k*slots[g] for g in grades}
    cb=[]; ss=set(); sm=defaultdict(list)
    for combo in itertools.combinations_with_replacement(grades,k):
        sg=rnd(sum(gpa_map[g] for g in combo)); idx=len(cb)
        cb.append({'idx':idx,'combo':combo,'sum_gpa':sg,'counts':Counter(combo)})
        ss.add(sg); sm[sg].append(idx)
    pool_combos[pid]=cb; sv=sorted(ss); pool_sums[pid]=sv
    pool_si[pid]={s:i for i,s in enumerate(sv)}; pool_cbs[pid]=dict(sm)
pool_groups=defaultdict(list)
for gid,g in enumerate(GROUPS):
    for pid in g['pools']: pool_groups[pid].append(gid)
print("完成")

def build_model():
    prob=pulp.LpProblem('M',pulp.LpMaximize)
    w={}
    for pid in POOLS:
        w[pid]={}
        for gid in pool_groups[pid]:
            w[pid][gid]=[pulp.LpVariable(f'w{pid}_g{gid}_c{i}',0,None,'Integer')
                         for i in range(len(pool_combos[pid]))]
    all_y_info={}; all_y_flat={}
    for gid in range(len(GROUPS)):
        g=GROUPS[gid]; pids=g['pools']; n_pools=len(pids)
        sums=[pool_sums[pid] for pid in pids]; sis=[pool_si[pid] for pid in pids]
        y_layers=[]; y_all_flat=[]
        y_l1={}
        for a in sums[0]:
            for b in sums[1]:
                var=pulp.LpVariable(f'y_g{gid}_l1_{sis[0][a]}_{sis[1][b]}',0,None,'Integer')
                y_l1[(a,b)]=var; y_all_flat.append(var)
        y_layers.append(y_l1)
        prev_sums=sorted(set(rnd(a+b) for a in sums[0] for b in sums[1]))
        prev_si={s:i for i,s in enumerate(prev_sums)}
        prev_to_ab=defaultdict(list)
        for a in sums[0]:
            for b in sums[1]: prev_to_ab[rnd(a+b)].append((a,b))
        cur_prev_sums,cur_prev_si,prev_y_dict=prev_sums,prev_si,y_l1
        for li in range(2,n_pools):
            sn,sin=sums[li],sis[li]; y_cur={}
            new_sums_set=set()
            for ps in cur_prev_sums:
                for ns in sn: new_sums_set.add(rnd(ps+ns))
            new_sums=sorted(new_sums_set); new_si={s:i for i,s in enumerate(new_sums)}
            for ps in cur_prev_sums:
                for ns in sn:
                    var=pulp.LpVariable(f'y_g{gid}_l{li}_{cur_prev_si[ps]}_{sin[ns]}',0,None,'Integer')
                    y_cur[(ps,ns)]=var
            if li==2:
                for ps in cur_prev_sums:
                    lhs=[y_cur[(ps,ns)] for ns in sn if (ps,ns) in y_cur]
                    rhs=[prev_y_dict[(a,b)] for (a,b) in prev_to_ab.get(ps,[])]
                    if lhs: prob+=pulp.lpSum(lhs)==(pulp.lpSum(rhs) if rhs else 0),f'lk_g{gid}_l{li}_ps{cur_prev_si[ps]}'
                    elif rhs: prob+=pulp.lpSum(rhs)==0,f'lkz2_g{gid}_l{li}_ps{cur_prev_si[ps]}'
            else:
                rev_map=defaultdict(list)
                for (pps,pcs),var in prev_y_dict.items(): rev_map[rnd(pps+pcs)].append((pps,pcs))
                for ps in cur_prev_sums:
                    lhs=[y_cur[(ps,ns)] for ns in sn if (ps,ns) in y_cur]
                    rhs_parts=[prev_y_dict[key] for key in rev_map.get(ps,[])]
                    if lhs: prob+=pulp.lpSum(lhs)==(pulp.lpSum(rhs_parts) if rhs_parts else 0),f'lk_g{gid}_l{li}_ps{cur_prev_si[ps]}'
                    elif rhs_parts: prob+=pulp.lpSum(rhs_parts)==0,f'lkz2_g{gid}_l{li}_ps{cur_prev_si[ps]}'
            y_layers.append(y_cur); prev_y_dict=y_cur
            if li<n_pools-1: cur_prev_sums,cur_prev_si=new_sums,new_si
        all_y_info[gid]=(y_layers,{'pids':pids,'sums':sums,'sis':sis,'final_layer':len(y_layers)-1})
        all_y_flat[gid]=y_all_flat
    for gid,g in enumerate(GROUPS): prob+=pulp.lpSum(all_y_flat[gid])==g['size'],f'sz_g{gid}'
    for gid,g in enumerate(GROUPS):
        pids=g['pools']; n_pools=len(pids)
        y_layers,info=all_y_info[gid]; sums,sis=info['sums'],info['sis']
        for pi,pid in enumerate(pids):
            s_idx_map=sis[pi]; cbs=pool_cbs[pid]
            for s_val,s_idx in s_idx_map.items():
                c_indices=cbs.get(s_val,[])
                lhs=pulp.lpSum(w[pid][gid][ci] for ci in c_indices) if c_indices else 0
                rhs_terms=[]
                yl=y_layers
                if n_pools<=2:
                    for (a,b),var in yl[0].items():
                        if pi==0 and abs(a-s_val)<0.001: rhs_terms.append(var)
                        elif n_pools==2 and pi==1 and abs(b-s_val)<0.001: rhs_terms.append(var)
                else:
                    for (a,b),var in yl[0].items():
                        if pi<=1 and abs((a if pi==0 else b)-s_val)<0.001: rhs_terms.append(var)
                    for li2 in range(1,n_pools-1):
                        for (ps,ns),var in yl[li2].items():
                            if pi==li2+1 and abs(ns-s_val)<0.001: rhs_terms.append(var)
                rhs=pulp.lpSum(rhs_terms) if rhs_terms else 0
                if lhs!=0 or rhs_terms: prob+=lhs==rhs,f'wy_g{gid}_p{pid}_si{s_idx}'
    for pid in POOLS:
        sl=pool_slots[pid]
        for gi,gr in enumerate(grades):
            terms=[]
            for gid in pool_groups[pid]:
                for c in pool_combos[pid]:
                    cnt=c['counts'].get(gr,0)
                    if cnt>0: terms.append(cnt*w[pid][gid][c['idx']])
            if terms: prob+=pulp.lpSum(terms)<=sl[gr],f'sl_p{pid}_gi{gi}'
    obj_35=[]; obj_33=[]
    for gid,g in enumerate(GROUPS):
        nc=g['nc']; y_layers,info=all_y_info[gid]; final_layer=y_layers[info['final_layer']]
        for key,var in final_layer.items():
            total=rnd(key[0]+key[1]) if isinstance(key,tuple) and len(key)==2 else key
            avg=total/nc
            if avg>=TH-1e-9: obj_35.append(var)
            if avg>=TL-1e-9: obj_33.append(var)
    return prob, pulp.lpSum(obj_35), pulp.lpSum(obj_33), obj_35, obj_33

# ── 细扫权重列表 (在 22.2% ~ 25.0% 之间) ──
# 同时包含两个边界外侧的点做锚
weights = [
    (2, 7,  '22.22% (锚: 过渡)'),
    (5, 17, '22.73%'),
    (3, 10, '23.08%'),
    (7, 23, '23.33%'),
    (4, 13, '23.53%'),
    (9, 29, '23.68%'),
    (5, 16, '23.81%'),
    (6, 19, '24.00%'),
    (7, 22, '24.14%'),
    (8, 25, '24.24%'),
    (1, 3,  '25.00% (锚: 精英)'),
]

print(f"\n{'='*80}")
print(f"细扫 [{22.2}%, {25.0}%] 区间 — {len(weights)} 个点")
print(f"{'='*80}")
print(f"{'w35:w33':>8s} {'ratio':>7s} {'cnt_35':>6s} {'cnt_33':>6s} {'[3.3,3.5)':>8s} {'模式':>14s} {'时间':>6s}")
print("-"*80)

results = []
for w35, w33, label in weights:
    t0 = time.time()
    prob, o35s, o33s, o35v, o33v = build_model()
    prob += w35 * o35s + w33 * o33s
    prob.solve(pulp.HiGHS(msg=False, time_limit=300))
    elapsed = time.time() - t0
    c35 = int(round(sum(pulp.value(v) or 0 for v in o35v)))
    c33 = int(round(sum(pulp.value(v) or 0 for v in o33v)))
    r = w35/(w35+w33)
    # 分类
    if c35 > 150: mode = 'ELITE'
    elif c35 < 50: mode = 'UNIVERSAL'
    else: mode = 'TRANSITION'
    results.append((w35, w33, r, c35, c33, mode, elapsed, label))
    print(f"  {w35:>2d}:{w33:<3d} {r:>6.2%} {c35:>5d}人 {c33:>5d}人 {c33-c35:>7d}人 {mode:>14s} {elapsed:>5.0f}s")

# ── 汇总 ──
print(f"\n{'='*80}")
print("汇总 (按 ratio 排序)")
print(f"{'='*80}")
results.sort(key=lambda x: x[2])
prev_mode = None
for w35, w33, r, c35, c33, mode, elapsed, label in results:
    marker = " ◄── 相变" if prev_mode and prev_mode != mode else ""
    print(f"  {w35:>2d}:{w33:<3d} {r:>6.2%}  cnt_35={c35:>3d}  cnt_33={c33:>3d}  [{c33-c35:>3d}]  {mode:<14s}{marker}")
    prev_mode = mode

# 临界区间
elite_rs = [r[2] for r in results if r[5]=='ELITE']
trans_rs = [r[2] for r in results if r[5]=='TRANSITION']
univ_rs = [r[2] for r in results if r[5]=='UNIVERSAL']

print(f"\n  精英  → 比率范围: {min(elite_rs):.2%} – {max(elite_rs):.2%}  (cnt_35 ≈ 175–198)")
if trans_rs:
    print(f"  过渡  → 比率范围: {min(trans_rs):.2%} – {max(trans_rs):.2%}  (cnt_35 ≈ {min(r[3] for r in results if r[5]=='TRANSITION')}–{max(r[3] for r in results if r[5]=='TRANSITION')})")
if univ_rs:
    print(f"  普惠  → 比率范围: {min(univ_rs):.2%} – {max(univ_rs):.2%}  (cnt_35 ≈ {min(r[3] for r in results if r[5]=='UNIVERSAL')}–{max(r[3] for r in results if r[5]=='UNIVERSAL')})")
print(f"\n  精英↔过渡边界: ~{min(elite_rs):.1%}")
if trans_rs and univ_rs:
    print(f"  过渡↔普惠边界: ~{max(univ_rs):.1%} – {min(trans_rs):.1%}")
