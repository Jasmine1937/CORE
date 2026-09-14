"""CORE-S screening, CORE-G selection and candidate-archive selectors.

Each call processes one GP variant and replicate. Candidates retain their
source-window identities through screening; final selection deduplicates
canonical expressions. Calibration and final evaluation are implemented in
test.py.
"""
from __future__ import annotations
import json
from collections import Counter
import numpy as np
import pandas as pd
from prepare import parse_expression,syntax_complexity,canonical,ExpressionResourceError,auc_standard_error

DEFAULTS={
    'tree_size_max':100,'depth_max':12,'fallback_count':20,
    'z_one_se':1.,'one_se_fallback_band':.005,
    'guard_quantile':.8,'guard_rate':.9,'use_safety_guards':True,
    'tier_low_quantile':.25,'tier_high_quantile':.75,
    'k_low':3,'k_mid':3,'k_high':4,'delta_mid':.01,'delta_high':.005,
    'stability_lambda':1.,'shrinkage_nu':2.,'std_weight':.7,'worst_drop_weight':.3,
    'structural_weight':.25,'temporal_weight':.75,
    'selection_budget':3,'minimum_new_coverage':.01,'random_seed':42,
    'complexity_weights':{'T':1.,'D':.5,'O':.5,'U':.8,'Nx':1.5,'G':.5,'H':.5},
    'greedy_weights':{'auc':1.,'stability':1.,'complexity':.5,'coverage':.5}}
ALIASES={'Full-CORE':'full','AUC-best':'auc_best','1SE-simplest':'one_se_simplest',
         'Pareto-knee':'pareto_knee','1SE-Random':'one_se_random'}
METHODS=('full','auc_best','one_se_simplest','pareto_knee','one_se_random')

def settings(config=None):
    values=dict(config or {})
    if 'core' in values:
        values={**values['core'],
                'random_seed':values.get('seed',42),
                'final_test_years':values.get('windows',{}).get('final_years',[])}
    out={**DEFAULTS,**values}
    out['complexity_weights']={**DEFAULTS['complexity_weights'],**values.get('complexity_weights',{})}
    out['greedy_weights']={**DEFAULTS['greedy_weights'],**values.get('greedy_weights',{})}
    if int(out['selection_budget'])<1 or out['shrinkage_nu']<=0 or out['stability_lambda']<1:
        raise ValueError('Invalid final budget, shrinkage strength or recurrence exponent')
    if out['z_one_se']<=0 or not 0<=out['guard_quantile']<=1 or not 0<out['guard_rate']<1:
        raise ValueError('Invalid one-SE or guard parameters')
    if not 0<=out['tier_low_quantile']<=out['tier_high_quantile']<=1:
        raise ValueError('Invalid tier quantiles')
    return out

def minmax(values):
    a=pd.Series(values,copy=True,dtype=float)
    if len(a)==0: return a
    if not np.isfinite(a).all(): raise ValueError('Normalization requires finite values')
    span=float(a.max()-a.min())
    return (a-a.min())/span if span>0 else pd.Series(0.,index=a.index)

minmax01=minmax

def _prepare(candidates,feature_count=None):
    frame=candidates.copy().reset_index(drop=True)
    renames={'expression_exact':'expression','window':'source_window','val_auc':'validation_auc',
             'val_macro_f1':'validation_mf1','val_balanced_accuracy':'validation_bacc'}
    for old,new in renames.items():
        if new not in frame and old in frame: frame[new]=frame[old]
    required=['expression','source_window','validation_auc','validation_mf1','validation_bacc','n_pos','n_neg']
    missing=[x for x in required if x not in frame]
    if missing: raise ValueError(f'Missing candidate fields: {missing}')
    for key in ('variant','replicate','base_seed'):
        if key in frame and frame[key].nunique()>1:
            raise ValueError(f'One CORE invocation must contain a single {key}')
    for col in required[2:]:
        frame[col]=pd.to_numeric(frame[col],errors='raise')
        if not np.isfinite(frame[col]).all(): raise ValueError(f'Nonfinite candidate field: {col}')
    for col in ('validation_auc','validation_mf1','validation_bacc'):
        if not frame[col].between(0,1).all(): raise ValueError(f'Invalid metric: {col}')
    if frame.empty: return frame
    programs=[parse_expression(expr,feature_count) for expr in frame.expression]
    for name in ('T','D','O','U','Nx','G','H'): frame[name]=[getattr(x,name) for x in programs]
    frame['canonical_expression']=[x.canonical for x in programs]
    frame['feature_indices']=[sorted(x.variables) for x in programs]
    if 'candidate_id' not in frame:
        frame['candidate_id']=[f'w{w}_c{i+1}' for i,w in enumerate(frame.source_window)]
    if frame.candidate_id.duplicated().any(): raise ValueError('Candidate IDs are not unique')
    return frame.reset_index(drop=True)

def _complexity(frame,cfg,name):
    out=frame.copy()
    total=pd.Series(0.,index=out.index)
    for key in ('T','D','O','U','Nx','G','H'):
        raw=np.log1p(out[key]) if key=='T' else out[key]
        normalized=minmax(raw)
        out[f'{name}_z_{key}']=normalized
        total+=float(cfg['complexity_weights'][key])*normalized
    out[name]=total
    return out

def _one_se(pool,cfg):
    if pool.empty: return pool.copy(),float('nan'),float('nan')
    if pool.n_pos.nunique()>1 or pool.n_neg.nunique()>1:
        raise ValueError('Validation class counts differ within the same source window')
    best=float(pool.validation_auc.max())
    se=auc_standard_error(best,int(pool.iloc[0].n_pos),int(pool.iloc[0].n_neg))
    width=float(cfg['z_one_se'])*se if np.isfinite(se) else float(cfg['one_se_fallback_band'])
    return pool[pool.validation_auc>=best-width].copy(),best-width,se

def core_s(candidates,config=None):
    """Screen source windows by validation performance and complexity.

    Return (menu, per-window summary). Selection uses source-window validation
    metrics. Forward metrics are carried into the menu for subsequent
    cross-window integration.
    """
    cfg=settings(config); frame=candidates.copy()
    for old,new in [('expression_exact','expression'),('window','source_window')]:
        if new not in frame and old in frame:frame[new]=frame[old]
    if frame.empty: return frame,pd.DataFrame()
    if not {'expression','source_window'}.issubset(frame):raise ValueError('Missing expression/source_window')
    for key in ('variant','replicate','base_seed'):
        if key in frame and frame[key].nunique()>1:raise ValueError(f'One CORE invocation requires one {key}')
    dimensions=[syntax_complexity(expr,cfg.get('feature_count')) for expr in frame.expression]
    for key in ('T','D','O','Nx'):frame[key]=[d[key] for d in dimensions]
    parts=[]; summaries=[]
    windows=sorted(frame.source_window.unique().tolist())
    for window,group in frame.groupby('source_window',sort=True):
        hard=group[group['T']<=cfg['tree_size_max']]
        hard=hard[hard.D<=cfg['depth_max']].copy()
        fallback=hard.empty
        if fallback:
            hard=group.sort_values(['T','D','O','expression'],kind='stable').head(int(cfg['fallback_count'])).copy()
        hard=_complexity(_prepare(hard,cfg.get('feature_count')),cfg,'C_window')
        band,floor,se=_one_se(hard,cfg)
        f1floor=float(cfg['guard_rate'])*float(hard.validation_mf1.quantile(cfg['guard_quantile']))
        bafloor=float(cfg['guard_rate'])*float(hard.validation_bacc.quantile(cfg['guard_quantile']))
        guarded=band[(band.validation_mf1>=f1floor)&(band.validation_bacc>=bafloor)].copy() if cfg['use_safety_guards'] else band.copy()
        guard_fallback=guarded.empty and not band.empty
        if guard_fallback: guarded=band.copy()
        ql,qh=guarded.C_window.quantile([cfg['tier_low_quantile'],cfg['tier_high_quantile']]).tolist()
        guarded['tier']=np.select([guarded.C_window<=ql,guarded.C_window<=qh],['Low','Mid'],default='High')
        best=float(guarded.validation_auc.max())
        selected=[]
        for tier,k,delta in [('Low',cfg['k_low'],None),('Mid',cfg['k_mid'],cfg['delta_mid']),('High',cfg['k_high'],cfg['delta_high'])]:
            sub=guarded[guarded.tier==tier]
            if delta is not None: sub=sub[sub.validation_auc>=best-float(delta)]
            selected.append(sub.sort_values(['validation_auc','C_window','canonical_expression'],ascending=[False,True,True],kind='stable').head(int(k)))
        menu=pd.concat(selected,ignore_index=True)
        menu['core_s_one_se_floor']=floor
        parts.append(menu)
        summaries.append({'source_window':window,'archive_candidates':len(group),'hard_candidates':len(hard),
                       'hard_fallback':fallback,'one_se_candidates':len(band),'one_se_floor':floor,'auc_se':se,
                       'guard_mf1_floor':f1floor,'guard_bacc_floor':bafloor,'guard_fallback':guard_fallback,
                       'guarded_candidates':len(guarded),'tier_low_cutoff':ql,'tier_high_cutoff':qh,'menu_candidates':len(menu)})
    result=pd.concat(parts,ignore_index=True)
    result.attrs['source_windows']=cfg.get('source_windows',candidates.attrs.get('source_windows',windows))
    result.attrs['final_test_years']=cfg.get('final_test_years',candidates.attrs.get('final_test_years',[]))
    return result,pd.DataFrame(summaries)

def _forward_values(value):
    if value is None: return np.array([],dtype=float)
    if isinstance(value,str): value=json.loads(value)
    result=np.asarray(value,dtype=float)
    if result.ndim!=1 or not np.isfinite(result).all() or not ((result>=0)&(result<=1)).all():
        raise ValueError('forward_auc requires a one-dimensional vector of finite AUC values in [0, 1]')
    return result

def global_properties(menu,config=None):
    """Compute static stability and complexity with menu-level normalization."""
    cfg=settings(config); windows=cfg.get('source_windows',menu.attrs.get('source_windows'))
    work=_prepare(menu,cfg.get('feature_count'))
    if work.empty: return work
    if 'forward_auc' not in work: raise ValueError('CORE-G requires explicit forward_auc lists (empty when none exist)')
    windows=list(windows) if windows is not None else sorted(work.source_window.unique())
    if not set(work.source_window).issubset(set(windows)): raise ValueError('Unknown source window')
    freq=Counter()
    for window in windows:
        sub=work[work.source_window==window]
        local=Counter(feature for features in sub.feature_indices for feature in features)
        if len(sub):
            for feature,count in local.items(): freq[feature]+=count/len(sub)/len(windows)
    work['feature_frequency']=[dict(freq) for _ in range(len(work))]
    work['structural_stability']=[float(np.mean([freq[j]**cfg['stability_lambda'] for j in features])) if features else 0. for features in work.feature_indices]
    arrays=[_forward_values(x) for x in work.forward_auc]
    if 'forward_years' in work:
        for row,values in zip(work.itertuples(),arrays):
            years=json.loads(row.forward_years) if isinstance(row.forward_years,str) else row.forward_years
            if len(years)!=len(values) or len(set(years))!=len(years): raise ValueError('Invalid forward cohort mapping')
            if hasattr(row,'source_year') and any(int(y)<=int(row.source_year) for y in years): raise ValueError('Forward cohorts must follow source year')
            if set(map(int,years))&set(map(int,cfg.get('final_test_years',menu.attrs.get('final_test_years',[])))): raise ValueError('Temporal-stability cohorts overlap final test years')
    n=np.array([len(a) for a in arrays]); eligible=n>=2
    raw_var=np.array([float(np.var(a,ddof=1)) if len(a)>=2 else np.nan for a in arrays])
    raw_drop=np.array([float(np.mean(a)-np.min(a)) if len(a)>=2 else np.nan for a in arrays])
    v0=float(np.median(raw_var[eligible])) if eligible.any() else 0.
    d0=float(np.median(raw_drop[eligible])) if eligible.any() else 0.
    degrees=np.maximum(n-1,0); nu=float(cfg['shrinkage_nu'])
    variance=np.where(eligible,v0+degrees/(degrees+nu)*(np.nan_to_num(raw_var)-v0),v0)
    drop=np.where(eligible,d0+degrees/(degrees+nu)*(np.nan_to_num(raw_drop)-d0),d0)
    work['forward_count']=n
    work['forward_mean_auc']=[float(a.mean()) if len(a) else np.nan for a in arrays]
    work['temporal_raw_variance']=raw_var;work['temporal_raw_drop']=raw_drop
    work['temporal_prior_variance']=v0;work['temporal_prior_drop']=d0
    work['temporal_variance']=variance;work['temporal_sd']=np.sqrt(variance);work['temporal_drop']=drop
    work['temporal_sd_z']=minmax(work.temporal_sd);work['temporal_drop_z']=minmax(work.temporal_drop)
    work['temporal_stability']=1.-float(cfg['std_weight'])*work.temporal_sd_z-float(cfg['worst_drop_weight'])*work.temporal_drop_z
    work['stability']=float(cfg['structural_weight'])*work.structural_stability+float(cfg['temporal_weight'])*work.temporal_stability
    work=_complexity(work,cfg,'C_global')
    work['auc_z']=minmax(work.validation_auc);work['stability_z']=minmax(work.stability);work['complexity_z']=minmax(work.C_global)
    work['temporal_variance_ddof']=1
    return work

def _finish(selected,cfg,method):
    result=selected.copy().reset_index(drop=True)
    result['selection_rank']=np.arange(1,len(result)+1)
    result['selection_budget']=int(cfg['selection_budget'])
    result['selection_shortfall']=max(0,int(cfg['selection_budget'])-len(result))
    result['method']=method
    result.attrs['selection_shortfall']=max(0,int(cfg['selection_budget'])-len(result))
    return result

def core_g(menu,config=None,method='full'):
    """Select up to K distinct canonical expressions; return (selection, pool)."""
    method=ALIASES.get(method,method)
    if method!='full': raise ValueError(f'CORE-G method must be full, got {method}')
    cfg=settings(config);work=global_properties(menu,cfg)
    if work.empty: return _finish(work,cfg,method),work
    weights=dict(cfg['greedy_weights'])
    work['base_score']=weights['auc']*work.auc_z+weights['stability']*work.stability_z-weights['complexity']*work.complexity_z
    remaining=list(work.index);covered=set();chosen=[]
    while remaining and len(chosen)<int(cfg['selection_budget']):
        gains={i:sum(work.at[i,'feature_frequency'].get(j,0.) for j in set(work.at[i,'feature_indices'])-covered) for i in remaining}
        maximum=max(gains.values(),default=0.)
        omit=bool(chosen) and maximum<float(cfg['minimum_new_coverage'])
        normalized={i:gains[i]/maximum if maximum>0 and not omit else 0. for i in remaining}
        scores={i:float(work.at[i,'base_score'])+weights['coverage']*normalized[i] for i in remaining}
        best=min(remaining,key=lambda i:(-scores[i],-work.at[i,'validation_auc'],work.at[i,'C_global'],work.at[i,'canonical_expression'],str(work.at[i,'source_window'])))
        row=work.loc[best].copy();row['selection_score']=scores[best];row['new_coverage']=gains[best];row['new_coverage_z']=normalized[best];row['coverage_omitted']=omit
        chosen.append(row);covered.update(work.at[best,'feature_indices'])
        identity=work.at[best,'canonical_expression']
        remaining=[i for i in remaining if work.at[i,'canonical_expression']!=identity]
    return _finish(pd.DataFrame(chosen),cfg,method),work

def _distinct(frame,columns,ascending):
    return frame.sort_values(columns+['canonical_expression','source_window'],ascending=ascending+[True,True],kind='stable').drop_duplicates('canonical_expression',keep='first')

def select_candidates(candidates,config=None,method='full',reference_menu=None):
    """Apply CORE or an alternative selector to the candidate archive.

    1SE selectors choose from source-window AUC bands. Pareto-knee chooses from
    the nondominated set. When the admissible set is smaller than the requested
    budget, the result records the selection shortfall. Alternative selectors
    use fixed complexity bounds from the CORE-S menu of the same archive.
    """
    method=ALIASES.get(method,method);cfg=settings(config)
    if method=='full':
        menu,summary=core_s(candidates,cfg);selected,_=core_g(menu,cfg,method)
        return selected,summary
    if method not in METHODS: raise ValueError(f'Unknown selector: {method}')
    pool=_prepare(candidates,cfg.get('feature_count'))
    if pool.empty: return _finish(pool,cfg,method),pd.DataFrame()
    if reference_menu is None: reference_menu,_=core_s(candidates,cfg)
    pool=characterize_candidates(pool,reference_menu,cfg)
    summary=[];bands=[]
    for window,part in pool.groupby('source_window',sort=True):
        eligible,floor,se=_one_se(part,cfg)
        bands.append(eligible)
        summary.append({'source_window':window,'one_se_floor':floor,'auc_se':se,'eligible_count':len(eligible)})
    band=pd.concat(bands,ignore_index=True)
    if method=='auc_best': chosen=_distinct(pool,['validation_auc','C_global'],[False,True])
    elif method=='one_se_simplest': chosen=_distinct(band,['C_global','validation_auc'],[True,False])
    elif method=='one_se_random':
        unique=_distinct(band,['validation_auc','C_global'],[False,True])
        chosen=unique.iloc[np.random.default_rng(int(cfg['random_seed'])).permutation(len(unique))]
    else:
        unique=_distinct(pool,['validation_auc','C_global'],[False,True])
        ordered=unique.sort_values(['C_global','validation_auc'],ascending=[True,False],kind='stable')
        keep=[];best=-np.inf
        for index,row in ordered.iterrows():
            if row.validation_auc>best+1e-15: keep.append(index);best=row.validation_auc
        front=ordered.loc[keep].copy()
        x=minmax(front.C_global).to_numpy();y=minmax(front.validation_auc).to_numpy()
        if len(front)<3:
            salience=-np.hypot(x,1.-y)
        else:
            dx,dy=x[-1]-x[0],y[-1]-y[0];norm=np.hypot(dx,dy)
            salience=np.abs(dx*(y[0]-y)-(x[0]-x)*dy)/norm if norm else np.zeros(len(front))
        front['knee_salience']=salience
        crowd=np.zeros(len(front))
        if len(front)<=2: crowd[:]=np.inf
        else:
            for a in (x,y):
                order=np.argsort(a,kind='stable');crowd[order[0]]=crowd[order[-1]]=np.inf
                if a[order[-1]]>a[order[0]]:
                    for k in range(1,len(order)-1): crowd[order[k]]+=(a[order[k+1]]-a[order[k-1]])/(a[order[-1]]-a[order[0]])
        front['pareto_crowding']=crowd
        chosen=_distinct(front,['knee_salience','pareto_crowding','validation_auc','C_global'],[False,False,False,True])
    return _finish(chosen.head(int(cfg['selection_budget'])),cfg,method),pd.DataFrame(summary)


def characterize_candidates(candidates,reference_menu,config=None,errors="raise"):
    """Measure complexity using fixed CORE-menu reference bounds.

    Normalization uses the reference menu independently of the query candidates.
    Values may extend beyond the reference range; zero-range components
    contribute zero. Selection and reporting use the same complexity values.
    """
    if errors not in ('raise','record'):raise ValueError("errors must be 'raise' or 'record'")
    cfg=settings(config);out=candidates.copy();reference=reference_menu.copy()
    if reference.empty: raise ValueError('Complexity characterization requires a nonempty reference menu')
    def dimensions(frame,record=False):
        expr_col='expression' if 'expression' in frame else 'expression_exact'
        records=[]
        for expr in frame[expr_col]:
            try:
                p=parse_expression(expr,cfg.get('feature_count'))
                records.append({**p.complexity(),'canonical_expression':p.canonical,
                                'feature_indices':sorted(p.variables),'syntactic_U':len(p.syntactic_variables),
                                'complexity_status':'exact','complexity_error':None})
            except ExpressionResourceError as exc:
                if not record:raise
                # Effective U counts variables after algebraic cancellation.
                from prepare import _postfix_tokens
                syntactic=sorted({int(t[1:]) for t in _postfix_tokens(expr,cfg.get('feature_count')) if t.startswith('X')})
                records.append({**syntax_complexity(expr,cfg.get('feature_count')),
                                'U':np.nan,'G':np.nan,'H':np.nan,'canonical_expression':canonical(expr),
                                'feature_indices':None,'syntactic_U':len(syntactic),
                                'complexity_status':'unavailable_exact_expansion','complexity_error':str(exc)})
        for key in ('T','D','O','U','Nx','G','H','canonical_expression','feature_indices','syntactic_U','complexity_status','complexity_error'):
            frame[key]=[r[key] for r in records]
        return frame
    reference=dimensions(reference);out=dimensions(out,errors=='record')
    score=np.zeros(len(out));bounds={}
    for key in ('T','D','O','U','Nx','G','H'):
        ref=np.log1p(reference[key]) if key=='T' else reference[key]
        query=np.log1p(out[key]) if key=='T' else out[key]
        lo,hi=float(ref.min()),float(ref.max());bounds[key]=[lo,hi]
        normalized=(query-lo)/(hi-lo) if hi>lo else np.zeros(len(out))
        out[f'C_reference_z_{key}']=normalized
        out[f'C_global_z_{key}']=normalized
        score+=float(cfg['complexity_weights'][key])*normalized
    score=np.where(out.complexity_status.eq('exact'),score,np.nan)
    if 'C_global' in out or 'selector_C_global' in out:out['selector_C_global']=score
    out['C_global']=score
    out['complexity_reference']='full_core_menu_frozen_bounds'
    out.attrs['complexity_reference_bounds']=bounds
    out.attrs['complexity_reference_size']=len(reference)
    return out



def main(argv=None):
    from clean import cli_config,load_state,save_state
    config=cli_config('Select CORE-S menus and CORE-G representatives',argv)
    state=load_state(config)
    cfg={**config['core'],'feature_count':len(state['features']),
         'source_windows':[w['index'] for w in state['windows']],
         'final_test_years':config['windows']['final_years'],'random_seed':config['seed']}
    menus=[];selections=[];pools=[]
    for variant,group in state['prepared'].groupby('variant',sort=True):
        menu,_=core_s(group,cfg)
        selected,pool=core_g(menu,cfg)
        menus.append(menu);selections.append(selected);pools.append(pool)
    if not menus:raise ValueError('No prepared candidates; run prepare.py first')
    state['menus']=pd.concat(menus,ignore_index=True)
    state['selected']=pd.concat(selections,ignore_index=True)
    state['pools']=pd.concat(pools,ignore_index=True)
    for key in ('forward','final','formulas'):state.pop(key,None)
    save_state(config,state)
    print(f"Selected {len(state['selected'])} formulas from {len(state['menus'])} CORE-S candidates.")

if __name__=='__main__':main()
