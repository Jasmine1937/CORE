"""Evaluate CORE-S menus on later development cohorts and CORE-G formulas on final cohorts."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from prepare import evaluate_expression,evaluate_metrics,youden_threshold

# Cache scores and thresholds for fixed formulas within this process.
_cache_state=None
_cache_data=None
_score_cache={}
_threshold_cache={}


def _cohorts(state,config):
    windows={int(w['index']):w for w in state['windows']}
    development=sorted({int(y) for w in windows.values() for y in w['validation_years']})
    final=sorted(set(map(int,config['windows']['final_years'])))
    if not final:raise ValueError('At least one final year is required')
    if set(development)&set(final):raise ValueError('Final years overlap development validation years')
    return windows,development,final


def _scores(state,window,expression,years):
    from clean import transform
    global _cache_state,_cache_data
    if _cache_state is not state or _cache_data is not state['data']:
        _score_cache.clear();_threshold_cache.clear()
        _cache_state=state;_cache_data=state['data']
    key=(int(window['index']),id(window['preprocessor']),expression,tuple(years))
    if key not in _score_cache:
        frame=state['data'].loc[state['data'].year.isin(years)]
        if set(map(int,years))-set(frame.year):raise ValueError(f'Missing evaluation cohort in {years}')
        x=transform(frame,window['preprocessor'])
        _score_cache[key]=(frame.target.to_numpy(),evaluate_expression(expression,x))
    return key,_score_cache[key]


def evaluate_selection(state,selected,config,method=None):
    """Compute annual metrics for a fixed set of formulas.

    Each formula retains its source-window preprocessing. Calibrate its threshold
    on source validation and subsequent development validation years, then apply
    the frozen threshold to each final-year cohort.
    """
    windows,development,final=_cohorts(state,config)
    records=[]
    properties=('selection_rank','C_global','stability','structural_stability',
                'temporal_stability','temporal_sd','temporal_drop','T','D','O','U','Nx','G','H')
    for position,row in selected.reset_index(drop=True).iterrows():
        wid=int(row.source_window);window=windows[wid];expression=str(row.expression)
        calibration=sorted(set(map(int,window['validation_years'])) |
                           {y for y in development if y>int(window['validation_year'])})
        key,(y_cal,s_cal)=_scores(state,window,expression,calibration)
        if key not in _threshold_cache:_threshold_cache[key]=youden_threshold(y_cal,s_cal)
        threshold=_threshold_cache[key]
        label=method if method is not None else row.get('method','full')
        variant=row.get('variant','G0')
        predictor_id=f'{variant}:{label}:w{wid}:p{position+1}'
        for year in final:
            _,(y,scores)=_scores(state,window,expression,[year])
            metrics=evaluate_metrics(y,scores,threshold)
            record={'variant':variant,'method':label,'predictor_id':predictor_id,
                    'candidate_id':row.get('candidate_id',f'w{wid}_p{position+1}'),
                    'source_window':wid,'expression':expression,'year':year,
                    'calibration_years':','.join(map(str,calibration)),
                    **{k:metrics[k] for k in ('threshold','auc','macro_f1','balanced_accuracy','n','n_positive')}}
            for name in properties:
                if name in row:record[name]=row[name]
            records.append(record)
    columns=['variant','method','predictor_id','candidate_id','source_window','expression',
             'year','calibration_years','threshold','auc','macro_f1','balanced_accuracy','n','n_positive']
    return pd.DataFrame(records) if records else pd.DataFrame(columns=columns)


def evaluate_forward(state,menu,config):
    """Evaluate later development years using the source-validation threshold."""
    windows,development,_=_cohorts(state,config)
    records=[]
    for position,row in menu.reset_index(drop=True).iterrows():
        wid=int(row.source_window);window=windows[wid];threshold=float(row.validation_threshold)
        prepared={int(record['year']):record for record in row.get('forward_metrics',[])}
        for year in development:
            if year<=int(window['validation_year']):continue
            if year in prepared:metrics=prepared[year]
            else:
                _,(y,scores)=_scores(state,window,str(row.expression),[year])
                metrics=evaluate_metrics(y,scores,threshold)
            records.append({'variant':row.get('variant','G0'),'method':'core_s',
                            'candidate_id':row.get('candidate_id',f'w{wid}_p{position+1}'),
                            'source_window':wid,'expression':row.expression,'year':year,
                            'threshold':threshold,
                            **{k:metrics[k] for k in ('auc','macro_f1','balanced_accuracy','n','n_positive')}})
    columns=['variant','method','candidate_id','source_window','expression','year',
             'threshold','auc','macro_f1','balanced_accuracy','n','n_positive']
    return pd.DataFrame(records,columns=columns)


def main(argv=None):
    from clean import cli_config,load_state,save_state
    config=cli_config('Evaluate CORE-S forward and CORE-G final predictions',argv)
    state=load_state(config)
    forward=evaluate_forward(state,state['menus'],config)
    final=evaluate_selection(state,state['selected'],config)
    keys=['variant','source_window','candidate_id','expression']
    fields=keys+[key for key in ('selection_rank','C_global','stability','T','D','O','U','Nx','G','H')
                 if key in state['selected']]
    formulas=state['selected'][fields].copy()
    calibration=final[keys+['threshold','calibration_years']].drop_duplicates(keys)
    formulas=formulas.merge(calibration,on=keys,how='left',validate='one_to_one')
    names=state['features']
    formulas['expression_features']=formulas.expression.map(
        lambda text:re.sub(r'\bX(\d+)\b',lambda m:'['+names[int(m.group(1))]+']',text))
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    for name,frame in [('forward',forward),('final',final),('formulas',formulas)]:
        frame.to_csv(output/(name+'.csv'),index=False)
        state[name]=frame
    save_state(config,state)
    print(f'Saved {len(forward)} forward rows and {len(final)} final rows.')


if __name__=='__main__':main()
