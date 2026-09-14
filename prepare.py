"""Expression arithmetic, validation metrics and forward preparation."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from decimal import Decimal
import numpy as np
import pandas as pd
from functools import lru_cache

class ExpressionError(ValueError):
    """Expression parsing or symbolic-computation error."""

class ExpressionResourceError(ExpressionError):
    """A symbolic computation exceeds its configured resource budget."""

@dataclass(frozen=True)
class ExpressionProgram:
    expression: str
    canonical: str
    variables: frozenset[int]
    T: int
    D: int
    O: int
    U: int
    Nx: int
    G: int
    H: int
    syntactic_variables: frozenset[int] = field(repr=False)
    postfix: tuple[str,...] = field(repr=False,default=())

    @property
    def canonical_expression(self):
        return self.canonical

    @property
    def degree(self):
        return self.G

    @property
    def interaction(self):
        return self.H

    def complexity(self):
        return {name: getattr(self, name) for name in ('T','D','O','U','Nx','G','H')}

    def evaluate(self, X):
        return _evaluate_postfix(self.postfix, X)


def _evaluate_postfix(tokens, X):
    values=np.asarray(X,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('X must be a finite two-dimensional feature matrix')
    stack=[]
    with np.errstate(over='ignore',invalid='ignore'):
        for token in tokens:
            if token in ('u+','u-'):
                value=stack.pop();stack.append(-value if token=='u-' else value)
            elif token in ('+','-','*'):
                right,left=stack.pop(),stack.pop()
                stack.append(left+right if token=='+' else left-right if token=='-' else left*right)
            elif token.startswith('X'):
                index=int(token[1:])
                if index>=values.shape[1]:raise ValueError('Expression terminal exceeds the feature mapping')
                stack.append(values[:,index])
            else:stack.append(float(token))
        out=np.broadcast_to(np.asarray(stack[0],dtype=float),(len(values),)).copy()
    return np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.)


def evaluate_expression(expression,X):
    """Evaluate expression arithmetic in its original operation order."""
    values=np.asarray(X,dtype=float)
    if values.ndim!=2:raise ValueError('X must be a finite two-dimensional feature matrix')
    return _evaluate_postfix(_postfix_tokens(expression,values.shape[1]),values)


def _arithmetic_dag(tokens):
    """Build a shared expression graph with exact algebraic simplifications."""
    nodes=[];intern={};stack=[]
    def node(key):
        if key not in intern:intern[key]=len(nodes);nodes.append(key)
        return intern[key]
    zero=node(('constant',0));one=node(('constant',1))
    for token in tokens:
        if token in ('+','-','*'):
            right,left=stack.pop(),stack.pop()
            if token=='-' and left==right:result=zero
            elif token=='*' and (left==zero or right==zero):result=zero
            elif token=='*' and left==one:result=right
            elif token=='*' and right==one:result=left
            elif token in ('+','-') and right==zero:result=left
            elif token=='+' and left==zero:result=right
            else:
                if token in ('+','*') and left>right:left,right=right,left
                result=node((token,left,right))
            stack.append(result)
        elif token in ('u+','u-'):
            value=stack.pop()
            stack.append(value if token=='u+' else node(('*',node(('constant',-1)),value)))
        elif token.startswith('X'):stack.append(node(('variable',int(token[1:]))))
        else:
            value=Fraction(token)
            stack.append(node(('constant',int(value) if value.denominator==1 else value)))
    root=stack[0];needed=set();pending=[root]
    while pending:
        index=pending.pop()
        if index in needed:continue
        needed.add(index)
        if nodes[index][0] in ('+','-','*'):pending.extend(nodes[index][1:])
    return nodes,sorted(needed),root


def _certified_dimensions(nodes,order,root,point_budget=4096,attempt=0):
    """Determine exact U, G and H by matching lower and upper bounds.

    Exact integer substitutions provide nonzero witnesses. Zero evaluations
    leave the corresponding lower bound unconfirmed. Return None when these
    witnesses do not determine every dimension.
    """
    from itertools import combinations
    variables=sorted({nodes[i][1] for i in order if nodes[i][0]=='variable'})
    if any(isinstance(nodes[i][1],Fraction) for i in order if nodes[i][0]=='constant'):
        return None
    primes=[];number=2
    while len(primes)<len(variables):
        if all(number%p for p in primes if p*p<=number):primes.append(number)
        number+=1
    point=dict(zip(variables,[p**(attempt+1) for p in primes]))
    # Projection Xi = a_i*t provides a lower bound on total degree.
    projected={};upper={};projection_operations=0
    for i in order:
        node=nodes[i];op=node[0]
        if op=='constant':poly={0:node[1]} if node[1] else {};bound=0 if poly else -1
        elif op=='variable':poly={1:point[node[1]]};bound=1
        else:
            left,right=projected[node[1]],projected[node[2]]
            projection_operations+=len(left)*len(right) if op=='*' else len(right)
            if projection_operations>2_000_000:return None
            if op=='*':
                bound=upper[node[1]]+upper[node[2]] if min(upper[node[1]],upper[node[2]])>=0 else -1
                poly={}
                for a,c in left.items():
                    for b,d in right.items():
                        value=poly.get(a+b,0)+c*d
                        if value:poly[a+b]=value
                        else:poly.pop(a+b,None)
            else:
                bound=max(upper[node[1]],upper[node[2]]);poly=dict(left)
                sign=1 if op=='+' else -1
                for a,c in right.items():
                    value=poly.get(a,0)+sign*c
                    if value:poly[a]=value
                    else:poly.pop(a,None)
        if any(abs(c).bit_length()>131072 for c in poly.values()):return None
        projected[i]=poly;upper[i]=bound
    degree=max(projected[root],default=-1)
    if degree!=upper[root]:return None
    if not variables:return frozenset(),max(degree,0),0
    del projected
    # A nonzero exact partial derivative proves that its variable is effective.
    values={};gradients={};positions={v:j for j,v in enumerate(variables)}
    for i in order:
        node=nodes[i];op=node[0]
        if op=='constant':value=node[1];gradient=[0]*len(variables)
        elif op=='variable':
            value=point[node[1]];gradient=[0]*len(variables);gradient[positions[node[1]]]=1
        else:
            a,b=values[node[1]],values[node[2]]
            ga,gb=gradients[node[1]],gradients[node[2]]
            if op=='*':value=a*b;gradient=[x*b+a*y for x,y in zip(ga,gb)]
            else:
                sign=1 if op=='+' else -1
                value=a+sign*b;gradient=[x+sign*y for x,y in zip(ga,gb)]
        if abs(value).bit_length()>131072 or any(abs(x).bit_length()>131072 for x in gradient):return None
        values[i]=value;gradients[i]=gradient
    if not all(gradients[root]):return None
    interaction=min(len(variables),degree)
    if interaction<=1:return frozenset(variables),degree,interaction
    # Inclusion-exclusion removes all terms missing a variable from the subset.
    # A nonzero result certifies a monomial containing the entire subset.
    if interaction>12 or (1<<interaction)>point_budget:return None
    used=0
    for subset in combinations(variables,interaction):
        cost=1<<interaction
        if used+cost>point_budget:return None
        used+=cost;difference=0
        for mask in range(cost):
            at={v:point[v] for j,v in enumerate(subset) if mask&(1<<j)}
            evaluated={}
            for i in order:
                node=nodes[i];op=node[0]
                if op=='constant':value=node[1]
                elif op=='variable':value=at.get(node[1],0)
                else:
                    a,b=evaluated[node[1]],evaluated[node[2]]
                    value=a*b if op=='*' else a+b if op=='+' else a-b
                if abs(value).bit_length()>131072:return None
                evaluated[i]=value
            difference+=(-1 if (interaction-mask.bit_count())%2 else 1)*evaluated[root]
        if difference:return frozenset(variables),degree,interaction
    return None


def _expanded_dimensions(nodes,order,root,token_count,max_monomials,max_operations):
    """Complete exact expansion with packed exponents and sparse coefficients."""
    variables=sorted({nodes[i][1] for i in order if nodes[i][0]=='variable'})
    bits=max(1,token_count.bit_length());mask=(1<<bits)-1
    shifts={v:j*bits for j,v in enumerate(variables)}
    polys={};uses={i:0 for i in order};operations=0
    for i in order:
        if nodes[i][0] in ('+','-','*'):
            for child in nodes[i][1:]:uses[child]+=1
    for i in order:
        node=nodes[i];op=node[0]
        if op=='constant':poly={0:node[1]} if node[1] else {}
        elif op=='variable':poly={1<<shifts[node[1]]:1}
        else:
            left,right=polys[node[1]],polys[node[2]]
            operations+=len(left)*len(right) if op=='*' else len(right)
            if operations>max_operations:raise ExpressionResourceError('Exact polynomial expansion exceeds operation budget')
            if op=='*':
                poly={}
                for a,c in left.items():
                    for b,d in right.items():
                        value=poly.get(a+b,0)+c*d
                        if value:poly[a+b]=value
                        else:poly.pop(a+b,None)
                        if len(poly)>max_monomials:raise ExpressionResourceError('Expanded polynomial exceeds monomial budget')
            else:
                poly=dict(left);sign=1 if op=='+' else -1
                for a,c in right.items():
                    value=poly.get(a,0)+sign*c
                    if value:poly[a]=value
                    else:poly.pop(a,None)
                if len(poly)>max_monomials:raise ExpressionResourceError('Expanded polynomial exceeds monomial budget')
            for child in node[1:]:
                uses[child]-=1
                if uses[child]==0:del polys[child]
        polys[i]=poly
    effective=set();degree=interaction=0
    for monomial in polys[root]:
        powers=[(monomial>>shifts[v])&mask for v in variables]
        effective.update(v for v,p in zip(variables,powers) if p)
        degree=max(degree,sum(powers));interaction=max(interaction,sum(p>0 for p in powers))
    return frozenset(effective),degree,interaction


@lru_cache(maxsize=20000)
def parse_expression(expression,n_features=None,max_monomials=200_000,max_polynomial_operations=2_000_000,**kwargs):
    """Compute structural complexity with exact algebraic cancellation.

    Large expressions use certified bounds where they determine U, G and H;
    remaining cases use rational polynomial expansion. Cached programs retain
    dimensions, canonical identity and the original evaluation instructions.
    """
    if 'feature_count' in kwargs:n_features=kwargs.pop('feature_count')
    if kwargs:raise TypeError(f'Unknown parser options: {sorted(kwargs)}')
    tokens=_postfix_tokens(expression,n_features)
    nodes,order,root=_arithmetic_dag(tokens)
    dimensions=None
    if len(tokens)>100:
        for attempt in range(4):
            dimensions=_certified_dimensions(nodes,order,root,point_budget=1024,attempt=attempt)
            if dimensions is not None:break
    if dimensions is None:
        dimensions=_expanded_dimensions(nodes,order,root,len(tokens),max_monomials,max_polynomial_operations)
    effective,degree,interaction=dimensions
    syntax=syntax_complexity(expression,n_features)
    syntactic=frozenset(int(t[1:]) for t in tokens if t.startswith('X'))
    return ExpressionProgram(expression,canonical(expression),effective,syntax['T'],syntax['D'],syntax['O'],
                             len(effective),syntax['Nx'],degree,interaction,syntactic,tuple(tokens))

def evaluate(expression,X):
    return evaluate_expression(expression,X)

def _canonical_number(token):
    sign,digits,exponent=Decimal(token).as_tuple()
    text=''.join(map(str,digits)).lstrip('0')
    if not text:return '0'
    zeros=len(text)-len(text.rstrip('0'))
    text=text.rstrip('0');exponent+=zeros
    return ('-' if sign else '')+text+(f'e{exponent}' if exponent else '')


@lru_cache(maxsize=4096)
def canonical(expression):
    stack=[]
    for token in _postfix_tokens(expression):
        if token in ('u+','u-'):stack.append('('+token+stack.pop()+')')
        elif token in ('+','-','*'):
            right,left=stack.pop(),stack.pop()
            if token in ('+','*') and left>right:left,right=right,left
            stack.append('('+left+token+right+')')
        else:stack.append('X'+str(int(token[1:])) if token.startswith('X') else _canonical_number(token))
    return stack[0]

def complexity(expression):
    return parse_expression(expression).complexity()


@lru_cache(maxsize=4096)
def _postfix_tokens(expression,n_features=None):
    """Convert arithmetic to postfix tokens using an iterative operator stack.

    The grammar supports variables, decimal constants, binary +, - and *,
    unary signs and nested parentheses.
    """
    if not isinstance(expression,str) or not expression.strip():raise ExpressionError('Empty expression')
    if len(expression)>1_000_000:raise ExpressionError('Expression exceeds input size limit')
    token_re=re.compile(r'\s*(X\d+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[()+*\-])')
    tokens=[];position=0
    while position<len(expression):
        match=token_re.match(expression,position)
        if match is None:
            if not expression[position:].strip():break
            raise ExpressionError('Invalid arithmetic token')
        tokens.append(match.group(1));position=match.end()
    output=[];operators=[];expected=True
    precedence={'+':1,'-':1,'*':2,'u+':3,'u-':3}
    for token in tokens:
        if token=='(':
            if not expected:raise ExpressionError('Missing binary operator')
            operators.append(token)
        elif token==')':
            if expected:raise ExpressionError('Missing operand')
            while operators and operators[-1]!='(':output.append(operators.pop())
            if not operators:raise ExpressionError('Unmatched parenthesis')
            operators.pop()
        elif token in ('+','-','*'):
            if expected:
                if token=='*':raise ExpressionError('Missing operand')
                operators.append('u'+token)
            else:
                while operators and operators[-1]!='(' and precedence[operators[-1]]>=precedence[token]:output.append(operators.pop())
                operators.append(token);expected=True
        else:
            if not expected:raise ExpressionError('Missing binary operator')
            if token.startswith('X'):
                if n_features is not None and int(token[1:])>=n_features:raise ExpressionError('Terminal outside mapping')
            elif not math.isfinite(float(token)):raise ExpressionError('Nonfinite constant')
            output.append(token);expected=False
    if expected:raise ExpressionError('Missing final operand')
    while operators:
        if operators[-1]=='(':raise ExpressionError('Unmatched parenthesis')
        output.append(operators.pop())
    return tuple(output)

@lru_cache(maxsize=4096)
def _syntax_counts(expression,n_features=None):
    """Count T, D, O and Nx from postfix expression syntax."""
    output=_postfix_tokens(expression,n_features)
    stack=[]
    for token in output:
        if token in ('u+','u-'):
            if not stack:raise ExpressionError('Missing unary operand')
            t,d,o,nx=stack.pop();stack.append((t+1,d+1,o,nx))
        elif token in ('+','-','*'):
            if len(stack)<2:raise ExpressionError('Missing binary operand')
            rt,rd,ro,rnx=stack.pop();lt,ld,lo,lnx=stack.pop()
            stack.append((lt+rt+1,max(ld,rd)+1,lo+ro+1,lnx+rnx+int(token=='*')))
        else:stack.append((1,1,0,0))
    if len(stack)!=1:raise ExpressionError('Malformed expression')
    return tuple(stack[0])


def syntax_complexity(expression,n_features=None):
    return dict(zip(('T','D','O','Nx'),_syntax_counts(expression,n_features)))


from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score

def _arrays(y,scores):
    labels,values=np.asarray(y),np.asarray(scores,dtype=float)
    if labels.ndim!=1 or values.ndim!=1 or len(labels)!=len(values) or not len(labels):
        raise ValueError('Labels and scores must be nonempty aligned vectors')
    if not np.isin(labels,[0,1]).all() or not np.isfinite(values).all():
        raise ValueError('Binary labels and finite scores are required')
    return labels.astype(int),values

def youden_threshold(y,scores):
    """Maximize Youden J for scores > threshold; choose the largest tied optimum.

    Tied scores move together. The search includes all-negative and all-positive
    decisions. Calibration inputs are source-window and development validation
    observations.
    """
    labels,values=_arrays(y,scores)
    n_pos,n_neg=int(labels.sum()),int((labels==0).sum())
    if not n_pos or not n_neg: raise ValueError('Youden calibration requires both classes')
    order=np.argsort(-values,kind='stable')
    ss,sy=values[order],labels[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(values)-1]
    tp=np.cumsum(sy)[ends]; fp=ends+1-tp
    thresholds=np.r_[ss[0],np.nextafter(ss[ends],-np.inf)]
    objectives=np.r_[0.,tp/n_pos-fp/n_neg]
    return float(thresholds[int(np.argmax(objectives))])

def evaluate_metrics(y,scores,threshold):
    labels,values=_arrays(y,scores)
    if math.isnan(float(threshold)): raise ValueError('A calibrated threshold is required')
    predicted=(values>threshold).astype(int)
    both=len(np.unique(labels))==2
    auc=float(roc_auc_score(labels,values)) if both else float('nan')
    mf1=float(f1_score(labels,predicted,labels=[0,1],average='macro',zero_division=0))
    bacc=float(balanced_accuracy_score(labels,predicted)) if both else float('nan')
    return {'AUC':auc,'MacroF1':mf1,'BAcc':bacc,'auc':auc,'macro_f1':mf1,
            'balanced_accuracy':bacc,'threshold':float(threshold),'n':len(labels),'n_positive':int(labels.sum())}

def auc_standard_error(auc,n_pos,n_neg):
    """Hanley--McNeil standard error for the source-window one-SE band."""
    if not (math.isfinite(auc) and 0<=auc<=1 and n_pos>0 and n_neg>0): return float('nan')
    q1,q2=auc/(2.-auc),2.*auc*auc/(1.+auc)
    variance=(auc*(1.-auc)+(n_pos-1)*(q1-auc*auc)+(n_neg-1)*(q2-auc*auc))/(n_pos*n_neg)
    return math.sqrt(max(0.,variance))

auc_se_hanley_mcneil=auc_standard_error


def prepare_candidates(state, candidates, config):
    """Score each source validation set and each strictly later development year."""
    from clean import transform
    frame=candidates.copy().reset_index(drop=True)
    if frame.empty:return frame
    windows={int(w['index']):w for w in state['windows']}
    final_years=set(map(int,config['windows']['final_years']))
    development_years=sorted({int(y) for w in windows.values() for y in w['validation_years']})
    if final_years.intersection(development_years):raise ValueError('Development validation years overlap final test years')
    transformed={};records=[]
    for position,row in frame.iterrows():
        record=row.to_dict();wid=int(row.source_window);window=windows[wid]
        expression=str(row.expression);source_year=int(window['validation_year'])
        score=evaluate_expression(expression,window['x_validation'])
        threshold=youden_threshold(window['y_validation'],score)
        metrics=evaluate_metrics(window['y_validation'],score,threshold)
        labels=np.asarray(window['y_validation'])
        record.update(syntax_complexity(expression,len(state['features'])))
        record.update(candidate_id=(row.get('candidate_id') if pd.notna(row.get('candidate_id'))
                                    else f'{row.variant}_w{wid}_c{position+1}'),
                      canonical_expression=canonical(expression),validation_year=source_year,
                      source_year=source_year,validation_auc=metrics['auc'],
                      validation_mf1=metrics['macro_f1'],validation_bacc=metrics['balanced_accuracy'],
                      validation_threshold=threshold,n_pos=int((labels==1).sum()),n_neg=int((labels==0).sum()))
        forward=[];aucs=[];years=[]
        for year in development_years:
            if year<=source_year:continue
            key=(wid,year)
            if key not in transformed:
                cohort=state['data'].loc[state['data'].year==year]
                if cohort.empty:raise ValueError(f'Missing development cohort {year}')
                transformed[key]=(transform(cohort,window['preprocessor']),cohort.target.to_numpy())
            x,y=transformed[key];values=evaluate_expression(expression,x)
            m=evaluate_metrics(y,values,threshold)
            forward.append({'year':year,**{k:m[k] for k in ('auc','macro_f1','balanced_accuracy','n','n_positive')}})
            if np.isfinite(m['auc']):aucs.append(m['auc']);years.append(year)
        record.update(forward_auc=aucs,forward_years=years,forward_metrics=forward)
        records.append(record)
    result=pd.DataFrame(records)
    result.attrs['source_windows']=sorted(windows)
    result.attrs['final_test_years']=sorted(final_years)
    return result


def main(argv=None):
    from clean import cli_config,load_state,save_state
    config=cli_config('Prepare source-validation and forward candidate metrics',argv)
    state=load_state(config)
    state['prepared']=prepare_candidates(state,state['candidates'],config)
    state['prepared_gp']=prepare_candidates(state,state['gp_only'],config)
    for key in ('menus','selected','pools','forward','final','formulas'):state.pop(key,None)
    save_state(config,state)
    print(f"Prepared {len(state['prepared'])} archived expressions.")


if __name__=='__main__':main()
