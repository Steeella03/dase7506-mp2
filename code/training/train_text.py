"""Reproducible NumPy-only text-feature boosting and linear experiments."""
import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from features.text_features import extract_features
from methods.text import tree_scores
from training_utils import load_supervised,selection_accuracy,softmax
from common import parse_answer


def fit_tree(bins,x,grad,hess,thresholds,depth,l2,min_leaf):
    nfeatures=x.shape[1]; nbins=32; nodes=2**(depth+1)-1
    feature=np.full(nodes,-1,dtype=np.int32); threshold=np.zeros(nodes,dtype=np.float32); values=np.zeros(nodes,dtype=np.float32)
    def fit(indices,node,level):
        g=grad[indices]; h=hess[indices]
        values[node]=np.clip(g.sum()/(h.sum()+l2),-4,4)
        if level==depth or len(indices)<2*min_leaf: return
        offset=(np.arange(nfeatures,dtype=np.int32)*nbins)[:,None]
        keys=(bins[indices].T+offset).ravel()
        gh=np.bincount(keys,weights=np.broadcast_to(g,(nfeatures,len(g))).ravel(),minlength=nfeatures*nbins).reshape(nfeatures,nbins)
        hh=np.bincount(keys,weights=np.broadcast_to(h,(nfeatures,len(h))).ravel(),minlength=nfeatures*nbins).reshape(nfeatures,nbins)
        cnt=np.bincount(keys,minlength=nfeatures*nbins).reshape(nfeatures,nbins)
        gl=np.cumsum(gh,axis=1)[:,:-1]; hl=np.cumsum(hh,axis=1)[:,:-1]; cl=np.cumsum(cnt,axis=1)[:,:-1]
        gr=gh.sum(axis=1)[:,None]-gl; hr=hh.sum(axis=1)[:,None]-hl
        gain=gl*gl/(hl+l2)+gr*gr/(hr+l2)
        gain[(cl<min_leaf)|((len(indices)-cl)<min_leaf)]=-np.inf
        fi,bi=np.unravel_index(np.argmax(gain),gain.shape)
        if not np.isfinite(gain[fi,bi]): return
        left=indices[bins[indices,fi]<=bi]; right=indices[bins[indices,fi]>bi]
        feature[node]=fi; threshold[node]=thresholds[fi,bi]
        fit(left,2*node+1,level+1); fit(right,2*node+2,level+1)
    fit(np.arange(len(x)),0,0)
    return feature,threshold,values


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--trees',type=int,default=180); parser.add_argument('--depth',type=int,default=3)
    parser.add_argument('--learning-rate',type=float,default=.06); parser.add_argument('--l2',type=float,default=10.)
    parser.add_argument('--min-leaf',type=int,default=50); parser.add_argument('--exclude',default='')
    parser.add_argument('--seed',type=int,default=2026); parser.add_argument('--output',type=Path,default=Path('runs/text/full'))
    parser.add_argument('--model',choices=['tree','linear'],default='tree'); parser.add_argument('--linear-steps',type=int,default=1200)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True); tick=time.perf_counter()
    train,_,y=load_supervised('train'); validation,_,yv=load_supervised('validation')
    hard=np.asarray([len({parse_answer(c['text']) for c in row['candidates'] if parse_answer(c['text']) is not None})>=2 for row in validation])
    cache=Path('runs/text/features.npz')
    if cache.exists():
        z=np.load(cache,allow_pickle=False); xt=z['train']; xv=z['validation']; names=list(z['names']); groups=list(z['groups'])
    else:
        xt,names,groups=extract_features(train,True); xv=extract_features(validation)
        np.savez(cache,train=xt,validation=xv,names=np.asarray(names),groups=np.asarray(groups))
    excluded=set(filter(None,args.exclude.split(','))); selected=np.array([i for i,g in enumerate(groups) if g not in excluded],dtype=np.int32)
    xt=xt[:,:,selected]; xv=xv[:,:,selected]; x=xt.reshape(-1,len(selected)); v=xv.reshape(-1,len(selected))
    # Unanimously correct groups contribute no ranking information.
    mixed=(y.sum(axis=1)<4).astype(np.float32)
    records=[]; best=-1; rng=np.random.default_rng(args.seed)
    if args.model=='tree':
        thresholds=np.quantile(x,np.arange(1,32)/32,axis=0).T.astype(np.float32)
        bins=np.empty(x.shape,dtype=np.int32)
        for k in range(x.shape[1]): bins[:,k]=np.searchsorted(thresholds[k],x[:,k],side='left')
        model=dict(model_type=np.array('tree'),feature_indices=selected,base_score=np.array(0.,dtype=np.float32),
                   depth=np.array(args.depth),learning_rate=np.array(args.learning_rate,dtype=np.float32))
        fs=[]; ts=[]; vs=[]; scores=np.zeros(y.shape,dtype=np.float32); valid_scores=np.zeros(yv.shape,dtype=np.float32)
        for step in range(1,args.trees+1):
            p=softmax(scores); target=y/np.maximum(y.sum(axis=1,keepdims=True),1)
            grad=((target-p)*mixed[:,None]).ravel(); hess=(np.maximum(p*(1-p),.05)*mixed[:,None]+.005).ravel()
            f,t,value=fit_tree(bins,x,grad,hess,thresholds,args.depth,args.l2,args.min_leaf)
            fs.append(f);ts.append(t);vs.append(value)
            one={**model,'tree_features':np.asarray([f]),'tree_thresholds':np.asarray([t]),'tree_values':np.asarray([value])}
            scores+=tree_scores(x,one).reshape(y.shape);valid_scores+=tree_scores(v,one).reshape(yv.shape)
            if step%10==0 or step==args.trees:
                acc=selection_accuracy(valid_scores,yv); record={'step':step,'validation_accuracy':acc,'hard_validation_accuracy':selection_accuracy(valid_scores[hard],yv[hard]),'hard_validation_n':int(hard.sum()),'train_accuracy':selection_accuracy(scores,y),'elapsed_seconds':time.perf_counter()-tick}
                records.append(record);print(json.dumps(record),flush=True)
                if acc>best:
                    best=acc; model.update(tree_features=np.asarray(fs),tree_thresholds=np.asarray(ts),tree_values=np.asarray(vs))
                    np.savez(args.output/'model.npz',**model)
    else:
        mean=x.mean(axis=0); scale=np.maximum(x.std(axis=0),.1); x=np.clip((x-mean)/scale,-8,8); v=np.clip((v-mean)/scale,-8,8)
        w=np.zeros(x.shape[1],dtype=np.float32); m=np.zeros_like(w); vv=np.zeros_like(w)
        target=y/np.maximum(y.sum(axis=1,keepdims=True),1)
        for step in range(1,args.linear_steps+1):
            scores=(x@w).reshape(y.shape); p=softmax(scores)
            gradient=x.T@((p-target)*mixed[:,None]).ravel()/max(1,mixed.sum())+.005*w
            m=.9*m+.1*gradient; vv=.999*vv+.001*gradient*gradient
            w-=.01*(m/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8)
            if step%100==0 or step==args.linear_steps:
                valid_scores=(v@w).reshape(yv.shape);acc=selection_accuracy(valid_scores,yv);record={'step':step,'validation_accuracy':acc,'hard_validation_accuracy':selection_accuracy(valid_scores[hard],yv[hard]),'hard_validation_n':int(hard.sum()),'elapsed_seconds':time.perf_counter()-tick};records.append(record);print(json.dumps(record),flush=True)
                if acc>best:
                    best=acc;np.savez(args.output/'model.npz',model_type=np.array('linear'),feature_indices=selected,mean=mean,scale=scale,weights=w,bias=np.array(0.))
    result={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'best_validation_accuracy':best,'records':records,'total_seconds':time.perf_counter()-tick,'features':len(selected),'asset_bytes':(args.output/'model.npz').stat().st_size}
    (args.output/'training.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__': main()
