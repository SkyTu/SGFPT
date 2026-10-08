#!/usr/bin/env python3
"""Run small real-model forward passes using existing local weights and data."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src/Client-Evaluation'))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',required=True)
p.add_argument('--roberta-path',required=True)
p.add_argument('--mrpc-arrow',required=True)
a=p.parse_args()
from vlm_server import build_prompt_clip
from llm_server import PromptRoBERTa, PromptTuningDataset, get_verbalizer_ids, evaluate_prompt
from algorithm.sep_cma_es import SepCMAES
from datasets import Dataset

np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
cfg=yaml.safe_load((root/'src/Client-Evaluation/configs/vlm.yaml').read_text())
cfg.update(cfg['cifar100'])
cfg.update(data_dir=a.data_root,batch_size=2,opt_name='sep_cma_es',output_dir='logs/real-inference',parallel=False)
clip_model,train_loader,test_loader=build_prompt_clip('cifar100',cfg,'cuda:0')
batch=next(iter(train_loader))
opt=SepCMAES({**cfg,'popsize':2})
candidates=opt.ask()
fitness=[]
for z in candidates:
    pt=clip_model.generate_text_prompts([z[:200]])[0]
    pv=clip_model.generate_visual_prompts([z[200:]])[0]
    loss,acc=clip_model.eval((pt,pv),batch,ii=0,r=1)
    assert np.isfinite(loss) and 0<=acc<=1
    fitness.append(loss)
opt.tell(candidates,fitness)
assert np.isfinite(opt.sigma) and np.all(opt.C>0)
print(f'PASS VLM: actual CIFAR100 batch of 2, CLIP ViT-B/16, two candidate CE losses={fitness}, one plaintext CMA update',flush=True)
del clip_model,train_loader,test_loader,batch,pt,pv
import gc;gc.collect();torch.cuda.empty_cache()
model=PromptRoBERTa(a.roberta_path,50,500,'cuda:0',seed=42)
mrpc=Dataset.from_file(a.mrpc_arrow).select(range(2))
ds=PromptTuningDataset(mrpc,'mrpc',model.tokenizer,max_seq_length=64)
loader=torch.utils.data.DataLoader(ds,batch_size=2)
loss,acc=evaluate_prompt(model,np.zeros(500),loader,get_verbalizer_ids('mrpc',model.tokenizer),'cuda:0',micro_batch=2)
assert np.isfinite(loss) and 0<=acc<=1
print(f'PASS LLM: actual MRPC batch of 2, cached RoBERTa-Large, loss={loss}, accuracy={acc}',flush=True)
print('These are bounded deployment checks, not reproduction of the paper tables.')
