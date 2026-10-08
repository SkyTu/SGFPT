#!/usr/bin/env python3
"""Run SGFPT CUDA tests in isolated pairs on localhost, with bounded lifetime."""
import argparse
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--timeout',type=int,default=180)
p.add_argument('--only',nargs='*',help='Optional case names for focused reruns')
a=p.parse_args()
logs=root/'logs/cuda';logs.mkdir(parents=True,exist_ok=True)

def pair(name, commands, cwd=root):
    if a.only and name not in a.only: return
    processes=[];files=[];deadline=time.monotonic()+a.timeout
    try:
        for party,command in enumerate(commands):
            f=(logs/f'{name}-p{party}.log').open('w');files.append(f)
            processes.append(subprocess.Popen(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT))
        for process in processes:
            status=process.wait(timeout=max(1,deadline-time.monotonic()))
            if status: raise RuntimeError(f'{name} exited {status}; see {logs}')
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill();process.wait()
        for f in files:f.close()
    print(f'PASS execution: {name}',flush=True)

for name in ['test_sample','test_select_top','test_update']:
    pair(name,[[f'src/tests/spt/{name}',str(i),'127.0.0.1','0'] for i in [0,1]])
print('SPT tests check execution only where the original test has no numerical assertions.')
print(f'Logs: {logs}')
