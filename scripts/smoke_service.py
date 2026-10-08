#!/usr/bin/env python3
"""Bounded local three-process test of the inherited SGFPT simulation, no model/data."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--timeout', type=int, default=240)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--port', type=int, default=42200)
p.add_argument('--log-dir', default=str(root / 'logs/smoke'))
a = p.parse_args()
logs = Path(a.log_dir); logs.mkdir(parents=True, exist_ok=True)
procs = []
handles = []

def start(name, command):
    handle = (logs / f'{name}.log').open('w')
    handles.append(handle)
    proc = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                            env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    procs.append(proc)
    return proc

try:
    server = start('dh', [sys.executable, '-u', 'src/Client-Evaluation/vlm_server.py', '--dry-run',
                         '--rounds', '2', '--lambda', '4', '--dim', '8',
                         '--host', '127.0.0.1', '--port', str(a.port), '--gpu', str(a.gpu)])
    deadline = time.monotonic() + a.timeout
    while 'Listening on' not in (logs / 'dh.log').read_text():
        if server.poll() is not None:
            raise RuntimeError(f'DH exited before listening: {logs / "dh.log"}')
        if time.monotonic() > deadline:
            raise TimeoutError('DH startup timeout')
        time.sleep(0.2)
    common = ['127.0.0.1', '127.0.0.1', str(a.port), '4', '2', '8', '24',
              '1', '1', '3', str(a.gpu), '2']
    for party in [0, 1]:
        start(f'sp{party}', ['src/experiments/sgfpt/sgfpt_client', str(party), *common])
    for proc in procs[1:]:
        status = proc.wait(timeout=max(1, deadline - time.monotonic()))
        if status:
            raise RuntimeError(f'Participant exited with {status}; inspect {logs}')
    server.wait(timeout=15)
    if server.returncode:
        raise RuntimeError('DH failed')
    for party in [0, 1]:
        log = (logs / f'sp{party}.log').read_text()
        assert log.count('generation=') == 2, f'SP{party}: incomplete evolution loop'
        assert 'generation=2 complete' in log
    dh_log = (logs / 'dh.log').read_text()
    assert dh_log.count('Returning random fitness') == 2
    assert 'Traceback' not in dh_log
    print(f'PASS: two generations through Sample -> DH -> SelectTop -> Update. Logs: {logs}')
    print('This verifies the research simulation only, not privacy or paper accuracy.')
finally:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    for handle in handles: handle.close()
