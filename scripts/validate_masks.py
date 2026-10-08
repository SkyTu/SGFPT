#!/usr/bin/env python3
"""Numerical SGFPT diagnostics with zero and full-width nonzero test masks.

A nonzero exit status is a failing result, not an expected-pass baseline.
The two local processes emulate dealer key generation with a fixed TEST seed.
Run sequentially: the library uses fixed localhost peer ports.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--seeds', nargs='+', type=int, default=[12345, 67890])
p.add_argument('--mask-modes', nargs='+', choices=['zero', 'nonzero'], default=['zero', 'nonzero'])
p.add_argument('--only', nargs='+', choices=['sample0', 'sample1', 'select1', 'select3', 'update', 'chain', 'chain2'])
p.add_argument('--timeout', type=int, default=180)
p.add_argument('--log-dir', type=Path, default=root / 'logs/nonzero')
p.add_argument('--binary', type=Path, default=root / 'src/tests/spt/test_masked_protocols',
               help='Alternative diagnostic binary, for temporary source overlays')
p.add_argument('--sanitizer', help='Optional compute-sanitizer executable (memcheck)')
a = p.parse_args()
a.log_dir.mkdir(parents=True, exist_ok=True)
plan = [('sample0', 'sample', 0, 3), ('sample1', 'sample', 1, 3),
        ('select1', 'select', 0, 1), ('select3', 'select', 0, 3),
        ('update', 'update', 0, 3), ('chain', 'chain', 0, 3), ('chain2', 'chain2', 0, 3)]
results = []
for mask in a.mask_modes:
    for seed in (a.seeds[:1] if mask == 'zero' else a.seeds):
        for case, mode, generation, holders in plan:
            if a.only and case not in a.only:
                continue
            name = '{}-{}-{}'.format(case, mask, seed)
            processes, files, timed_out = [], [], False
            start = time.monotonic()
            try:
                for party in [0, 1]:
                    handle = (a.log_dir / '{}-p{}.log'.format(name, party)).open('w')
                    files.append(handle)
                    command = [str(a.binary.resolve()), str(party),
                               '127.0.0.1', mode, mask, str(seed), str(generation), str(holders)]
                    if a.sanitizer:
                        command = [a.sanitizer, '--tool', 'memcheck', '--error-exitcode', '99',
                                   '--force-blocking-launches', 'yes'] + command
                    processes.append(subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT))
                for process in processes:
                    try:
                        process.wait(timeout=max(1, a.timeout - (time.monotonic() - start)))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        break
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                for handle in files:
                    handle.close()
            fields = []
            for party in [0, 1]:
                log = (a.log_dir / '{}-p{}.log'.format(name, party)).read_text()
                fields.extend(json.loads(line[7:]) for line in log.splitlines() if line.startswith('RESULT '))
            entry = dict(case=case, mask=mask, seed=seed, generation=generation, holders=holders,
                         exit_codes=[pr.returncode for pr in processes], timeout=timed_out,
                         seconds=round(time.monotonic() - start, 3), fields=fields)
            expected_fields = {'sample': 2, 'select': 2, 'update': 5, 'chain': 9, 'chain2': 18}[mode] * 2
            entry['pass'] = (not timed_out and entry['exit_codes'] == [0, 0]
                             and len(fields) == expected_fields and all(f['bad'] == 0 for f in fields))
            results.append(entry)
            (a.log_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            summary = ', '.join('{}:{}/{}'.format(f['field'], f['bad'], f['n'])
                                for f in fields if f['party'] == 0)
            print('{} {} exit={} {} {:.1f}s'.format('PASS' if entry['pass'] else 'FAIL', name,
                                                   entry['exit_codes'], summary, entry['seconds']), flush=True)
print('Results: {}'.format(a.log_dir / 'results.json'))
sys.exit(0 if results and all(row['pass'] for row in results) else 1)
