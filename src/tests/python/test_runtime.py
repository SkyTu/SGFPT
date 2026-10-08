"""Regression checks for the extracted runtime and its cross-language protocol."""
from pathlib import Path
import socket
import struct
import sys
import threading
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Client-Evaluation'))
from protocol import HEADER_SIZE, MAGIC, PromptConnection, fixed_to_float, float_to_fixed
from algorithm.sep_cma_es import SepCMAES

class RuntimeTests(unittest.TestCase):
    def test_signed_fixed_point(self):
        expected = np.array([-2.5, -0.25, 0, 1.5])
        for bits in (32, 64):
            np.testing.assert_array_equal(fixed_to_float(float_to_fixed(expected, bits, 24), bits, 24), expected)

    def test_fragmented_wire_frame_and_fitness(self):
        left, right = socket.socketpair()
        left.settimeout(5); right.settimeout(5)
        data = np.array([-1.25, 2.5, 0, 0.125])
        wire = struct.pack('<4sQQQQQ', b'B2TP', 2, 2, 64, 24, 3) + float_to_fixed(data, 64, 24).astype('<u8').tobytes()
        def send():
            for offset in range(0, len(wire), 3): right.sendall(wire[offset:offset+3])
        worker = threading.Thread(target=send); worker.start()
        try:
            conn = PromptConnection(left, ('local', 0))
            raw, lam, d, bw, scale, parties = conn.recv_raw_share()
            self.assertEqual(HEADER_SIZE, 44)
            self.assertEqual((lam,d,bw,scale,parties),(2,2,64,24,3))
            np.testing.assert_array_equal(fixed_to_float(raw,bw,scale),data)
            scores=np.array([0.1,0.1,0.1,0.2,0.2,0.2])
            conn.send_fitness(scores)
            np.testing.assert_array_equal(np.frombuffer(right.recv(48, socket.MSG_WAITALL),dtype='<f8'),scores)
        finally:
            worker.join(); left.close(); right.close()

    def test_truncated_frame_fails(self):
        left,right=socket.socketpair()
        right.sendall(MAGIC);right.close()
        try:
            with self.assertRaises(ConnectionError): PromptConnection(left,('local',0)).recv_raw_share()
        finally: left.close()

    def test_optimizer_mean_follows_best_candidates(self):
        opt=SepCMAES({'intrinsic_dim_L':4,'intrinsic_dim_V':0,'popsize':4,'seed':7})
        candidates=np.asarray(opt.ask())
        # Known ordering: candidates 2 and 0 are selected; sigma starts at 1, mean at 0.
        expected_mean=opt.w @ candidates[[2,0]]
        opt.tell(candidates,[1.,3.,0.,2.])
        np.testing.assert_allclose(opt.m,expected_mean,atol=1e-12)
        self.assertTrue(np.all(np.isfinite(opt.C)))
        self.assertTrue(np.all(opt.C>0))
        self.assertTrue(np.isfinite(opt.sigma) and opt.sigma>0)

if __name__=='__main__': unittest.main()
