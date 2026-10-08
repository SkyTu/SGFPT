"""SGFPT research-simulation TCP protocol (little-endian, original B2TP magic).

Header: <4sQQQQQ = magic, lambda, d, bw, scale, party_num (44 bytes).
Payload: lambda*d uint64 fixed-point values. Reply: lambda*party_num float64.
Both C++ parties currently send duplicate zero-mask values; the DH uses one
copy, and returns the same unencrypted scores to both. This is not additive
secret sharing. bw=64 values decode as int64 / 2**scale.
"""

import socket
import struct
import numpy as np

# ──────────────────────────────────────────────────────────────
# Protocol constants
# ──────────────────────────────────────────────────────────────
MAGIC = b'B2TP'
HEADER_FMT = '<4sQQQQQ'   # magic(4) + lambda(8) + d(8) + bw(8) + scale(8) + party_num(8)
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 44 bytes


# ──────────────────────────────────────────────────────────────
# Low-level socket helpers
# ──────────────────────────────────────────────────────────────
def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Block until exactly n bytes have been received."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Peer closed connection while receiving data")
        buf.extend(chunk)
    return bytes(buf)


def send_exactly(sock: socket.socket, data: bytes) -> None:
    """Block until all bytes have been sent."""
    total = 0
    view = memoryview(data)
    while total < len(data):
        sent = sock.send(view[total:])
        if sent == 0:
            raise ConnectionError("Peer closed connection while sending data")
        total += sent


# ──────────────────────────────────────────────────────────────
# Fixed-point ↔ float conversion
# ──────────────────────────────────────────────────────────────
def fixed_to_float(arr_u64: np.ndarray, bw: int, scale: int) -> np.ndarray:
    """
    Convert an array of u64 fixed-point values to float64.

    Mirrors SecPromptTuning's asFloat(x, bw, scale):
        result = (int64_t)(sign_extend(x, bw)) / 2^scale

    For bw=64 this is just a bit-cast to int64 then divide by 2^scale.
    For bw<64 the bw-th bit is treated as the sign bit.
    """
    arr = arr_u64.copy().view(np.int64)   # reinterpret u64 bits as i64
    if bw < 64:
        mask = np.int64((1 << bw) - 1)
        arr &= mask
        sign_bit = np.int64(1 << (bw - 1))
        # Two's-complement sign extension
        arr -= ((arr & sign_bit) << 1)
    return arr.astype(np.float64) / float(1 << scale)


def float_to_fixed(arr_f64: np.ndarray, bw: int, scale: int) -> np.ndarray:
    """
    Convert float64 array to u64 fixed-point (inverse of fixed_to_float).
    Saturates at the representable range for the given bw.
    """
    scaled = np.round(np.asarray(arr_f64, dtype=np.float64) * float(1 << scale))
    signed = scaled.astype(np.int64)
    if bw < 64:
        lo = -(1 << (bw - 1))
        hi = (1 << (bw - 1)) - 1
        signed = np.clip(signed, lo, hi)
    return signed.view(np.uint64)


# ──────────────────────────────────────────────────────────────
# Server-side connection wrapper
# ──────────────────────────────────────────────────────────────
class PromptConnection:
    """
    Wraps one accepted TCP connection from a SecPromptTuning party.

    Typical usage (inside the inference server loop):
        conn = server.accept()
        prompts, lambda_, d, bw, scale, party_num = conn.recv_prompts()
        ...evaluate...
        conn.send_fitness(fitness_array)
        conn.close()
    """

    def __init__(self, sock: socket.socket, addr):
        self.sock = sock
        self.addr = addr

    def recv_prompts(self):
        """
        Receive a batch of prompt vectors.

        Returns
        -------
        prompts   : np.ndarray, shape (lambda_, d), dtype float64
        lambda_   : int   – number of candidates
        d         : int   – intrinsic dimension per candidate
        bw        : int   – fixed-point bit-width (usually 64)
        scale     : int   – fixed-point scale  (usually 24)
        party_num : int   – number of data-holding parties
        """
        header = recv_exactly(self.sock, HEADER_SIZE)
        magic, lambda_, d, bw, scale, party_num = struct.unpack(HEADER_FMT, header)
        if magic != MAGIC:
            raise ValueError(f"Bad magic bytes: {magic!r}, expected {MAGIC!r}")

        raw_bytes = recv_exactly(self.sock, int(lambda_ * d * 8))
        raw_u64 = np.frombuffer(raw_bytes, dtype=np.uint64).copy()
        prompts = fixed_to_float(raw_u64, int(bw), int(scale)).reshape(int(lambda_), int(d))
        return prompts, int(lambda_), int(d), int(bw), int(scale), int(party_num)

    def recv_raw_share(self):
        """
        Receive raw u64 words without converting to float.
        Their representation is determined by the sender. The current SGFPT
        simulation sends duplicate zero-mask openings; its caller uses one
        copy. This method does not add shares or remove a mask.

        Returns
        -------
        raw_u64   : np.ndarray, dtype=uint64, shape (lambda_ * d,)
        lambda_   : int
        d         : int
        bw        : int
        scale     : int
        party_num : int
        """
        header = recv_exactly(self.sock, HEADER_SIZE)
        magic, lambda_, d, bw, scale, party_num = struct.unpack(HEADER_FMT, header)
        if magic != MAGIC:
            raise ValueError(f"Bad magic bytes: {magic!r}, expected {MAGIC!r}")
        raw_bytes = recv_exactly(self.sock, int(lambda_ * d * 8))
        raw_u64 = np.frombuffer(raw_bytes, dtype=np.uint64).copy()
        return raw_u64, int(lambda_), int(d), int(bw), int(scale), int(party_num)

    def send_fitness(self, fitness: np.ndarray) -> None:
        """
        Send the supplied flat float64 fitness array to the C++ party.

        Parameters
        ----------
        fitness : one-dimensional array-like (lambda_ * party_num in SGFPT)
            Fitness values (e.g. cross-entropy loss; lower = better for CMA-ES).
        """
        data = np.asarray(fitness, dtype=np.float64).tobytes()
        send_exactly(self.sock, data)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ──────────────────────────────────────────────────────────────
# Server wrapper
# ──────────────────────────────────────────────────────────────
class PromptServer:
    """
    TCP server that C++ parties connect to for each evaluation round.

    Example
    -------
    server = PromptServer(port=42200)
    server.start()
    while True:
        conn = server.accept()
        prompts, lam, d, bw, scale, party_num = conn.recv_prompts()
        fitness = evaluate(prompts)
        conn.send_fitness(fitness)
        conn.close()
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 42200):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(5)
        print(f"[SGFPT] Server listening on {self.host}:{self.port}")

    def accept(self) -> PromptConnection:
        conn_sock, addr = self._sock.accept()
        print(f"[SGFPT] Accepted connection from {addr[0]}:{addr[1]}")
        return PromptConnection(conn_sock, addr)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()
