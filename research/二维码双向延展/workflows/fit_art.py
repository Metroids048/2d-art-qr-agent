"""Fit guide shapes inside real QR cells using post-terminator padding bits."""
import numpy as np
import qrcode
from qrcode import base, constants, util


def optimize(payload, version, desired, priorities):
    correction = constants.ERROR_CORRECT_M
    blocks = base.rs_blocks(version, correction)
    size = sum(block.data_count for block in blocks)
    data = util.QRData(payload, mode=util.MODE_8BIT_BYTE)
    buf = util.BitBuffer()
    buf.put(data.mode, 4)
    buf.put(len(data), util.length_in_bits(data.mode, version))
    data.write(buf)
    if len(buf) + 4 >= size*8:
        raise ValueError("载荷过长。增大模板码宽或使用较短的原有入口")
    buf.put(0, 4)
    while len(buf) % 8:
        buf.put_bit(False)
    first_pad = len(buf) // 8
    for i in range(size-first_pad):
        buf.put((0xec, 0x11)[i % 2], 8)

    def matrix(mask):
        code = qrcode.QRCode(version=version, error_correction=correction, border=0)
        code.data_cache = util.create_bytes(buf, blocks)
        code.makeImpl(False, mask)
        return np.asarray(code.modules, dtype=np.uint8)

    bits = (size - first_pad) * 8
    baseline = matrix(0)
    deltas = np.empty((baseline.size, bits), dtype="uint8")
    for i in range(bits):
        byte, bit = first_pad + i//8, 1 << (i%8)
        buf.buffer[byte] ^= bit
        deltas[:, i] = (baseline ^ matrix(0)).ravel()
        buf.buffer[byte] ^= bit

    rows = [int.from_bytes(r.tobytes(), "little") for r in np.packbits(deltas, axis=1, bitorder="little")]
    order = np.argsort(-priorities.ravel(), kind="stable")
    goal = desired.ravel()
    best_score = float("-inf")
    for mask in range(8):
        start = matrix(mask)
        echelon = {}
        for pos in order:
            row, value = rows[pos], int(goal[pos] ^ start.ravel()[pos])
            while row:
                lead = row.bit_length() - 1
                if lead not in echelon:
                    echelon[lead] = row, value
                    break
                old, prior = echelon[lead]
                row ^= old
                value ^= prior

        chosen = 0
        for lead, (row, value) in sorted(echelon.items()):
            if (bin(chosen & row).count("1") % 2) ^ value:
                chosen |= 1 << lead
        padding = chosen.to_bytes(bits//8, "little")
        for j, variant in enumerate(padding):
            buf.buffer[first_pad + j] ^= variant
        final = matrix(mask)
        score = float(np.sum(priorities * (final == desired)))
        if score > best_score:
            best_score = score
            best = (final, start, dict(mask=mask, grid=len(final), pad_bits=bits,
                         filled=int(np.sum(final != start)), score=round(score, 3)))
        for j, variant in enumerate(padding):
            buf.buffer[first_pad + j] ^= variant
    return best
