#!/usr/bin/env python3

import numpy as np
import pyopencl as cl
from tqdm import tqdm
import sys

kernel_source = """
__constant uint A0 = 0x67452301, B0 = 0xEFCDAB89, C0 = 0x98BADCFE, D0 = 0x10325476;
__constant uint s[64] = {
    7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
    5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
    4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
    6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
};

__constant uint K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,
    0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,
    0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,
    0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,
    0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,
    0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,
    0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,
    0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,
    0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

inline uint rotl(uint x, uint c) { return (x << c) | (x >> (32 - c)); }

inline void md5_transform(const uchar *msg, uint len, __private uchar *digest) {
    uint a = A0, b = B0, c = C0, d = D0;
    uint w[16] = {0};
    for(uint i=0; i<len; i++) w[i>>2] |= ((uint)msg[i]) << ((i%4)*8);
    w[len>>2] |= 0x80u << ((len%4)*8);
    w[14] = len * 8u;
    for(uint i=0; i<64; i++) {
        uint f, g;
        if(i<16) { f = (b & c) | (~b & d); g = i; }
        else if(i<32) { f = (d & b) | (~d & c); g = (5 * i + 1) % 16; }
        else if(i<48) { f = b ^ c ^ d; g = (3 * i + 5) % 16; }
        else { f = c ^ (b | ~d); g = (7 * i) % 16; }
        uint tmp = d; d = c; c = b; b = b + rotl(a + f + K[i] + w[g], s[i]); a = tmp;
    }
    uint H[4] = {A0 + a, B0 + b, C0 + c, D0 + d};
    for(int i=0; i<4; i++) {
        digest[i*4 + 0] = (uchar)(H[i] & 0xFF);
        digest[i*4 + 1] = (uchar)((H[i] >> 8) & 0xFF);
        digest[i*4 + 2] = (uchar)((H[i] >> 16) & 0xFF);
        digest[i*4 + 3] = (uchar)((H[i] >> 24) & 0xFF);
    }
}

__constant char hex_digits[16] = "0123456789abcdef";

__kernel void find_md5_matches(
    __global uint *match_count,
    __global uchar *plains,
    __global uchar *hashes,
    uint min_len,
    uint max_len,
    uint max_matches,
    uint charset_size,
    uint target_len,
    uint offset,
    __global const uchar *charset,
    __global const char *target_substr
){
    uint gid = get_global_id(0);
    if (gid >= get_global_size(0)) return;

    uchar plain[32] = {0};
    uint x = gid;

    for(uint i=0; i<max_len; i++){
        plain[i] = charset[x % charset_size];
        x /= charset_size;
    }

    for(uint len = min_len; len <= max_len; len++){
        uchar digest[16];
        md5_transform(plain, len, digest);

        int ok = 1;
        for (uint j = 0; j < target_len; j++) {
            uint bit_pos = offset + j;
            uchar byte = digest[bit_pos / 2];
            uchar nibble = (bit_pos % 2 == 0) ? (byte >> 4) & 0xF : byte & 0xF;
            if (hex_digits[nibble] != target_substr[j]) {
                ok = 0;
                break;
            }
        }

        if(ok){
            uint idx = atomic_inc(match_count);
            if(idx < max_matches){
                for(uint k=0; k<max_len; k++) plains[idx*max_len + k] = plain[k];

                // Save full hex digest for reference
                for(int i=0; i<16; i++){
                    hashes[idx*32 + 2*i]   = hex_digits[(digest[i] >> 4) & 0xF];
                    hashes[idx*32 + 2*i+1] = hex_digits[digest[i] & 0xF];
                }
            }
            break;
        }
    }
}
"""

def choose_local_size(kernel, device, global_size):
    max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    pref = kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
    local_size = pref if pref and pref <= max_work_group_size else max_work_group_size
    while local_size > 1 and global_size % local_size != 0:
        local_size -= 1
    return local_size if local_size > 0 else None

def main():
    if not (4 <= len(sys.argv) <= 5):
        print("Usage: python3 md5_vanity_search.py <target_substring> <min_length> <max_length> [offset]")
        return

    target_substr = sys.argv[1].lower()
    min_len = int(sys.argv[2])
    max_len = int(sys.argv[3])
    offset = int(sys.argv[4]) if len(sys.argv) == 5 else 0

    if not all(c in "0123456789abcdef" for c in target_substr):
        print("Target must be hexadecimal characters only (0-9a-f)")
        return
    if min_len <= 0 or max_len < min_len or max_len > 32:
        print("Invalid lengths: min_length > 0, max_length >= min_length, max_length <= 32")
        return
    if offset < 0 or offset + len(target_substr) > 32:
        print("Offset out of range. MD5 hex hash is 32 characters.")
        return

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    dev = queue.device
    mf = cl.mem_flags

    program = cl.Program(ctx, kernel_source).build()
    kernel = cl.Kernel(program, "find_md5_matches")

    charset = [i for i in range(32, 127)]
    charset_size = len(charset)
    total = pow(charset_size, max_len)

    charset_np = np.array(charset, dtype=np.uint8)
    charset_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=charset_np)

    target_bytes = np.array(list(target_substr.encode('ascii')), dtype=np.byte)
    target_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_bytes)

    chunk = 2**22
    max_matches = 10000000

    match_count = np.zeros(1, dtype=np.uint32)
    plains = np.zeros((max_matches, max_len), dtype=np.uint8)
    hashes = np.zeros((max_matches, 32), dtype=np.uint8)

    buf_count = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=match_count)
    buf_pl = cl.Buffer(ctx, mf.READ_WRITE, plains.nbytes)
    buf_hs = cl.Buffer(ctx, mf.READ_WRITE, hashes.nbytes)

    found_hashes = set()

    with open("results.txt", "w", encoding="utf-8") as fout:
        for offset_global in tqdm(range(0, total, chunk), total=(total + chunk - 1) // chunk):
            count_np = np.zeros(1, dtype=np.uint32)
            cl.enqueue_copy(queue, buf_count, count_np)

            kernel.set_arg(0, buf_count)
            kernel.set_arg(1, buf_pl)
            kernel.set_arg(2, buf_hs)
            kernel.set_arg(3, np.uint32(min_len))
            kernel.set_arg(4, np.uint32(max_len))
            kernel.set_arg(5, np.uint32(max_matches))
            kernel.set_arg(6, np.uint32(charset_size))
            kernel.set_arg(7, np.uint32(len(target_substr)))
            kernel.set_arg(8, np.uint32(offset))
            kernel.set_arg(9, charset_buf)
            kernel.set_arg(10, target_buf)

            gsize = min(chunk, total - offset_global)
            lsize = choose_local_size(kernel, dev, gsize) or 64

            cl.enqueue_nd_range_kernel(queue, kernel, (gsize,), (lsize,))

            cl.enqueue_copy(queue, count_np, buf_count)
            nfound = count_np[0]

            if nfound > 0:
                plains_np = np.empty_like(plains)
                hashes_np = np.empty_like(hashes)

                cl.enqueue_copy(queue, plains_np[:nfound], buf_pl)
                cl.enqueue_copy(queue, hashes_np[:nfound], buf_hs)
                queue.finish()

                for i in range(min(nfound, plains_np.shape[0])):
                    try:
                        hsh = bytes(hashes_np[i]).decode("ascii")
                        if hsh in found_hashes:
                            continue
                        found_hashes.add(hsh)
                        plain_str = bytes(plains_np[i][:max_len]).decode("ascii", errors="replace").rstrip("\x00")
                        fout.write(f"{plain_str} : {hsh}\n")
                        fout.flush()
                    except Exception as e:
                        print(f"Skipping entry {i}: {e}")

    print("Search complete.")

if __name__ == "__main__":
    main()
