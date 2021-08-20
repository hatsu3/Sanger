import torch
import numpy as np


def bench_qk_pv(seq_len=512, head_size=64, number=10000, repeats=10):
    q = torch.randn(seq_len, head_size).cuda()
    k = torch.randn(head_size, seq_len).cuda()
    v = torch.randn(seq_len, head_size).cuda()

    def run_func():
        p = torch.matmul(q, k)
        o = torch.matmul(p, v)

    run_func()
    bench_res = []

    for i in range(repeats):
        torch.cuda.synchronize()
        
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        
        tic.record()
        
        for j in range(number):
            run_func()

        toc.record()
        torch.cuda.synchronize()

        elapsed = tic.elapsed_time(toc)
        bench_res.append(elapsed / number)

    return bench_res


if __name__ == "__main__":
    seq_len = 512
    bench_res = bench_qk_pv(seq_len=seq_len)
    # NB: PyTorch’s framework overhead is not negligible when profiling small workloads, use `nvprof` instead.
    # print(f"(seq_len: {seq_len}) mean: {np.mean(bench_res):.4f} ms, std: {np.std(bench_res):.4f} ms")
