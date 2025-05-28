import time
import torch
import torch.nn.functional as F

C = 512
N = 1048576

def benchmark():
    input = torch.randn(N, C, requires_grad=False)
    target = torch.randint(C, (N,), dtype=torch.int64, requires_grad=False)
    loss = F.cross_entropy(input, target)

    # Ensure all CUDA operations are finished
    torch.cuda.synchronize()  

    total_time = 0
    n_iters = 5

    for i in range(n_iters):
        # Measure time
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        start = time.time()
        _ = F.cross_entropy(input, target)
        torch.cuda.synchronize()  # Synchronize again
        end = time.time()
        
        total_time += (end - start) * 1000
        print(total_time)

    print(f"Cross entropy computation time (average): {(total_time/n_iters):.3f} ms")

def main():
    input = torch.randn(N, C, requires_grad=False)
    target = torch.randint(C, (N,), dtype=torch.int64)
    loss = F.cross_entropy(input, target)

    # Flatten the tensors
    input_flat = input.flatten().tolist()
    target_flat = target.flatten().tolist()

    # Write to a file
    with open("output.txt", "w") as f:
        f.write(f"input = {input_flat}\n")
        f.write(f"target = {target_flat}\n")
        f.write(f"loss = {loss}\n")

if __name__ == "__main__":
    benchmark()