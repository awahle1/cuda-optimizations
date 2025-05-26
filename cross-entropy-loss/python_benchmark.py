import torch
import torch.nn.functional as F

C = 8
N = 2

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
    main()