import torch
from silu_triton_kernel import silu_triton
import torch.nn.functional as F

# Test the Triton kernel

def main():
    print('hello world')
    shape = (8192, 8192)
    test_iters = 10
    print(f"testing silu triton implementation {test_iters} times")
    for i in range( test_iters ):
        input = torch.randn( shape )
        # test against pytorch
        expected = F.silu( input )
        actual = silu_triton( input )
        if not torch.allclose( actual, expected ):
            print(f"expected: {expected}, actual: {actual}")
    print("testing complete")

    
if __name__ == "__main__":
    main()
    