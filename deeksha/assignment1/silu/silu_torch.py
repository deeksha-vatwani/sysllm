# implement silu function w/matrix of shape (8192, 8192)
import torch
import torch.nn.functional as F


"""
given input tensor x, apply Sigmoid Linear 
Unit (SiLU) function, element-wise and 
return resulting tensor, functionally 
equivalent to pytorch's implementation
"""
def silu( x ):
    return x * sigmoid( x )


"""
given input tensor x, apply sigmoid function, 
element-wise and return resulting tensor, 
functionally equivalent to pytorch's implementation
"""
def sigmoid( x ):
    return 1 / ( 1 + torch.exp( -x ) )

"""
test silu implementation
"""
def main():
    shape = (8192, 8192)
    test_iters = 10
    print(f"testing silu implementation {test_iters} times")
    for i in range( test_iters ):
        input = torch.randn( shape )
        # test against pytorch
        expected = F.silu( input )
        actual = silu( input )
        if not torch.allclose( actual, expected ):
            print(f"expected: {expected}, actual: {actual}")
    print("testing complete")
        
if __name__ == '__main__':
    main()