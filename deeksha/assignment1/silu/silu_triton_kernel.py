import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel( input_ptr, output_ptr, num_elems, BLOCK_SIZE: tl.constexpr ):
    pid = tl.program_id( axis=0 )
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange( 0, BLOCK_SIZE )
    
    mask = offsets < num_elems
    input = tl.load( input_ptr + offsets, mask=mask )
    
    output = input * ( 1 / ( 1 + tl.exp( -input ) ) )
    tl.store( output_ptr + offsets, output, mask=mask )


def silu_triton(input):
    output = torch.empty_like( input )
    num_elems = output.numel()
    input = input.to("cuda")
    output = output.to("cuda")
    grid = lambda meta: ( triton.cdiv( num_elems, meta[ 'BLOCK_SIZE' ] ), )
    silu_kernel[ grid ]( input, output, num_elems, BLOCK_SIZE=64 )
    return output.to("cpu")
