#include <cuda_runtime.h>
#include "silu.h"

int main() {
    // Launch the kernel

    size_t num_elems = 100000000;
    float* input = new float[ num_elems ];
    for( int i = 0; i < num_elems; i++ ) {
        input[ i ] = static_cast< float >( i );
    }


}