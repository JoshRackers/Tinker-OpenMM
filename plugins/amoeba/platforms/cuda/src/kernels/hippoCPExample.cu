extern "C" __global__ void hippoCPExample()
{
    printf("Function hippoCPExample Thread = %2d Block = %2d\n", threadIdx.x, blockIdx.x);
}
