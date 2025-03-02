# Notes on vector copy

These are my learning notes on the vector copy example. Please see in `cute_visualize` to make your own visualizations which can help in understanding.

## `cute_vector_copy`

### `launch_vector_copy`

```cpp
template <typename T>
static cudaError_t launch_vector_copy(T const* input_vector, T* output_vector,
                                      unsigned int size, cudaStream_t stream)
```

We parametrize over the type of the vector we copy.

```cpp
    auto const tensor_shape{cute::make_shape(size)};
    auto const global_memory_layout_src{cute::make_layout(tensor_shape)};
    auto const global_memory_layout_dst{cute::make_layout(tensor_shape)};

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_vector),
                                            global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_vector),
                                            global_memory_layout_dst)};

    using TileSizeX = cute::Int<2048>;

    constexpr auto block_shape{cute::make_shape(TileSizeX{})};

    auto const tiled_tensor_src{cute::tiled_divide(tensor_src, block_shape)};
    auto const tiled_tensor_dst{cute::tiled_divide(tensor_dst, block_shape)};
```

Create the tensors in global memory. In a next step tile it according to block shape.

```cpp
    using ThreadBlockSizeX = cute::Int<256>;

    constexpr auto thread_block_shape{cute::make_shape(ThreadBlockSizeX{})};
    constexpr auto thread_layout{cute::make_layout(thread_block_shape)};

    dim3 const grid_dim{cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{cute::size(thread_layout)};

    CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeX{} == cute::Int<0>{},
                         "TileSizeX must be divisible by ThreadBlockSizeX");
    vector_copy<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst, size, thread_layout);

    return cudaGetLastError();
```

Launch the kernel with predefined thread block size.

We can visualize as follows for a vector of `size=16` with `TileSizeX=4`

```
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
---
0 1 2 3 4   <-> Tile 0
5 6 7 8 9   <-> Tile 1
10 11 12 13 <-> Tile 2
14 15 16 17 <-> Tile 3
```

### `vector_copy`

```cpp
template <class TensorSrc, class TensorDst, class ThreadLayout>
static __global__ void vector_copy(TensorSrc tensor_src, TensorDst tensor_dst,
                                   unsigned int size, ThreadLayout)
```

As above we parametrize over the type of the vector to copy. We also parametrize over the Thread Layout.

```cpp
    using Element = typename TensorSrc::value_type;

    auto global_tile_src{tensor_src(cute::make_coord(cute::_), blockIdx.x)};
    auto global_tile_dst{tensor_dst(cute::make_coord(cute::_), blockIdx.x)};

    auto thread_global_tile_src{
        cute::local_partition(global_tile_src, ThreadLayout{}, threadIdx.x)};
    auto thread_global_tile_dst{
        cute::local_partition(global_tile_dst, ThreadLayout{}, threadIdx.x)};
```

We access the specific tile tile. We than partition the tile according to the thread layout and access the corresponding element for the current thread.


```cpp
    auto const identity_tensor{cute::make_identity_tensor(
        cute::make_shape(cute::size(global_tile_src)))};
    auto const thread_identity_tensor{
        cute::local_partition(identity_tensor, ThreadLayout{}, threadIdx.x)};

    auto fragment{cute::make_fragment_like(thread_global_tile_src)};
    auto predicator{
        cute::make_tensor<bool>(cute::make_shape(cute::size(fragment)))};

    constexpr auto tile_size{cute::size<0>(global_tile_src)};

    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size(predicator); ++i)
    {
        auto const thread_identity{thread_identity_tensor(i)};
        bool const is_in_bound{
            cute::get<0>(thread_identity) + blockIdx.x * tile_size < size};
        predicator(i) = is_in_bound;
    }

    cute::copy_if(predicator, thread_global_tile_src, fragment);
    cute::copy_if(predicator, fragment, thread_global_tile_dst);
```

This is the main logic. We use a predicator to use it for bound checking. We than use `cute::copy_if` to copy the vector.