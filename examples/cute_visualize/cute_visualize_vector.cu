#include "cute/pointer.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/tensor_impl.hpp"
#include <cute/tensor.hpp>

#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

void visualizeVector() {
  auto const tensor_shape{cute::make_shape(16)};
  auto const global_memory_layout_src{cute::make_layout(tensor_shape)};
  std::cout << "Visualize initial layout layout: ";
  cute::print(global_memory_layout_src);

  thrust::host_vector<float> h_vec(16);
  thrust::sequence(h_vec.begin(), h_vec.end(), 0);
  auto tensor_src = cute::make_tensor(cute::make_gmem_ptr(h_vec.data()), global_memory_layout_src);
  using TileSizeX = cute::Int<8>;
  constexpr auto block_shape{cute::make_shape(TileSizeX{})};
  auto const tiled_tensor_src{cute::tiled_divide(tensor_src, block_shape)};

  std::cout << "\nTensor before tile: ";
  cute::print_tensor(tensor_src);
  std::cout << "\n";
  std::cout << "\nTensor after tiling with block shape 8: ";
  cute::print_tensor(tiled_tensor_src);
  std::cout << "\n";

  using ThreadBlockSizeX = cute::Int<2>;
  constexpr auto thread_block_shape{cute::make_shape(ThreadBlockSizeX{})};
  constexpr auto thread_layout{cute::make_layout(thread_block_shape)};  
  dim3 const grid_dim{static_cast<unsigned int>(cute::size<1>(tiled_tensor_src))};
  dim3 const thread_dim{static_cast<unsigned int>(cute::size(thread_layout))};

  std::cout << "Use 2 threads per block\nBlock shape: ";
  cute::print(thread_block_shape);
  std::cout << "\n";
  std::cout << "Thread layout: ";
  cute::print(thread_layout);
  std::cout << "\n";
  cute::print(grid_dim);
  std::cout << "\n";
  cute::print(thread_dim);
  std::cout << "\n";

  std::cout << "\nAccessing original tensor with coordinates:\n";
  for (int i = 0; i < 4; i++) {
    if (i < cute::size<1>(tiled_tensor_src)) {
      auto global_tile = tiled_tensor_src(cute::_, i);
      std::cout << "Tile " << i << ": ";
      cute::print_tensor(global_tile);
      std::cout << "\n";
    }
  }
  
}

int main() {
  visualizeVector();
}