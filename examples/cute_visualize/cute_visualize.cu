#include <cute/tensor.hpp>

using namespace cute;

template<class Shape, class Stride>
void print1D(Layout<Shape, Stride> const& layout) {
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
  printf("\n");
}

template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}

void print1Dexamples() {
  std::cout << "1D Visualization: " << "\n";
  Layout s8 = make_layout(Int<8>{});
  Layout d8 = make_layout(8);
  std::cout << "s8" << "\n";
  print1D(s8);
  std::cout << "d8" << "\n";
  print1D(d8);
}

void print2Dexamples() {
  std::cout << "2D Visualization: " << "\n";
  Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
  Layout s2xd4 = make_layout(make_shape(Int<2>{},4));
  std::cout << "s2xs4" << "\n";
  print2D(s2xs4);
  std::cout << "s2xd4" << "\n";
  print2D(s2xd4);
}

int main() {
  print1Dexamples();
  print2Dexamples();
}