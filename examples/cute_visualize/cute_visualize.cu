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
  std::cout << "1D Visualization:\n\n";
  Layout s8 = make_layout(Int<8>{});
  Layout d8 = make_layout(8);
  std::cout << "Layout (s8): ";
  print(s8);
  std::cout << "\n";
  print1D(s8);
  std::cout << "\nLayout (d8): ";
  print(d8);
  std::cout << "\n";
  print1D(d8);
  std::cout << "\n";
}

void print2Dexamples() {
  std::cout << "2D Visualization:\n\n";
  Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
  Layout s2xd4 = make_layout(make_shape(Int<2>{},4));
  std::cout << "Layout (s2xs4): ";
  print(s2xs4);
  std::cout << "\n";
  print2D(s2xs4);
  std::cout << "\nLayout (s2xd4): ";
  print(s2xd4);
  std::cout << "\n";
  print2D(s2xd4);
  std::cout << "\n";
}

int main() {
  print1Dexamples();
  print2Dexamples();
}