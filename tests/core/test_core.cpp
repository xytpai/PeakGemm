#include <cassert>
#include <iostream>

#include "peak_gemm/peak_gemm.hpp"

int main() {
    using Tile = peak_gemm::core::Shape<128, 64, 32>;
    static_assert(Tile::dim == 3);
    static_assert(
        Tile::get<0>() == 128 && Tile::get<1>() == 64 && Tile::get<2>() == 32);
    using TensorShape = peak_gemm::core::Shape<2, 3, 4, 5>;
    static_assert(TensorShape::dim == 4);
    static_assert(TensorShape::extent(3) == 5);
    static_assert(peak_gemm::core::ceil_div(17, 8) == 3);
    static_assert(peak_gemm::core::Log2<32>::value == 5);

    using Layout2D = peak_gemm::core::StridedLayout<2>;
    using Layout3D = peak_gemm::core::StridedLayout<3>;
    constexpr Layout2D sliced{{32, 2}};
    constexpr Layout3D batched{{64, 8, 1}};
    assert(sliced(2, 3) == 70);
    assert(batched(2, 3, 4) == 156);
    assert(batched.stride(1) == 8);

    peak_gemm::core::Vector<int, 4> values{};
    values.fill(7);
    assert(values[0] == 7 && values[3] == 7);
    std::cout << "ok\n";
    return 0;
}
