#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "peak_gemm/data.hpp"

constexpr auto cpu = peak_gemm::DataDevice::cpu;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

template <typename Exception, typename Function>
void expect_throw(Function &&function) {
    bool thrown = false;
    try {
        function();
    } catch (const Exception &) {
        thrown = true;
    }
    assert(thrown);
}

template <typename T>
void expect_values(const peak_gemm::Data<T> &data, const std::vector<T> &expected) {
    assert(data.device() == cpu);
    assert(data.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert(data[i] == expected[i]);
    }
}

void test_cpu_data() {
    using Data = peak_gemm::Data<int>;
    static_assert(std::is_same_v<Data::value_type, int>);
    static_assert(std::is_same_v<Data::Shape, std::vector<std::size_t>>);
    static_assert(!std::is_copy_constructible_v<Data>);
    static_assert(!std::is_copy_assignable_v<Data>);
    static_assert(std::is_nothrow_move_constructible_v<Data>);
    static_assert(std::is_nothrow_move_assignable_v<Data>);

    Data data({2, 3});
    assert(data.shape() == Data::Shape({2, 3}));
    assert(data.dim() == 2);
    assert(data.size() == 6);
    assert(data.bytes() == 6 * sizeof(int));
    assert(data.device() == cpu);
    assert(data.device_index() == -1);
    assert(data.data() != nullptr);
    data.fill(7);
    expect_values(data, {7, 7, 7, 7, 7, 7});
    data[2] = 11;
    const Data &constant = data;
    assert(constant.data() == data.data());
    assert(constant[2] == 11);

    Data scalar(Data::Shape{});
    assert(scalar.dim() == 0);
    assert(scalar.size() == 1);
    scalar[0] = 5;
    assert(scalar[0] == 5);

    auto copy = data.copy_to(cpu);
    assert(copy.data() != data.data());
    expect_values(copy, {7, 7, 11, 7, 7, 7});
    auto *pointer = copy.data();
    copy.to(cpu);
    assert(copy.data() == pointer);
    expect_values(copy, {7, 7, 11, 7, 7, 7});
}

void test_move_semantics() {
    using Data = peak_gemm::Data<int>;
    Data source({4});
    source.fill(3);
    auto *source_pointer = source.data();
    Data moved(std::move(source));
    assert(source.data() == nullptr);
    assert(source.size() == 0);
    assert(moved.data() == source_pointer);
    expect_values(moved, {3, 3, 3, 3});

    Data destination({2});
    destination.fill(9);
    destination = std::move(moved);
    assert(moved.data() == nullptr);
    assert(moved.size() == 0);
    assert(destination.data() == source_pointer);
    expect_values(destination, {3, 3, 3, 3});
    destination = std::move(destination);
    expect_values(destination, {3, 3, 3, 3});
}

void test_random_data() {
    using Data = peak_gemm::Data<float>;
    Data first({1024});
    Data second({1024});
    first.random_uniform(-2.0F, 3.0F, 100);
    second.random_uniform(-2.0F, 3.0F, 100);
    for (std::size_t i = 0; i < first.size(); ++i) {
        assert(first[i] >= -2.0F && first[i] <= 3.0F);
        assert(first[i] == second[i]);
    }

    first.random_normal(2.0F, 0.5F, 200);
    second.random_normal(2.0F, 0.5F, 200);
    for (std::size_t i = 0; i < first.size(); ++i) {
        assert(std::isfinite(first[i]));
        assert(first[i] == second[i]);
    }

    auto uniform = Data::uniform({128}, -4.0F, -1.0F, cpu, 300);
    auto uniform_again = Data::uniform({128}, -4.0F, -1.0F, cpu, 300);
    auto normal = Data::normal({128}, 1.0F, 2.0F, cpu, 400);
    auto normal_again = Data::normal({128}, 1.0F, 2.0F, cpu, 400);
    for (std::size_t i = 0; i < uniform.size(); ++i) {
        assert(uniform[i] >= -4.0F && uniform[i] <= -1.0F);
        assert(uniform[i] == uniform_again[i]);
        assert(std::isfinite(normal[i]));
        assert(normal[i] == normal_again[i]);
    }
}

void test_validation() {
    using Data = peak_gemm::Data<int>;
    expect_throw<std::invalid_argument>([] { Data data({2, 0, 3}); });
    expect_throw<std::overflow_error>([] {
        Data data(Data::Shape{std::numeric_limits<std::size_t>::max(), 2});
    });
    expect_throw<std::overflow_error>([] {
        Data data(Data::Shape{
            std::numeric_limits<std::size_t>::max() / sizeof(int) + 1});
    });
    expect_throw<std::invalid_argument>([] {
        Data data({4}, gpu, -1);
    });
    Data target_validation({4});
    expect_throw<std::invalid_argument>([&] {
        (void)target_validation.copy_to(gpu, -1);
    });
    expect_throw<std::invalid_argument>([&] {
        target_validation.to(gpu, -1);
    });

    peak_gemm::Data<float> data({8});
    expect_throw<std::invalid_argument>([&] {
        data.random_uniform(1.0F, 1.0F);
    });
    expect_throw<std::invalid_argument>([&] {
        data.random_uniform(2.0F, 1.0F);
    });
    expect_throw<std::invalid_argument>([&] {
        data.random_normal(0.0F, 0.0F);
    });
    expect_throw<std::invalid_argument>([&] {
        data.random_normal(0.0F, -1.0F);
    });
}

void test_gpu_data() {
    using Data = peak_gemm::Data<int>;
    int device_count = 0;
    assert(gpuGetDeviceCount(&device_count) == gpuSuccess);
    assert(device_count > 0);
    int original_device = 0;
    assert(gpuGetDevice(&original_device) == gpuSuccess);

    Data host({2, 3});
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<int>(i + 1);
    }
    auto device = host.copy_to(gpu, 0);
    assert(device.shape() == host.shape());
    assert(device.device() == gpu);
    assert(device.device_index() == 0);
    expect_throw<std::logic_error>([&] { (void)device[0]; });
    auto round_trip = device.copy_to(cpu);
    expect_values(round_trip, {1, 2, 3, 4, 5, 6});

    auto device_copy = device.copy_to(gpu, 0);
    auto device_copy_host = device_copy.copy_to(cpu);
    expect_values(device_copy_host, {1, 2, 3, 4, 5, 6});
    device_copy.fill(12);
    expect_values(device_copy.copy_to(cpu), {12, 12, 12, 12, 12, 12});

    assert(gpuSetDevice(device_count - 1) == gpuSuccess);
    device.set_current_device();
    int current_device = -1;
    assert(gpuGetDevice(&current_device) == gpuSuccess);
    assert(current_device == 0);
    host.set_current_device();
    assert(gpuGetDevice(&current_device) == gpuSuccess);
    assert(current_device == 0);

    auto gpu_uniform =
        peak_gemm::Data<float>::uniform({256}, -3.0F, 2.0F, gpu, 500, 0);
    auto gpu_normal =
        peak_gemm::Data<float>::normal({256}, 1.0F, 0.5F, gpu, 600, 0);
    auto cpu_uniform = gpu_uniform.copy_to(cpu);
    auto cpu_normal = gpu_normal.copy_to(cpu);
    for (std::size_t i = 0; i < cpu_uniform.size(); ++i) {
        assert(cpu_uniform[i] >= -3.0F && cpu_uniform[i] <= 2.0F);
        assert(std::isfinite(cpu_normal[i]));
    }

    Data migrating({6});
    for (std::size_t i = 0; i < migrating.size(); ++i) {
        migrating[i] = static_cast<int>(10 + i);
    }
    migrating.to(gpu, 0);
    assert(migrating.device() == gpu);
    assert(migrating.device_index() == 0);
    auto *gpu_pointer = migrating.data();
    migrating.to(gpu, 0);
    assert(migrating.data() == gpu_pointer);
    migrating.to(cpu);
    assert(migrating.device() == cpu);
    assert(migrating.device_index() == -1);
    expect_values(migrating, {10, 11, 12, 13, 14, 15});

    if (device_count > 1) {
        auto peer = device.copy_to(gpu, 1);
        assert(peer.device_index() == 1);
        expect_values(peer.copy_to(cpu), {1, 2, 3, 4, 5, 6});
        peer.to(gpu, 0);
        assert(peer.device_index() == 0);
        expect_values(peer.copy_to(cpu), {1, 2, 3, 4, 5, 6});
    }
    assert(gpuSetDevice(original_device) == gpuSuccess);
}

int main() {
    test_cpu_data();
    test_move_semantics();
    test_random_data();
    test_validation();
    test_gpu_data();
    std::cout << "ok\n";
    return 0;
}
