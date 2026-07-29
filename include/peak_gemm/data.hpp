#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "peak_gemm/backend/runtime.hpp"

namespace peak_gemm {

enum class DataDevice {
    cpu,
    gpu,
};

template <typename T>
class Data {
public:
    using value_type = T;
    using Shape = std::vector<std::size_t>;

    static_assert(
        std::is_trivially_copyable_v<T>,
        "Data requires a trivially copyable value type");

    explicit Data(
        Shape shape,
        DataDevice device = DataDevice::cpu,
        int device_index = 0) :
        shape_(std::move(shape)),
        size_(compute_size(shape_)),
        device_(device),
        device_index_(normalize_device_index(device_, device_index)),
        data_(allocate(size_, device_, device_index_)) {
    }

    Data(
        std::initializer_list<std::size_t> shape,
        DataDevice device = DataDevice::cpu,
        int device_index = 0) :
        Data(Shape(shape), device, device_index) {
    }

    ~Data() {
        release(data_, device_, device_index_);
    }

    Data(const Data &) = delete;
    Data &operator=(const Data &) = delete;

    Data(Data &&other) noexcept :
        shape_(std::move(other.shape_)),
        size_(std::exchange(other.size_, 0)),
        device_(other.device_),
        device_index_(other.device_index_),
        data_(std::exchange(other.data_, nullptr)) {
    }

    Data &operator=(Data &&other) noexcept {
        if (this != &other) {
            release(data_, device_, device_index_);
            shape_ = std::move(other.shape_);
            size_ = std::exchange(other.size_, 0);
            device_ = other.device_;
            device_index_ = other.device_index_;
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    T *data() {
        return data_;
    }

    const T *data() const {
        return data_;
    }

    const Shape &shape() const {
        return shape_;
    }

    std::size_t dim() const {
        return shape_.size();
    }

    std::size_t size() const {
        return size_;
    }

    std::size_t bytes() const {
        return size_ * sizeof(T);
    }

    DataDevice device() const {
        return device_;
    }

    int device_index() const {
        return device_index_;
    }

    void set_current_device() const {
        if (device_ == DataDevice::gpu) {
            set_gpu_device(device_index_);
        }
    }

    T &operator[](std::size_t index) {
        require_cpu();
        return data_[index];
    }

    const T &operator[](std::size_t index) const {
        require_cpu();
        return data_[index];
    }

    void fill(T value) {
        std::vector<T> values(size_, value);
        copy_from_cpu(values.data());
    }

    void random_uniform(
        float lower = 0.0F,
        float upper = 1.0F,
        std::uint64_t seed = 5489U) {
        if (!(lower < upper)) {
            throw std::invalid_argument(
                "Uniform lower bound must be less than upper bound");
        }
        std::mt19937_64 generator(seed);
        std::uniform_real_distribution<float> distribution(
            lower,
            upper);
        generate(generator, distribution);
    }

    void random_normal(
        float mean = 0.0F,
        float standard_deviation = 1.0F,
        std::uint64_t seed = 5489U) {
        if (!(standard_deviation > 0.0F)) {
            throw std::invalid_argument(
                "Normal standard deviation must be positive");
        }
        std::mt19937_64 generator(seed);
        std::normal_distribution<float> distribution(
            mean,
            standard_deviation);
        generate(generator, distribution);
    }

    static Data uniform(
        Shape shape,
        float lower = 0.0F,
        float upper = 1.0F,
        DataDevice device = DataDevice::cpu,
        std::uint64_t seed = 5489U,
        int device_index = 0) {
        Data result(std::move(shape), device, device_index);
        result.random_uniform(lower, upper, seed);
        return result;
    }

    static Data normal(
        Shape shape,
        float mean = 0.0F,
        float standard_deviation = 1.0F,
        DataDevice device = DataDevice::cpu,
        std::uint64_t seed = 5489U,
        int device_index = 0) {
        Data result(std::move(shape), device, device_index);
        result.random_normal(mean, standard_deviation, seed);
        return result;
    }

    Data copy_to(
        DataDevice target,
        int target_device_index = 0) const {
        Data result(shape_, target, target_device_index);
        copy(
            result.data_,
            target,
            result.device_index_,
            data_,
            device_,
            device_index_,
            bytes());
        return result;
    }

    void to(
        DataDevice target,
        int target_device_index = 0) {
        const auto normalized_target_index =
            normalize_device_index(target, target_device_index);
        if (target == device_ && normalized_target_index == device_index_) {
            return;
        }

        auto *new_data =
            allocate(size_, target, normalized_target_index);
        try {
            copy(
                new_data,
                target,
                normalized_target_index,
                data_,
                device_,
                device_index_,
                bytes());
        } catch (...) {
            release(new_data, target, normalized_target_index);
            throw;
        }

        release(data_, device_, device_index_);
        data_ = new_data;
        device_ = target;
        device_index_ = normalized_target_index;
        set_current_device();
    }

private:
    static std::size_t compute_size(const Shape &shape) {
        std::size_t size = 1;
        for (const auto extent : shape) {
            if (extent == 0) {
                throw std::invalid_argument(
                    "Data shape extents must be positive");
            }
            if (size > std::numeric_limits<std::size_t>::max() / extent) {
                throw std::overflow_error("Data shape is too large");
            }
            size *= extent;
        }
        if (size > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("Data allocation is too large");
        }
        return size;
    }

    static int normalize_device_index(
        DataDevice device,
        int device_index) {
        if (device == DataDevice::cpu) {
            return -1;
        }
        if (device_index < 0) {
            throw std::invalid_argument(
                "GPU device index must be non-negative");
        }
        return device_index;
    }

    static void set_gpu_device(int device_index) {
        check_gpu(
            gpuSetDevice(device_index),
            "gpuSetDevice");
    }

    static T *allocate(
        std::size_t size,
        DataDevice device,
        int device_index) {
        if (device == DataDevice::cpu) {
            return new T[size];
        }
        set_gpu_device(device_index);
        T *pointer = nullptr;
        check_gpu(
            gpuMalloc(&pointer, size * sizeof(T)),
            "gpuMalloc");
        return pointer;
    }

    static void release(
        T *pointer,
        DataDevice device,
        int device_index) noexcept {
        if (pointer == nullptr) {
            return;
        }
        if (device == DataDevice::cpu) {
            delete[] pointer;
            return;
        }
        (void)gpuSetDevice(device_index);
        (void)gpuFree(pointer);
    }

    static void copy(
        T *destination,
        DataDevice destination_device,
        int destination_device_index,
        const T *source,
        DataDevice source_device,
        int source_device_index,
        std::size_t bytes) {
        if (destination_device == DataDevice::cpu && source_device == DataDevice::cpu) {
            std::copy_n(source, bytes / sizeof(T), destination);
            return;
        }
        if (destination_device == DataDevice::gpu && source_device == DataDevice::cpu) {
            set_gpu_device(destination_device_index);
            check_gpu(
                gpuMemcpy(
                    destination,
                    source,
                    bytes,
                    gpuMemcpyHostToDevice),
                "gpuMemcpyHostToDevice");
            return;
        }
        if (destination_device == DataDevice::cpu && source_device == DataDevice::gpu) {
            set_gpu_device(source_device_index);
            check_gpu(
                gpuMemcpy(
                    destination,
                    source,
                    bytes,
                    gpuMemcpyDeviceToHost),
                "gpuMemcpyDeviceToHost");
            return;
        }
        set_gpu_device(destination_device_index);
        if (destination_device_index != source_device_index) {
            check_gpu(
                gpuMemcpyPeerAsync(
                    destination,
                    destination_device_index,
                    source,
                    source_device_index,
                    bytes,
                    nullptr),
                "gpuMemcpyPeerAsync");
            check_gpu(
                gpuDeviceSynchronize(),
                "gpuDeviceSynchronize");
            return;
        }
        check_gpu(
            gpuMemcpy(
                destination,
                source,
                bytes,
                gpuMemcpyDeviceToDevice),
            "gpuMemcpyDeviceToDevice");
    }

    void copy_from_cpu(const T *source) {
        copy(
            data_,
            device_,
            device_index_,
            source,
            DataDevice::cpu,
            -1,
            bytes());
    }

    template <typename Generator, typename Distribution>
    void generate(
        Generator &generator,
        Distribution &distribution) {
        std::vector<T> values(size_);
        for (auto &value : values) {
            value = static_cast<T>(distribution(generator));
        }
        copy_from_cpu(values.data());
    }

    void require_cpu() const {
        if (device_ != DataDevice::cpu) {
            throw std::logic_error(
                "CPU element access requires CPU Data");
        }
    }

    template <typename Error>
    static void check_gpu(Error error, const char *operation) {
        if (error != gpuSuccess) {
            throw std::runtime_error(
                std::string(operation) + " failed: " + gpuGetErrorString(error));
        }
    }

    Shape shape_;
    std::size_t size_;
    DataDevice device_;
    int device_index_;
    T *data_;
};

} // namespace peak_gemm
