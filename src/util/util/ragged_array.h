#ifndef RAGGED_ARRAY_H
#define RAGGED_ARRAY_H

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

namespace Ibis {

template <typename DataType, class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
class RaggedArray {
public:
    using ArrayType = Kokkos::View<DataType *, Layout, Space>;
    using host_layout = Layout;
    using host_space = Kokkos::DefaultHostExecutionSpace;
    using HostMirror = RaggedArray<DataType, host_layout, host_space>;

private:
    using OffsetType = Kokkos::View<size_t *, Layout, Space>;

    // allow accessing the private data of the same class with differe
    // template parameters (e.g. living in different memory spaces)
    template <typename OtherDataType, class OtherLayout, class OtherSpace>
    friend class RaggedArray;

public:
    RaggedArray() {}

    RaggedArray(size_t num_values, size_t num_rows)
        : data_("RaggedArray::data", num_values),
          offsets_("RaggedArray::offsets", num_rows + 1) {}

    RaggedArray(ArrayType data, OffsetType offsets) : data_(data), offsets_(offsets) {}

    RaggedArray(std::vector<std::vector<DataType>> data) {
        // count the total number of entries
        size_t n = 0;
        for (const auto &row : data) {
            n += row.size();
        }

        // allocate memory
        data_ = ArrayType("RaggedArray::data", n);
        offsets_ = OffsetType("RaggedArray::offsets", data.size() + 1);

        // initialise memory (on the CPU)
        auto data_host = Kokkos::create_mirror_view(data_);
        auto offsets_host = Kokkos::create_mirror_view(offsets_);
        size_t i = 0;
        size_t row_i = 0;
        for (const auto &row : data) {
            offsets_host(row_i) = i;
            for (const auto &value : row) {
                data_host(i) = value;
                i++;
            }
            row_i++;
        }
        offsets_host(row_i) = i;

        // Copy data from CPU to the GPU (if needed)
        Kokkos::deep_copy(data_, data_host);
        Kokkos::deep_copy(offsets_, offsets_host);
    }

    // Read-write value at (row, col)
    KOKKOS_INLINE_FUNCTION
    DataType &operator()(const size_t row, const size_t col) const {
        size_t index = offsets_(row) + col;
        return data_(index);
    }

    // Read-only value at (row, col)
    // KOKKOS_INLINE_FUNCTION
    // const DataType &operator()(const int row, const int col) const {
    //     int index = offsets_(row) + col;
    //     return data_(index);
    // }

    // Read-write view to a particular row
    KOKKOS_INLINE_FUNCTION
    auto operator()(const size_t row) {
        size_t first = offsets_(row);
        size_t last = offsets_(row + 1);
        return Kokkos::subview(data_, Kokkos::make_pair(first, last));
    }

    // Read-only view to a particular row
    KOKKOS_INLINE_FUNCTION
    auto operator()(const size_t row) const {
        size_t first = offsets_(row);
        size_t last = offsets_(row + 1);
        return Kokkos::subview(data_, Kokkos::make_pair(first, last));
    }

    // Create a host mirror
    HostMirror host_mirror() const {
        auto data_mirror = Kokkos::create_mirror_view(data_);
        auto offsets_mirror = Kokkos::create_mirror_view(offsets_);
        return HostMirror(data_mirror, offsets_mirror);
    }

    // Create a host mirror and copy contents to the mirror
    HostMirror host_mirror_and_copy() {
        auto data_mirror = Kokkos::create_mirror_view(data_);
        auto offsets_mirror = Kokkos::create_mirror_view(offsets_);
        HostMirror host_mirror(data_mirror, offsets_mirror);
        host_mirror.deep_copy(*this);
        return host_mirror;
    }

    // Copy data from another ragged array (with the same dimensions)
    template <class OtherSpace>
    void deep_copy(const RaggedArray<DataType, Layout, OtherSpace> &other) {
        Kokkos::deep_copy(data_, other.data_);
        Kokkos::deep_copy(offsets_, other.offsets_);
    }

    KOKKOS_INLINE_FUNCTION
    size_t num_rows() const { return offsets_.extent(0) - 1; }

    KOKKOS_INLINE_FUNCTION
    size_t num_values() const { return data_.extent(0); }

    const OffsetType &offsets() const { return offsets_; }

    const ArrayType &data() const { return data_; }

    bool operator==(const RaggedArray<DataType, Layout, Space> &other) const {
        for (size_t i = 0; i < data_.extent(0); i++) {
            if (data_(i) != other.data_(i)) return false;
        }

        for (size_t i = 0; i < offsets_.extent(0); i++) {
            if (offsets_(i) != other.offsets_(i)) return false;
        }
        return true;
    }

    std::string to_string() const {
        std::string result = "RaggedArray(offsets(";
        result.append(std::to_string(offsets_.extent(0)));
        result.append(") = [");
        for (size_t i = 0; i < offsets_.extent(0); i++) {
            result.append(std::to_string(offsets_(i)));
            result.append(", ");
        }
        result.append("], data(");
        result.append(std::to_string(data_.extent(0)));
        result.append(") = [");
        for (size_t i = 0; i < data_.extent(0); i++) {
            result.append(std::to_string(data_(i)));
            result.append(", ");
        }
        result.append("], ragged array = ");

        result.append("[");
        for (size_t row = 0; row < num_rows(); row++) {
            size_t first = offsets_(row);
            size_t last = offsets_(row + 1);
            auto row_data = Kokkos::subview(data_, Kokkos::make_pair(first, last));
            result.append("[");
            // int size = last - first;
            for (size_t col = 0; col < row_data.size(); col++) {
                result.append(std::to_string(row_data(col)));
                result.append(", ");
            }
            result.append("], ");
        }
        result.append("])");
        return result;
    }

private:
    ArrayType data_;
    OffsetType offsets_;
};

}  // namespace Ibis

#endif
