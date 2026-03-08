#include "ct/memory.cuh"

namespace ct {

// =================== MemoryPool Implementation =================

MemoryPool::~MemoryPool() {
    destroy();
}

void MemoryPool::create(size_t capacity_bytes, const char* name) {
    if (base_) destroy();

    name_ = name;
    capacity_ = capacity_bytes;
    offset_ = 0;
    peak_ = 0;

    CT_CHECK_CUDA(cudaMalloc(&base_, capacity_));

    fprintf(stderr, "[CT] %s pool: %.1f MB allocated\n",
            name_, capacity_ / (1024.0 * 1024.0));
}

void MemoryPool::destroy() {
    if (base_) {
        CT_CHECK_CUDA(cudaFree(base_));
        base_ = nullptr;
        capacity_ = 0;
        offset_ = 0;
        peak_ = 0;
    }
}

// ------------------- raw allocation -------------------

void* MemoryPool::alloc_bytes(size_t nbytes) {
    size_t aligned = align_up(nbytes, ALIGNMENT);
    if (offset_ + aligned > capacity_) {
        fprintf(stderr, "[CT] OOM in %s pool: need %zu bytes, have %zu free of %zu total\n",
                name_, aligned, capacity_ - offset_, capacity_);
        exit(EXIT_FAILURE);
    }

    void* ptr = static_cast<char*>(base_) + offset_;
    offset_ += aligned;

    if (offset_ > peak_) peak_ = offset_;

    return ptr;
}

// ------------------- tensor allocation -------------------

Tensor MemoryPool::alloc(const int* shape, int ndim, DType dtype) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    size_t nbytes = static_cast<size_t>(n) * dtype_size(dtype);

    void* ptr = alloc_bytes(nbytes);
    return make_tensor(ptr, shape, ndim, dtype);
}

Tensor MemoryPool::alloc(int d0, DType dtype) {
    int s[] = {d0};
    return alloc(s, 1, dtype);
}

Tensor MemoryPool::alloc(int d0, int d1, DType dtype) {
    int s[] = {d0, d1};
    return alloc(s, 2, dtype);
}

Tensor MemoryPool::alloc(int d0, int d1, int d2, DType dtype) {
    int s[] = {d0, d1, d2};
    return alloc(s, 3, dtype);
}

Tensor MemoryPool::alloc(int d0, int d1, int d2, int d3, DType dtype) {
    int s[] = {d0, d1, d2, d3};
    return alloc(s, 4, dtype);
}

// ------------------- zeroed allocation -------------------

Tensor MemoryPool::alloc_zeroed(const int* shape, int ndim, DType dtype) {
    Tensor t = alloc(shape, ndim, dtype);
    CT_CHECK_CUDA(cudaMemset(t.data, 0, t.bytes()));
    return t;
}

Tensor MemoryPool::alloc_zeroed(int d0, DType dtype) {
    int s[] = {d0};
    return alloc_zeroed(s, 1, dtype);
}

Tensor MemoryPool::alloc_zeroed(int d0, int d1, DType dtype) {
    int s[] = {d0, d1};
    return alloc_zeroed(s, 2, dtype);
}

Tensor MemoryPool::alloc_zeroed(int d0, int d1, int d2, DType dtype) {
    int s[] = {d0, d1, d2};
    return alloc_zeroed(s, 3, dtype);
}

Tensor MemoryPool::alloc_zeroed(int d0, int d1, int d2, int d3, DType dtype) {
    int s[] = {d0, d1, d2, d3};
    return alloc_zeroed(s, 4, dtype);
}

// ------------------- pool state -------------------

void MemoryPool::reset() {
    offset_ = 0;
}

size_t MemoryPool::save_mark() const {
    return offset_;
}

void MemoryPool::restore_mark(size_t mark) {
    if (mark > offset_) {
        fprintf(stderr, "[CT] %s pool: restore_mark(%zu) > current offset(%zu)\n",
                name_, mark, offset_);
        exit(EXIT_FAILURE);
    }
    offset_ = mark;
    // peak_ untouched — it tracks lifetime high-water mark
}

size_t MemoryPool::used() const     { return offset_; }
size_t MemoryPool::peak() const     { return peak_; }
size_t MemoryPool::capacity() const { return capacity_; }
const char* MemoryPool::name() const { return name_; }

} // namespace ct
