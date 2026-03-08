#pragma once

#include "tensor.cuh"

namespace ct {

// =================== GPU Memory Pool =================
//
// Bump allocator over a single cudaMalloc. No fragmentation, no runtime alloc.
// Two instances in practice:
//   - persistent: weights, optimizer states — allocated once, never freed
//   - scratch: activations, intermediates — reset every training step

class MemoryPool {
public:
    MemoryPool() = default;
    ~MemoryPool();

    // ---- no copy, no move (owns a raw GPU allocation) ----
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // ------------------- lifecycle -------------------

    void create(size_t capacity_bytes, const char* name = "pool");
    void destroy();

    // ------------------- allocation -------------------

    /* bump-allocate and return a Tensor view into pool memory */
    Tensor alloc(const int* shape, int ndim, DType dtype);

    /* convenience overloads */
    Tensor alloc(int d0, DType dtype);
    Tensor alloc(int d0, int d1, DType dtype);
    Tensor alloc(int d0, int d1, int d2, DType dtype);
    Tensor alloc(int d0, int d1, int d2, int d3, DType dtype);

    /* bump-allocate + zero (for gradient accumulators, atomicAdd targets) */
    Tensor alloc_zeroed(const int* shape, int ndim, DType dtype);
    Tensor alloc_zeroed(int d0, DType dtype);
    Tensor alloc_zeroed(int d0, int d1, DType dtype);
    Tensor alloc_zeroed(int d0, int d1, int d2, DType dtype);
    Tensor alloc_zeroed(int d0, int d1, int d2, int d3, DType dtype);

    /* raw bytes (for non-tensor scratch space) */
    void* alloc_bytes(size_t nbytes);

    // ------------------- control -------------------

    void reset();                  // rewind offset to 0 — for scratch pool between steps
    size_t save_mark() const;      // snapshot current offset for nested scoping
    void restore_mark(size_t mark); // rewind to a saved mark (must be <= current offset)

    size_t used() const;
    size_t peak() const;
    size_t capacity() const;
    const char* name() const;

private:
    void*  base_     = nullptr;
    size_t capacity_ = 0;
    size_t offset_   = 0;
    size_t peak_     = 0;
    const char* name_ = "pool";
};

} // namespace ct
