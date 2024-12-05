#ifndef __BITMAP_CUH
#define __BITMAP_CUH

#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

template <typename word_t, int sft> struct bits_t {
  int build(int nvertexs) {
    n_words = (nvertexs + 8 * sizeof(word_t) - 1) >> sft;
    H_ERR(cudaMalloc((void **)&dg_bits, sizeof(word_t) * n_words));
    reset();
    return sizeof(word_t) * n_words;
  }

  int get_size() { return n_words; }
  void reset() { CLEAN(dg_bits, n_words); }
  void mark_all(int x) {
    H_ERR(cudaMemset(dg_bits, x, sizeof(word_t) * n_words));
  }
  void mark_one(int root) {
    int loc = (root >> sft);
    word_t word = 1 << (root & MASK);
    excudaMemset<word_t>(dg_bits + loc, word, 1);
  }

  __device__ __tbdinline__ char *to_bytes() { return (char *)(void *)dg_bits; }

  __device__ __tbdinline__ int bytes_size() { return n_words << (sft - 3); }

  __device__ __tbdinline__ word_t load_word(int v) { return dg_bits[v >> sft]; }

  __device__ __tbdinline__ word_t *load_word_pos(int v) {
    return &dg_bits[v >> sft];
  }

  __device__ __tbdinline__ int *load_word_pos_as_int(int v) {
    int *ptr = (int *)dg_bits;
    return &ptr[v >> 5];
  }

  __device__ __tbdinline__ void store_word(int v, word_t word) {
    dg_bits[v >> sft] = word;
  }

  __device__ __tbdinline__ void store_word_as_int(int v, int word) {
    int *ptr = (int *)dg_bits;
    ptr[v >> 5] = word;
  }

  __device__ __tbdinline__ int loc(int v) { return (1 << (v & MASK)); }

  __device__ __tbdinline__ int loc_as_int(int v) { return (1 << (v & 31)); }

  // if bit[v]==0 return true else false;
  __device__ __tbdinline__ bool query(int v) {
    word_t mask_word = load_word(v);
    word_t mask_bit = loc(v);
    if (!(mask_bit & mask_word))
      return true;
    return false;
  }

  __device__ __tbdinline__ bool query_byte(int v) {
    char mask_word = to_bytes()[v >> 3];
    char mask_bit = 1 << (v & 7);
    if (!(mask_bit & mask_word))
      return true;
    return false;
  }

  // mark the bits of v
  // WARNING: no consistency guarantee, so hope for the best.
  __device__ __tbdinline__ void mark(int v) {
    word_t mask_word = load_word(v); // [A]
    word_t mask_bit = loc(v);
    if (!(mask_bit & mask_word)) {
      do {
        mask_word |= mask_bit;
        store_word(v, mask_word); // others may commit changes after [A]
        mask_word = load_word(v);
      } while (!(mask_bit & mask_word));
    }
  }

  // mark the bit of v, and return true if the bit==0
  // WARNING: no consistency guarantee, so hope for the best.
  __device__ __tbdinline__ bool query_and_mark(int v) {
    word_t mask_word = load_word(v); // [A]
    word_t mask_bit = loc(v);
    if (!(mask_bit & mask_word)) {
      do {
        mask_word |= mask_bit;
        store_word(v, mask_word); // others may commit changes after [A]
        mask_word = load_word(v);
      } while (!(mask_bit & mask_word));
      return true;
    }
    return false;
  }

  // mark the bit of v in atomic manner.
  // it guarantees the consistency, with higher overhead.
  __device__ __tbdinline__ bool query_and_mark_atomic(int v) {
    int mask_bit = loc_as_int(v);
    int x = atomicOr(load_word_pos_as_int(v), mask_bit);
    if (!(x & mask_bit))
      return true;
    return false;
  }

  const int MASK = (1 << sft) - 1;
  word_t *dg_bits;
  int n_words;
};

struct bitmap_t {
  int build(int nv) {
    int gpu_bytes = 0;
    gpu_bytes += visited.build(nv);
    gpu_bytes += active.build(nv);
    gpu_bytes += inactive.build(nv);
    return gpu_bytes;
  }

  void reset() {
    visited.reset();
    active.reset();
    inactive.reset();
  }

  void clean() { visited.reset(); }

  void init(int root) {
    if (root == -1) {
      active.mark_all(0xff);
      inactive.mark_all(0xff);
    } else if (root >= 0) {
      active.mark_one(root);
      inactive.mark_one(root);
    } else {
      active.mark_all(0);
      inactive.mark_all(0);
    }
  }

  __device__ bool is_active(int v) { return !active.query(v); }
  __device__ bool is_inactive(int v) { return inactive.query(v); }
  __device__ bool is_valid(int v) { return is_active(v) || is_inactive(v); }

  // return ture if the v is not visited
  __device__ bool mark_duplicate_lite(int v) {
    return visited.query_and_mark(v);
  }
  __device__ bool mark_duplicate_atomic(int v) {
    return visited.query_and_mark_atomic(v);
  }

  bits_t<char, 3> visited;  // use atomicOr
  bits_t<char, 3> active;   // 1 for active, 0 for others
  bits_t<char, 3> inactive; // 0 for inactive , 1 for others
  // bits_t<unsigned int, 5> visited;  // use atomicOr, must be 4-byte-long
  // bits_t<unsigned int, 5> active;   // 1 for active, 0 for others
  // bits_t<unsigned int, 5> inactive; // 0 for inactive , 1 for others
};
#endif
