// based on https://github.com/rigtorp/MPMCQueue

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef> // offsetof
#include <limits>
#include <memory>
#include <new> // std::hardware_destructive_interference_size
#include <stdexcept>

#ifndef __cpp_aligned_new
#ifdef _WIN32
#include <malloc.h> // _aligned_malloc
#else
#include <stdlib.h> // posix_memalign
#endif
#endif

namespace mpmc {

#if defined(__cpp_lib_hardware_interference_size) && !defined(__APPLE__)
static constexpr size_t hardware_interference_size =
    std::hardware_destructive_interference_size;
#else
static constexpr size_t hardware_interference_size = 64;
#endif

#if defined(__cpp_aligned_new)
template <typename T> using aligned_allocator = std::allocator<T>;
#else
template <typename T> struct aligned_allocator {
  using value_type = T;

  T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
#ifdef _WIN32
    auto *p = static_cast<T *>(_aligned_malloc(sizeof(T) * n, alignof(T)));
    if (p == nullptr) {
      throw std::bad_alloc();
    }
#else
    T *p;
    if (posix_memalign(reinterpret_cast<void **>(&p), alignof(T),
                       sizeof(T) * n) != 0) {
      throw std::bad_alloc();
    }
#endif
    return p;
  }

  void deallocate(T *p, std::size_t) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
  }
};
#endif

template <typename T> struct slot {
  ~slot() noexcept {
    if (turn & 1) {
      destroy();
    }
  }

  template <typename... Args> void construct(Args &&...args) noexcept {
    static_assert(std::is_nothrow_constructible<T, Args &&...>::value,
                  "T must be nothrow constructible with Args&&...");
    new (&storage) T(std::forward<Args>(args)...);
  }

  void destroy() noexcept {
    static_assert(std::is_nothrow_destructible<T>::value,
                  "T must be nothrow destructible");
    reinterpret_cast<T *>(&storage)->~T();
  }

  T &&move() noexcept { return reinterpret_cast<T &&>(storage); }

  // align to avoid false sharing between adjacent slots
  alignas(hardware_interference_size) std::atomic<size_t> turn = {0};
  typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
};

template <typename T, typename Allocator = aligned_allocator<slot<T>>>
class queue {
private:
  static_assert(std::is_nothrow_copy_assignable<T>::value ||
                    std::is_nothrow_move_assignable<T>::value,
                "T must be nothrow copy or move assignable");

  static_assert(std::is_nothrow_destructible<T>::value,
                "T must be nothrow destructible");

public:
  explicit queue(const size_t capacity,
                 const Allocator &alloc = Allocator())
      : capacity_(capacity), allocator_(alloc), head_(0), tail_(0) {
    if (capacity_ < 1) {
      throw std::invalid_argument("capacity < 1");
    }
    // allocate one extra slot to prevent false sharing on the last slot
    slots_ = allocator_.allocate(capacity_ + 1);
    // allocators are not required to honor alignment for over-aligned types
    // (see http://eel.is/c++draft/allocator.requirements#10) so we verify
    // alignment here
    if (reinterpret_cast<size_t>(slots_) % alignof(slot<T>) != 0) {
      allocator_.deallocate(slots_, capacity_ + 1);
      throw std::bad_alloc();
    }
    for (size_t i = 0; i < capacity_; ++i) {
      new (&slots_[i]) slot<T>();
    }
    static_assert(
        alignof(slot<T>) == hardware_interference_size,
        "slot must be aligned to cache line boundary to prevent false sharing");
    static_assert(sizeof(slot<T>) % hardware_interference_size == 0,
                  "slot size must be a multiple of cache line size to prevent "
                  "false sharing between adjacent slots");
    static_assert(sizeof(queue) % hardware_interference_size == 0,
                  "queue size must be a multiple of cache line size to "
                  "prevent false sharing between adjacent queues");
    static_assert(
        offsetof(queue, tail_) - offsetof(queue, head_) ==
            static_cast<std::ptrdiff_t>(hardware_interference_size),
        "head and tail must be a cache line apart to prevent false sharing");
  }

  ~queue() noexcept {
    for (size_t i = 0; i < capacity_; ++i) {
      slots_[i].~slot();
    }
    allocator_.deallocate(slots_, capacity_ + 1);
  }

  // non-copyable and non-movable
  queue(const queue &) = delete;
  queue &operator=(const queue &) = delete;

  template <typename... Args> void emplace(Args &&...args) noexcept {
    static_assert(std::is_nothrow_constructible<T, Args &&...>::value,
                  "T must be nothrow constructible with Args&&...");
    auto const head = head_.fetch_add(1);
    auto &slot = slots_[idx_(head)];
    while (turn_(head) * 2 != slot.turn.load(std::memory_order_acquire))
      ;
    slot.construct(std::forward<Args>(args)...);
    slot.turn.store(turn_(head) * 2 + 1, std::memory_order_release);
  }

  template <typename... Args> bool try_emplace(Args &&...args) noexcept {
    static_assert(std::is_nothrow_constructible<T, Args &&...>::value,
                  "T must be nothrow constructible with Args&&...");
    auto head = head_.load(std::memory_order_acquire);
    for (;;) {
      auto &slot = slots_[idx_(head)];
      if (turn_(head) * 2 == slot.turn.load(std::memory_order_acquire)) {
        if (head_.compare_exchange_strong(head, head + 1)) {
          slot.construct(std::forward<Args>(args)...);
          slot.turn.store(turn_(head) * 2 + 1, std::memory_order_release);
          return true;
        }
      } else {
        auto const prev_head = head;
        head = head_.load(std::memory_order_acquire);
        if (head == prev_head) {
          return false;
        }
      }
    }
  }

  void push(const T &v) noexcept {
    static_assert(std::is_nothrow_copy_constructible<T>::value,
                  "T must be nothrow copy constructible");
    emplace(v);
  }

  template <typename P,
            typename = typename std::enable_if<
                std::is_nothrow_constructible<T, P &&>::value>::type>
  void push(P &&v) noexcept {
    emplace(std::forward<P>(v));
  }

  bool try_push(const T &v) noexcept {
    static_assert(std::is_nothrow_copy_constructible<T>::value,
                  "T must be nothrow copy constructible");
    return try_emplace(v);
  }

  template <typename P,
            typename = typename std::enable_if<
                std::is_nothrow_constructible<T, P &&>::value>::type>
  bool try_push(P &&v) noexcept {
    return try_emplace(std::forward<P>(v));
  }

  void pop(T &v) noexcept {
    auto const tail = tail_.fetch_add(1);
    auto &slot = slots_[idx_(tail)];
    while (turn_(tail) * 2 + 1 != slot.turn.load(std::memory_order_acquire))
      ;
    v = slot.move();
    slot.destroy();
    slot.turn.store(turn_(tail) * 2 + 2, std::memory_order_release);
  }

  bool try_pop(T &v) noexcept {
    auto tail = tail_.load(std::memory_order_acquire);
    for (;;) {
      auto &slot = slots_[idx_(tail)];
      if (turn_(tail) * 2 + 1 == slot.turn.load(std::memory_order_acquire)) {
        if (tail_.compare_exchange_strong(tail, tail + 1)) {
          v = slot.move();
          slot.destroy();
          slot.turn.store(turn_(tail) * 2 + 2, std::memory_order_release);
          return true;
        }
      } else {
        auto const prev_tail = tail;
        tail = tail_.load(std::memory_order_acquire);
        if (tail == prev_tail) {
          return false;
        }
      }
    }
  }

  /// returns the number of elements in the queue.
  /// the size can be negative when the queue is empty and there is at least one
  /// reader waiting. since this is a concurrent queue the size is only a best
  /// effort guess until all reader and writer threads have been joined.
  ptrdiff_t size() const noexcept {
    // todo: how can we deal with wrapped queue on 32bit?
    return static_cast<ptrdiff_t>(head_.load(std::memory_order_relaxed) -
                                  tail_.load(std::memory_order_relaxed));
  }

  /// returns true if the queue is empty.
  /// since this is a concurrent queue this is only a best effort guess
  /// until all reader and writer threads have been joined.
  bool empty() const noexcept { return size() <= 0; }

private:
  constexpr size_t idx_(size_t i) const noexcept { return i % capacity_; }

  constexpr size_t turn_(size_t i) const noexcept { return i / capacity_; }

private:
  const size_t capacity_;
  slot<T> *slots_;
#if defined(__has_cpp_attribute) && __has_cpp_attribute(no_unique_address)
  Allocator allocator_ [[no_unique_address]];
#else
  Allocator allocator_;
#endif

  // align to avoid false sharing between head_ and tail_
  alignas(hardware_interference_size) std::atomic<size_t> head_;
  alignas(hardware_interference_size) std::atomic<size_t> tail_;
};

} // namespace mpmc
