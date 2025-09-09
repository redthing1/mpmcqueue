#undef NDEBUG

#include <cassert>
#include <chrono>
#include <iostream>
#include <mpmc/mpmcqueue.hpp>
#include <set>
#include <thread>
#include <vector>

// test_type tracks correct usage of constructors and destructors
struct test_type {
  static std::set<const test_type *> constructed;
  test_type() noexcept {
    assert(constructed.count(this) == 0);
    constructed.insert(this);
  };
  test_type(const test_type &other) noexcept {
    assert(constructed.count(this) == 0);
    assert(constructed.count(&other) == 1);
    constructed.insert(this);
  };
  test_type(test_type &&other) noexcept {
    assert(constructed.count(this) == 0);
    assert(constructed.count(&other) == 1);
    constructed.insert(this);
  };
  test_type &operator=(const test_type &other) noexcept {
    assert(constructed.count(this) == 1);
    assert(constructed.count(&other) == 1);
    return *this;
  };
  test_type &operator=(test_type &&other) noexcept {
    assert(constructed.count(this) == 1);
    assert(constructed.count(&other) == 1);
    return *this;
  }
  ~test_type() noexcept {
    assert(constructed.count(this) == 1);
    constructed.erase(this);
  };
  // to verify that alignment and padding calculations are handled correctly
  char data[129];
};

std::set<const test_type *> test_type::constructed;

int main(int argc, char *argv[]) {
  (void)argc, (void)argv;

  {
    mpmc::queue<test_type> q(11);
    assert(q.size() == 0 && q.empty());
    for (int i = 0; i < 10; i++) {
      q.emplace();
    }
    assert(q.size() == 10 && !q.empty());
    assert(test_type::constructed.size() == 10);

    test_type t;
    q.pop(t);
    assert(q.size() == 9 && !q.empty());
    assert(test_type::constructed.size() == 10);

    q.pop(t);
    q.emplace();
    assert(q.size() == 9 && !q.empty());
    assert(test_type::constructed.size() == 10);
  }
  assert(test_type::constructed.size() == 0);

  {
    mpmc::queue<int> q(1);
    int t = 0;
    assert(q.try_push(1) == true);
    assert(q.size() == 1 && !q.empty());
    assert(q.try_push(2) == false);
    assert(q.size() == 1 && !q.empty());
    assert(q.try_pop(t) == true && t == 1);
    assert(q.size() == 0 && q.empty());
    assert(q.try_pop(t) == false && t == 1);
    assert(q.size() == 0 && q.empty());
  }

  // copyable only type
  {
    struct test {
      test() {}
      test(const test &) noexcept {}
      test &operator=(const test &) noexcept { return *this; }
      test(test &&) = delete;
    };
    mpmc::queue<test> q(16);
    // lvalue
    test v;
    q.emplace(v);
    q.try_emplace(v);
    q.push(v);
    q.try_push(v);
    // xvalue
    q.push(test());
    q.try_push(test());
  }

  // movable only type
  {
    mpmc::queue<std::unique_ptr<int>> q(16);
    // lvalue
    auto v = std::unique_ptr<int>(new int(1));
    // q.emplace(v);
    // q.try_emplace(v);
    // q.push(v);
    // q.try_push(v);
    // xvalue
    q.emplace(std::unique_ptr<int>(new int(1)));
    q.try_emplace(std::unique_ptr<int>(new int(1)));
    q.push(std::unique_ptr<int>(new int(1)));
    q.try_push(std::unique_ptr<int>(new int(1)));
  }

  {
    bool throws = false;
    try {
      mpmc::queue<int> q(0);
    } catch (std::exception &) {
      throws = true;
    }
    assert(throws == true);
  }

  // fuzz test
  {
    const uint64_t num_ops = 1000;
    const uint64_t num_threads = 10;
    mpmc::queue<uint64_t> q(num_threads);
    std::atomic<bool> flag(false);
    std::vector<std::thread> threads;
    std::atomic<uint64_t> sum(0);
    for (uint64_t i = 0; i < num_threads; ++i) {
      threads.push_back(std::thread([&, i] {
        while (!flag)
          ;
        for (auto j = i; j < num_ops; j += num_threads) {
          q.push(j);
        }
      }));
    }
    for (uint64_t i = 0; i < num_threads; ++i) {
      threads.push_back(std::thread([&, i] {
        while (!flag)
          ;
        uint64_t thread_sum = 0;
        for (auto j = i; j < num_ops; j += num_threads) {
          uint64_t v;
          q.pop(v);
          thread_sum += v;
        }
        sum += thread_sum;
      }));
    }
    flag = true;
    for (auto &thread : threads) {
      thread.join();
    }
    assert(sum == num_ops * (num_ops - 1) / 2);
  }

  return 0;
}
