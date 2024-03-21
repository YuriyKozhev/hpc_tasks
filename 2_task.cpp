#include <iostream>
#include <cassert>
#include <thread>
#include <mutex>
#include <numeric>
#include <random>
#include <algorithm>
#include <iterator>
#include <vector>
#include <execution>
#include <functional>
#include <queue>
#include <condition_variable>


// 2. Напишите с помощью потоков C++ программу, 
// сортирующую вектор из 100 мегабайт случайных целочисленных значений. 
// Внутри каждого потока можно использовать функцию sort() из стандартной библиотеки. 
// Сравните скорость со стандартной однопоточной функцией sort() 
// и многопоточной версией sort(std::execution::par, ...) из C++17.


std::vector<std::int32_t> create_random_vector() {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()}; 
    // 100 MB = 100 * 2**20 B = 25 * 2**20 * (4 B)
    const auto vec_size = 25 * 1024 * 1024;
    const auto min_number = 1;
    const auto max_number = 10000;
    std::uniform_int_distribution<std::int32_t> dist {min_number, max_number};
    auto gen = [&dist, &mersenne_engine](){
        return dist(mersenne_engine);
    };
    std::vector<std::int32_t> vec(vec_size);
    std::generate(vec.begin(), vec.end(), gen);
    return std::move(vec);
}


void print_vec(const std::vector<std::int32_t>& vec) {
    const auto visible_half_size = 10;
    if (vec.size() <= visible_half_size * 2) {
        for (auto el : vec)
            std::cout << el << ' ';
        std::cout << std::endl;
        return;
    }
    for (auto it = vec.begin(); it != vec.begin() + visible_half_size; it++)
        std::cout << *it << ' ';
    std::cout << "... ";
    for (auto it = vec.end() - visible_half_size; it != vec.end(); it++)
        std::cout << *it << ' ';
    std::cout << std::endl;
}


void test_sorting_algorithm(std::function<void(std::vector<std::int32_t>&)> sort_function) {
    auto vec = create_random_vector();
    print_vec(vec);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    sort_function(vec);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> timedelta = t2 - t1;
    std::cout << timedelta.count() / 1000 << " seconds" << std::endl;

    print_vec(vec);
    assert(std::is_sorted(vec.begin(), vec.end()));
}

struct Task {
    std::function<void()> func;
};


class MyThreadPool
{
public:
    MyThreadPool(const unsigned int N) {
        for (size_t i = 0; i < N; ++i) {
            _threads.push_back(std::thread([this] {
                bool is_running = 0;
                Task task;
                while (_is_alive) {
                    {
                        std::lock_guard<std::mutex> guard(_queue_mutex);
                        if (_tasks_queue.size() > 0) {
                            task = _tasks_queue.front();
                            _tasks_queue.pop();
                            is_running = 1;
                        }
                    }
                    if (is_running) {
                        is_running = 0;    
                        task.func();
                        {
                            std::lock_guard<std::mutex> guard(_tasks_mutex);
                            _tasks_left -= 1;
                        }
                        _cv_tasks.notify_one();
                    } else {
                        std::unique_lock lk(_queue_mutex);
                        _cv_queue.wait(lk, [this]{ return _tasks_queue.size() > 0 | _is_alive == 0; });
                    }
                }
            }));
        }
    }

    ~MyThreadPool() {
        _is_alive = 0;
        _cv_queue.notify_all();
        for (auto &th : _threads)
            th.join();
    }

    void queue_task(Task task) {
        {
            std::lock_guard<std::mutex> guard(_queue_mutex);
            _tasks_queue.push(task);
        }
        {
            std::lock_guard<std::mutex> guard(_tasks_mutex);
            _tasks_left += 1;
        }
        _cv_queue.notify_one();
    }

    void wait_all_tasks() {
        std::unique_lock lk(_tasks_mutex);
        _cv_tasks.wait(lk, [this]{ return _tasks_left == 0; });
    }

private:
    bool _is_alive = 1;
    std::vector<std::thread> _threads;

    std::queue<Task> _tasks_queue;
    std::mutex _queue_mutex;
    std::condition_variable _cv_queue;

    std::size_t _tasks_left = 0;
    std::mutex _tasks_mutex;
    std::condition_variable _cv_tasks;
};


void my_sort(std::vector<std::int32_t>& vec) {
    const auto vec_size = vec.size();
    const auto N = std::thread::hardware_concurrency();
    const std::size_t batch_size = vec_size / N;
    std::cout << "vec_size = " << vec_size << "; N = " << N << "; batch_size = " << batch_size << std::endl;

    auto thread_pool = MyThreadPool{N};

    // parallel sorting of non-overlapping intervals
    std::vector<size_t> points{0};
    for (size_t i = 0; i < N; ++i) {
        const std::size_t start = i * batch_size;
        const std::size_t finish = i == N - 1 ? vec_size : (i + 1) * batch_size;
        points.push_back(finish);
        auto task = Task{
            [&vec, start, finish]() {
                std::sort(vec.begin() + start, vec.begin() + finish);
            }
        };
        thread_pool.queue_task(task);
    }
    thread_pool.wait_all_tasks();

    // parallel merging of non-overlapping intervals
    // until vector is sorted from the first to the last element
    while (points.size() > 2) {
        for (auto it = points.begin(); it < points.end() - 2;) {
            const std::size_t start = *it;
            const std::size_t middle = *(it + 1);
            const std::size_t finish = *(it + 2);

            auto task = Task{
                [&vec, start, finish, middle]() {
                    std::inplace_merge(vec.begin() + start, vec.begin() + middle, vec.begin() + finish);
                }
            };
            thread_pool.queue_task(task);

            it = points.erase(it + 1);
        }
        thread_pool.wait_all_tasks();
    }
}


int main(int argc, char **argv) {
    std::cout << "Default std sort" << std::endl;
    test_sorting_algorithm([](std::vector<std::int32_t>& vec) -> void {
        std::sort(vec.begin(), vec.end());
    });
    std::cout << "Parallel std sort" << std::endl;
    test_sorting_algorithm([](std::vector<std::int32_t>& vec) -> void {
        std::sort(std::execution::par, vec.begin(), vec.end());
    });
    std::cout << "Custom sort" << std::endl;
    test_sorting_algorithm(my_sort);
    return 0;
}
