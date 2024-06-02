#include <iostream>
#include <cassert>
#include <thread>
#include <mutex>
#include <numeric>
#include <random>
#include <algorithm>
#include <iterator>
#include <vector>
#include <functional>
#include <queue>
#include <condition_variable>
#include <complex>
// simplified version of a BPM lib (https://github.com/sol-prog/cpp-bmp-images/blob/master/BMP.h)
#include "BMP.h"

// 4. Напишите реализацию фрактала "Бассейны Ньютона". 
// Для распараллеливания необходимо использовать пул потоков C++ 
// по количеству аппаратных потоков процессора. 
// Основной поток заполняет очередь с заданиями для пула потоков. 
// Обратите внимание на возможность математических оптимизаций.
// Максимальный балл — 15.


int newton(double real, double imag, int iters_limit, 
    std::function<std::complex<double>(std::complex<double>)> newtons_iter, 
    std::vector<std::complex<double>> polynomial_roots
) {
	auto z = real + imag * std::literals::complex_literals::operator""i(1.0L);
	for (int iter = 0; iter < iters_limit; iter++)
		z = newtons_iter(z);
    
    double min_dist = std::numeric_limits<double>::max();
    size_t min_idx = 0;
    size_t idx = 0;
    for (auto root : polynomial_roots) {
        double dist = std::abs(z - root);
        if (dist <= min_dist) {
            min_dist = dist;
            min_idx = idx;
        }
        idx++;
    }
	return min_idx;
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


void parallel_fill(
    BMP& bmp, double x_min, double y_min, double dx, double dy, 
    size_t iters_limit, size_t width, size_t height, int batch_size,
    std::function<std::complex<double>(std::complex<double>)> newtons_iter, 
    std::vector<std::complex<double>> polynomial_roots,
    std::vector<std::tuple<int, int, int>> colors_vec
) {
    const auto N = std::thread::hardware_concurrency();
    auto thread_pool = MyThreadPool{N};
	for (size_t w = 0; w < width; w++) {
	    for (size_t batch = 0; batch <= height / batch_size; batch++) {
            auto task = Task{[&bmp, x_min, y_min, dx, dy, w, batch, iters_limit, batch_size, height,
                &newtons_iter, &polynomial_roots, &colors_vec
            ] () {
                for (size_t hh = batch * batch_size; hh < (batch + 1) * batch_size & hh < height; hh++) {
                    double x = x_min + w * dx;
                    double y = y_min + hh * dy;
                    int idx = newton(x, y, iters_limit, newtons_iter, polynomial_roots);
                    auto colors = colors_vec[idx];
                    bmp.set_pixel(w, hh, std::get<0>(colors), std::get<1>(colors), std::get<2>(colors));
                }
            }};
            thread_pool.queue_task(task);
		}
	}
    thread_pool.wait_all_tasks();
}


void sequential_fill(
    BMP& bmp, double x_min, double y_min, double dx, double dy, 
    size_t iters_limit, size_t width, size_t height, int batch_size,
    std::function<std::complex<double>(std::complex<double>)> newtons_iter, 
    std::vector<std::complex<double>> polynomial_roots,
    std::vector<std::tuple<int, int, int>> colors_vec
) {
    assert(batch_size == -1);
	for (size_t w = 0; w < width; w++) {
	    for (size_t h = 0; h < height; h++) {
            double x = x_min + w * dx;
            double y = y_min + h * dy;
            int idx = newton(x, y, iters_limit, newtons_iter, polynomial_roots);
            auto colors = colors_vec[idx];
            bmp.set_pixel(w, h, std::get<0>(colors), std::get<1>(colors), std::get<2>(colors));
		}
	}
}


void openmp_fill(
    BMP& bmp, double x_min, double y_min, double dx, double dy, 
    size_t iters_limit, size_t width, size_t height, int batch_size,
    std::function<std::complex<double>(std::complex<double>)> newtons_iter, 
    std::vector<std::complex<double>> polynomial_roots,
    std::vector<std::tuple<int, int, int>> colors_vec
) {
    assert(batch_size == -1);
    #pragma omp parallel for collapse(2) schedule(guided)
	for (size_t w = 0; w < width; w++) {
	    for (size_t h = 0; h < height; h++) {
            double x = x_min + w * dx;
            double y = y_min + h * dy;
            int idx = newton(x, y, iters_limit, newtons_iter, polynomial_roots);
            auto colors = colors_vec[idx];
            bmp.set_pixel(w, h, std::get<0>(colors), std::get<1>(colors), std::get<2>(colors));
		}
	}
}


using fill_function_type = void(
    BMP&, double, double, double, double, size_t, size_t, size_t, int,
    std::function<std::complex<double>(std::complex<double>)>,
    std::vector<std::complex<double>>,
    std::vector<std::tuple<int, int, int>>
);

void test_filling_algorithm(
    std::function<fill_function_type> fill_function,
    size_t width, size_t height, double x_min, double y_min, double dx, double dy, size_t iters_limit,
    const char* output_filename, int batch_size,
    std::function<std::complex<double>(std::complex<double>)> newtons_iter, 
    std::vector<std::complex<double>> polynomial_roots,
    std::vector<std::tuple<int, int, int>> colors_vec
) {
    BMP bmp(width, height);
    auto t1 = std::chrono::high_resolution_clock::now();
    fill_function(
        bmp, x_min, y_min, dx, dy, iters_limit,  width, height, batch_size,
        newtons_iter, polynomial_roots, colors_vec
    );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> timedelta = t2 - t1;
    std::cout << timedelta.count() / 1000 << " seconds" << std::endl;    
    bmp.write(output_filename);
}


int main(int argc, char **argv) {
    double scale = 1;
	size_t width = (size_t) (1280 * scale);
	size_t height = (size_t) (1024 * scale);

    const double x_min = -1;
	const double x_max = 1;
	const double y_min = -1;
	const double y_max = 1;

    double dx = (x_max - x_min) / (width - 1);
	double dy = (y_max - y_min) / (height - 1);

    size_t iters_limit = 100;
    
    // derivate for f(z) = z*^3 - 1
    auto newtons_iter = [] (std::complex <double> z) -> std::complex<double> {
        return z - (z * z * z - 1.0) / (3.0 * z * z); 
    };

    // roots for f(z) = z*^3 - 1
    std::vector<std::complex<double>> polynomial_roots { 
        -1.0 / 2.0 - std::sqrt(3) / 2.0 * std::literals::complex_literals::operator""i(1.0L) , 
        -1.0 / 2.0 + std::sqrt(3) / 2.0 * std::literals::complex_literals::operator""i(1.0L), 
        1.0
    };

    // corresponding colors for roots
    std::vector<std::tuple<int, int, int>> colors_vec {
        std::tuple<int, int, int>(255, 0, 0),
        std::tuple<int, int, int>(0, 255, 0),
        std::tuple<int, int, int>(0, 0, 255)
    };

    std::cout << "Sequential fill" << std::endl;
    test_filling_algorithm(sequential_fill, width, height, x_min, y_min, dx, dy, 
        iters_limit, "newton_seq.bmp", -1, newtons_iter, polynomial_roots, colors_vec);
    for (int batch_size = 10; batch_size <= 200; batch_size *= 2) {
        std::cout << "Parallel fill, batch = " << batch_size << std::endl;
        test_filling_algorithm(
            parallel_fill, width, height, x_min, y_min, dx, dy, iters_limit, 
            "newton_par.bmp", batch_size, newtons_iter, polynomial_roots, colors_vec);
    }
    std::cout << "OpenMP fill" << std::endl;
    test_filling_algorithm(openmp_fill, width, height, x_min, y_min, dx, dy, 
        iters_limit, "newton_openmp.bmp", -1, newtons_iter, polynomial_roots, colors_vec);
    return 0;
}
