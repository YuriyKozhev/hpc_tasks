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
#include <omp.h>
// simplified version of a BPM lib (https://github.com/sol-prog/cpp-bmp-images/blob/master/BMP.h)
#include "BMP.h"

// 3. Напишите реализацию фрактала Мандельброта. 
// Для распараллеливания необходимо использовать пул потоков C++ 
// по количеству аппаратных потоков процессора. 
// Основной поток заполняет очередь с заданиями для пула потоков.

// 2. Напишите реализацию фрактала Мандельброта, ускоренную с помощью OpenMP.


int mandelbrot(double c_real, double c_imag, int iters_limit) {
	double z_real = c_real;
	double z_imag = c_imag;
	for (int iter = 0; iter < iters_limit; iter++) {
		double r2 = z_real * z_real;
		double i2 = z_imag * z_imag;
		
		if (r2 + i2 > 4.0) 
            return iter;

		z_imag = 2.0 * z_real * z_imag + c_imag;
		z_real = r2 - i2 + c_real;
	}
	return iters_limit;
}


// use smooth polynomials to make beautiful color gradient
std::tuple<int, int, int> get_bgr(int iter, int iter_max) {
	double t = (double) iter / (double) iter_max;
	int r = (int) (9 * (1 - t) * t * t * t * 255);
	int g = (int) (15 * (1 - t) * (1 - t) * t * t * 255);
	int b =  (int) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);	
	return std::tuple<int, int, int>(b, g, r);
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
    size_t iters_limit, size_t width, size_t height, int batch_size 
) {
    const auto N = std::thread::hardware_concurrency();
    auto thread_pool = MyThreadPool{N};
	for (size_t w = 0; w < width; w++) {
	    for (size_t batch = 0; batch <= height / batch_size; batch++) {
            auto task = Task{[&bmp, x_min, y_min, dx, dy, w, batch, iters_limit, batch_size, height] () {
                for (size_t hh = batch * batch_size; hh < (batch + 1) * batch_size & hh < height; hh++) {
                    double x = x_min + w * dx;
                    double y = y_min + hh * dy;
                    int value = mandelbrot(x, y, iters_limit);
                    auto colors = get_bgr(value, iters_limit);
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
    size_t iters_limit, size_t width, size_t height, int batch_size
) {
    assert(batch_size == -1);
	for (size_t w = 0; w < width; w++) {
	    for (size_t h = 0; h < height; h++) {
            double x = x_min + w * dx;
            double y = y_min + h * dy;
            int value = mandelbrot(x, y, iters_limit);
            auto colors = get_bgr(value, iters_limit);
            bmp.set_pixel(w, h, std::get<0>(colors), std::get<1>(colors), std::get<2>(colors));
		}
	}
}


void openmp_fill(
    BMP& bmp, double x_min, double y_min, double dx, double dy, 
    size_t iters_limit, size_t width, size_t height, int batch_size
) {
    #pragma omp parallel for collapse(2) schedule(dynamic,batch_size)
    for (size_t w = 0; w < width; w++) {
	    for (size_t h = 0; h < height; h++) {
            double x = x_min + w * dx;
            double y = y_min + h * dy;
            int value = mandelbrot(x, y, iters_limit);
            auto colors = get_bgr(value, iters_limit);
            bmp.set_pixel(w, h, std::get<0>(colors), std::get<1>(colors), std::get<2>(colors));
		}
	}
}

void test_filling_algorithm(
    std::function<void(BMP&, double, double, double, double, size_t, size_t, size_t, int)> fill_function,
    size_t width, size_t height, double x_min, double y_min, double dx, double dy, size_t iters_limit,
    const char* output_filename, int batch_size
) {
    BMP bmp(width, height);
    auto t1 = std::chrono::high_resolution_clock::now();
    fill_function(
        bmp, x_min, y_min, dx, dy, iters_limit,  width, height, batch_size
    );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> timedelta = t2 - t1;
    std::cout << timedelta.count() / 1000 << " seconds" << std::endl;    
    bmp.write(output_filename);
}


int main(int argc, char **argv) {
    double scale = 4;
	size_t width = (size_t) (1280 * scale);
	size_t height = (size_t) (1024 * scale);

    double padding_size = 0.1;
    const double x_min = -2 - padding_size;
	const double x_max = 0.47 + padding_size;
	const double y_min = -1.12 - padding_size;
	const double y_max = 1.12 + padding_size;

    double dx = (x_max - x_min) / (width - 1);
	double dy = (y_max - y_min) / (height - 1);

    size_t iters_limit = 500;

    std::cout << "Sequential fill" << std::endl;
    test_filling_algorithm(sequential_fill, width, height, x_min, y_min, dx, dy, 
        iters_limit, "seq.bmp", -1);
    
    for (int batch_size = 10; batch_size <= 200; batch_size *= 2) {
        std::cout << "Parallel fill, batch = " << batch_size << std::endl;
        test_filling_algorithm(
            parallel_fill, width, height, x_min, y_min, dx, dy, iters_limit, 
            "par.bmp", batch_size);
    }
    
    for (int batch_size = 10; batch_size <= 200; batch_size *= 2) {
        std::cout << "OpenMP fill, batch = " << batch_size << std::endl;
        test_filling_algorithm(
            openmp_fill, width, height, x_min, y_min, dx, dy, iters_limit, 
            "openmp.bmp", batch_size);
    }

    return 0;
}
