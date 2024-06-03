#include <mpi.h>
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
#include <stdio.h>

// 1. Напишите программу, 
// которая сортирует массив целочисленных случайных значений 
// используя распараллеливание на несколько процессов с помощью MPI 
// и стандартный std::sort внутри каждого из процессов.


std::vector<std::int32_t> create_random_vector(size_t vec_size) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()}; 
    // 100 MB = 100 * 2**20 B = 25 * 2**20 * (4 B)
    // const auto vec_size = 25 * 1024 * 1024;
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


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Status status;
    int proc_id, num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    const auto vec_size = 25 * 1024 * 1024;
    int chunk_size = vec_size / num_procs;
    
    if (proc_id == 0) {
        std::cout << "vec_size = " << vec_size << "; num_procs = " << num_procs
             << "; chunk_size = " << chunk_size << std::endl;
        auto vec = create_random_vector(vec_size);
        print_vec(vec); 

        double starttime = MPI_Wtime();
        for (int slave_id = 1; slave_id < num_procs; slave_id++)
            MPI_Send(&vec[(slave_id - 1) * chunk_size], chunk_size, MPI_INT, slave_id, 0, MPI_COMM_WORLD);
        std::sort(vec.begin() + (num_procs - 1) * chunk_size, vec.end());
        for (int slave_id = 1; slave_id < num_procs; slave_id++)
            MPI_Recv(&vec[(slave_id - 1) * chunk_size], chunk_size, MPI_INT, slave_id, 0, MPI_COMM_WORLD, &status);
        for (int slave_id = 1; slave_id < num_procs; slave_id++) {
            auto middle = slave_id * chunk_size;
            auto finish = slave_id == num_procs - 1 ? vec_size : middle + chunk_size;
            std::inplace_merge(vec.begin(), vec.begin() + middle, vec.begin() + finish);
        }
        double endtime = MPI_Wtime();

        std::cout << endtime - starttime << " seconds" << std::endl;
        print_vec(vec);
        assert(std::is_sorted(vec.begin(), vec.end()));
    }
    else { 
        std::vector<int> data (chunk_size, 0);
        MPI_Recv(&data[0], chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std::sort(data.begin(), data.end());
        MPI_Send(&data[0], chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
