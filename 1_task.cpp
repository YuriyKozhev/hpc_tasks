#include <iostream>
#include <cassert>
#include <vector>
#include <map>
#include <thread>
#include <mutex>

// 1. Напишите программу, которая считает и выводит в консоль гистограмму значений байтов 
// заданного входным аргументом файла. 
// Программа должна запустить несколько C++ потоков, 
// которые параллельно считывают данные из разных мест файла и объединяют свои результаты.

void calc_hist(std::map<std::uint8_t, int>& hist, const char* filename, const std::size_t start, const std::size_t finish) {
    std::FILE* fp = std::fopen(filename, "r");
    assert(fp);
    std::vector<std::uint8_t> buffer(finish - start);
    std::fseek(fp, start, SEEK_SET);
    std::fread(buffer.data(), sizeof(std::uint8_t), buffer.size(), fp);
    std::fclose(fp);
    for (const auto& ch : buffer)
        hist[ch] += 1;
}


int main(int argc, char **argv) {
    char *filename = argv[1];
    std::FILE* fp = std::fopen(filename, "r");
    assert(fp);
    std::fseek(fp, 0, SEEK_END);
    const std::size_t filesize = std::ftell(fp);
    std::fclose(fp);

    const auto N = std::thread::hardware_concurrency();
    const std::size_t batch_size = filesize / N;
    std::cout << "filesize = " << filesize << "; N = " << N << "; batch_size = " << batch_size << std::endl;

    std::map<std::uint8_t, int> histogram{};
    std::mutex mutex;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < N; ++i) {
        const std::size_t start = i * batch_size;
        const std::size_t finish = i == N - 1 ? filesize : (i + 1) * batch_size;
        threads.push_back(std::thread([&histogram, &mutex, &filename, start, finish] {
            std::map<std::uint8_t, int> hist{};
            calc_hist(hist, filename, start, finish);
            {
                std::lock_guard<std::mutex> guard(mutex);
                std::cout << start << ' ' << finish << std::endl;
                for (const auto& [key, value] : hist)
                    histogram[key] += value;
            }
        }));
    }

    for (auto &thread : threads)
        thread.join();

    for (const auto& [key, value] : histogram)
        if (key == '\n')
            std::cout << '[' << "\\n" << "] = " << value << "; ";
        else
            std::cout << '[' << key << "] = " << value << "; ";
    std::cout << std::endl;
    
    return 0;
}
