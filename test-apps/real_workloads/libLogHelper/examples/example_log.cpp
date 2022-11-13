#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

#include "log_helper.hpp"

int main() {
    constexpr auto size = (size_t) 8192 << 8;
    // This variable is generally set to infinity iterations, for debug we set to 100 iterations
    constexpr auto iterations = 100lu;
    // Start the log with filename including "my_benchmark" to its name, and
    // header inside log file will print the detail "size:x repetition:y"
    log_helper::start_log_file("SampleQuicksortCXX",
                               "size:" + std::to_string(size) +
                               " repetition:" + std::to_string(iterations));

    // set the maximum number of errors allowed for each iteration,
    // default is 500
    log_helper::set_max_errors_iter(800);

    // set the interval of iteration to print details of current test, 
    // default is 1.
    // CAUTION: Avoid writing too much on disc/ethernet, this can break your test
    log_helper::set_iter_interval_print(5);

    std::cout << "Starting sample benchmark in C++, log file is " << log_helper::get_log_file_name() << "\n";
    std::random_device random_device;
    std::mt19937 random_engine(random_device());
    std::uniform_real_distribution<double> di(-1, 1);
    std::vector<double> gold_array(size), input_array(size);

    // This is just an example of input. In a real scenario, we should load the data from a  file,
    // then we calculate nothing on the Device Under Test but the evaluation kernel
    std::generate(gold_array.begin(),
                  gold_array.end(),
                  [&di, &random_engine] {
                      return di(random_engine);
                  });

    for (auto i = 0; i < iterations; i++) {
        //copy to the tmp vector
        input_array = gold_array;

        log_helper::start_iteration();
        // Execute the test (ONLY THE KERNEL), log functions will measure kernel time
        std::sort(input_array.begin(), input_array.end());
        log_helper::end_iteration();
        // insert a fake SDC to test the comparator
        if (i % 20 == 0) {
            input_array[333] = size;
        }

        int error_count = 0;
        int info_count = 0;

        // Testing with error_count > 0 for some iterations
        // You can call as many log_error_detail(str) as you need
        // However, it will log only the 500 errors or the
        // max_errors_iter set with set_max_errors_iter()
        for (int p = 0; p < size - 1; p++) {
            auto p0 = input_array[p];
            auto p1 = input_array[p + 1];
            if (p0 > p1) {
                auto error_detail = "position " + std::to_string(p) + " val " + std::to_string(p0) +
                                    " is higher than next position " + std::to_string(p1);
                std::cout << "Error at iteration " << i << " " << error_detail << std::endl;
                log_helper::log_error_detail(error_detail);
                error_count++;
            }
        }
        if (i == 16) {
            // only the value passed to set_max_infos_iter() will be logged (default is 500)
            log_helper::log_info_detail("info of event during iteration");
            info_count = info_count + 520;
        }

        // log how many errors the iteration had
        // if error_count is greater than 800, or the
        // max_errors_iter set with set_max_errors_iter()
        // it will terminate the execution of the benchmark
        log_helper::log_error_count(error_count);
        // the logging info function does not terminate the program
        log_helper::log_info_count(info_count);
    }
    // Finish the log file
    log_helper::end_log_file();
    std::cout << "Sample benchmark has finished!\n";
}
