//
// Created by fernando on 14/06/2021.
//

#ifndef LOG_HELPER_BASE_HPP
#define LOG_HELPER_BASE_HPP

#include <string>

namespace log_helper {
    int32_t start_log_file(std::string benchmark_name, std::string test_info);

    int32_t start_iteration();

    int32_t end_iteration();

    int32_t end_log_file();

    int32_t log_error_count(size_t kernel_errors);

    int32_t log_info_count(size_t info_count);

    int32_t log_error_detail(std::string error_detail);

    int32_t log_info_detail(std::string info_detail);

    size_t set_max_errors_iter(size_t max_errors);

    size_t set_max_infos_iter(size_t max_infos);

    int32_t set_iter_interval_print(int32_t interval);

    void disable_double_error_kill();

    void update_timestamp();

    std::string get_log_file_name();
} /*END NAMESPACE LOG_HELPER*/

#endif //LOG_HELPER_BASE_HPP
