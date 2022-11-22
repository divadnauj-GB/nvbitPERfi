#include <memory>
#include <string>
#include <iomanip>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <climits>

#include "common.hpp"
#include "file_writer.hpp"
#include "env_vars.h"

namespace log_helper {
    // does not change over the time
    // Default path to the config file
    std::unordered_map<std::string, std::string> configuration_parameters;

    // Max errors that can be found for a single iteration
    // If more than max errors is found, exit the program
    size_t max_errors_per_iter = 2048;
    size_t max_infos_per_iter = 2048;

    // Kernel total errors
    size_t kernels_total_errors = 0;
    // Used to print the log only for some iterations, equal 1 means print every iteration
    int32_t iter_interval_print = 1;

    // Saves the last amount of error found for a specific iteration
    size_t last_iter_errors = 0;
    // Saves the last iteration index that had an error
    size_t last_iter_with_errors = 0;

    size_t iteration_number = 0;
    double kernel_time_acc = 0;
    double kernel_time = 0;
    long long it_time_start;

    // Used to log max_error_per_iter details each iteration
    size_t log_error_detail_counter = 0;
    size_t log_info_detail_counter = 0;
    bool double_error_kill = true;

    // File writer
    std::shared_ptr<FileBase> file_writer_ptr = nullptr;

    // Default ECC status
    auto is_ecc_enabled = false;

//    auto float_output_format = std::scientific;
    auto float_output_format = std::fixed;

    /**
     * Read a configuration file from the default path
     */
    void read_configuration_file() {
        std::string config_file_path = CONFIG_FILE_PATH;
        std::ifstream config_file(config_file_path);
        // split string
        auto split = [](std::string &string_to_split) {
            std::vector<std::string> tokens;
            std::string token;
            std::istringstream token_stream(string_to_split);
            while (std::getline(token_stream, token, '=')) {
                tokens.push_back(token);
            }
            return tokens;
        };
        constexpr char whitespace[] = " \t\r\n\v\f";

        // trim leading white-spaces
        auto ltrim = [&whitespace](std::string &s) {
            size_t start_pos = s.find_first_not_of(whitespace);
            auto ret = s;
            if (std::string::npos != start_pos) {
                ret = ret.substr(start_pos);
            }
            return ret;
        };

        // trim trailing white-spaces
        auto rtrim = [&whitespace](std::string &s) {
            size_t end_pos = s.find_last_not_of(whitespace);
            auto ret = s;
            if (std::string::npos != end_pos) {
                ret = ret.substr(0, end_pos + 1);
            }
            return ret;
        };

        // Parse the lines of the configuration file and stores in the map
        if (config_file.good()) {
            for (std::string line; std::getline(config_file, line);) {
                if (!line.empty() && line[0] != '#' && line[0] != '[') {
                    line = ltrim(line);
                    line = rtrim(line);
                    auto split_line = split(line);
                    auto key = ltrim(split_line[0]);
                    key = rtrim(key);
                    auto value = ltrim(split_line[1]);
                    value = rtrim(value);
                    configuration_parameters[key] = value;
                    std::string msg_str = "CONFIG_PARAMETER-KEY:";
                    msg_str += key;
                    msg_str += " VALUE:" + value;
                    DEBUG_MESSAGE(msg_str);
                }
            }
        } else {
            EXCEPTION_MESSAGE("[Couldn't open " + std::string(CONFIG_FILE_PATH) + "]");
        }
    }

    bool check_ecc_status() {
        //check for enabled ECC
        auto key = std::string(ECC_INFO_KEY);
        std::ifstream ecc_info_file(configuration_parameters[key]);
        if (ecc_info_file.good()) {
            uint32_t ecc_status = 0;
            ecc_info_file >> ecc_status;
            ecc_info_file.close();
            return bool(ecc_status);
        }
        EXCEPTION_MESSAGE("ERROR ON READING THE ECC INFO FILE " + configuration_parameters[key]);

        return false;
    }


    void update_timestamp() {
        // We only update the timestamp if we are using the local
#if LOGGING_TYPE == LOCAL_ONLY
        auto signal_cmd = configuration_parameters[SIGNAL_CMD_KEY];
        __attribute__((unused)) auto sys_ret = system(signal_cmd.c_str());
//        if (sys_ret != 0) {
//            EXCEPTION_MESSAGE("ERROR ON SYSTEM CMD " + signal_cmd);
//        }

        auto timestamp_watchdog_path = configuration_parameters[VAR_DIR_KEY] + "/" + TIMESTAMP_FILE;
        std::ofstream timestamp_file(timestamp_watchdog_path);
        if (timestamp_file.good()) {
            auto now = std::chrono::system_clock::now();
            auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
            auto value = now_ms.time_since_epoch();
            timestamp_file << value.count();
            timestamp_file.close();
        }
#endif
    }

    template<int logging_t>
    std::shared_ptr<FileBase> make_file_writer(std::string &log_file_path) {
        THROW_EXCEPTION("INVALID LOG_HELPER CONFIGURATION, USE: ONLY_LOCAL=0, UDP_ONLY=1, or LOCAL_AND_UDP=2");
    }

    template<>
    std::shared_ptr<FileBase> make_file_writer<LOCAL_ONLY>(std::string &log_file_path) {
        return std::make_shared<LocalFile>(log_file_path);
    }

    template<>
    std::shared_ptr<FileBase> make_file_writer<UDP_ONLY>(std::string &log_file_path) {
        // Load server ip and port
        auto server_ip = configuration_parameters[SERVER_IP_KEY];
        auto server_port = std::stoi(configuration_parameters[SERVER_PORT_KEY]);
        return std::make_shared<UDPFile>(server_ip, server_port, is_ecc_enabled);
    }

    template<>
    std::shared_ptr<FileBase> make_file_writer<LOCAL_AND_UDP>(std::string &log_file_path) {
        // Load server ip and port
        auto server_ip = configuration_parameters[SERVER_IP_KEY];
        auto server_port = std::stoi(configuration_parameters[SERVER_PORT_KEY]);
        return std::make_shared<LocalAndUDPFile>(log_file_path, server_ip, server_port, is_ecc_enabled);
    }

    void check_file_writer() {
        if (file_writer_ptr == nullptr) {
            THROW_EXCEPTION("INVALID CONFIGURATION, FIRST CALL start_log_file before calling the other functions");
        }
    }

    int32_t start_log_file(std::string benchmark_name, std::string test_info) { // NOLINT(performance-unnecessary-value-param)
        // Necessary for all configurations (network or local)
        read_configuration_file();

        // Create the file name
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        // log example: 2021_11_15_22_08_25_cuda_trip_half_lava_ECC_OFF_fernando.log
        auto date_fmt = "%Y_%m_%d_%H_%M_%S";
        ss << std::put_time(std::localtime(&in_time_t), date_fmt);
        auto ecc_str = "OFF";
        is_ecc_enabled = check_ecc_status();
        if (is_ecc_enabled) {
            ecc_str = "ON";
        }
        char host[HOST_NAME_MAX] = "hostnameunknown";
        if (gethostname(host, HOST_NAME_MAX) != 0) {
            EXCEPTION_MESSAGE("[ERROR in gethostname(char *, int)] Could not access the host name");
        }

        auto log_file_path = configuration_parameters[LOG_DIR_KEY] + "/" +
                             ss.str() + "_" + benchmark_name + "_ECC_" + ecc_str + "_" + host + ".log";
        DEBUG_MESSAGE("Log file path " + log_file_path);
        file_writer_ptr = make_file_writer<LOGGING_TYPE>(log_file_path);

        bool file_creation_outcome = file_writer_ptr->write("#HEADER " + test_info + "\n");
        ss.str("");
        ss << std::put_time(std::localtime(&in_time_t), "#BEGIN Y:%Y M:%m D:%d Time:%H:%M:%S")
           << std::endl;
        file_creation_outcome &= file_writer_ptr->write(ss.str());

        update_timestamp();
        return !file_creation_outcome;
    }

    int32_t start_iteration() {
        update_timestamp();
        log_error_detail_counter = 0;
        log_info_detail_counter = 0;
        // Get current time with native precision
        auto now = std::chrono::system_clock::now();
        // Convert time_point to signed integral type
        it_time_start = now.time_since_epoch().count();
        return 0;
    }

    int32_t end_iteration() {
        update_timestamp();
        check_file_writer();
        //stackoverflow.com/questions/31255486/c-how-do-i-convert-a-stdchronotime-point-to-long-and-back
        std::chrono::system_clock::time_point start_it_tp{
                std::chrono::system_clock::duration{it_time_start}
        };
        std::chrono::duration<double> difference = std::chrono::system_clock::now() - start_it_tp;
        kernel_time = difference.count();
        kernel_time_acc += kernel_time;

        log_error_detail_counter = 0;
        log_info_detail_counter = 0;
        bool file_writing_outcome = false;
        if (iteration_number % iter_interval_print == 0) {
            std::ostringstream output;
            // Decimal places for the precision, 7 places
            output.precision(7);
            output << "#IT " << iteration_number
                   << float_output_format
                   << " KerTime:" << kernel_time
                   << " AccTime:" << kernel_time_acc
                   << std::endl;
            file_writing_outcome = file_writer_ptr->write(output.str());
        }
        iteration_number++;
        return !file_writing_outcome;
    }

    int32_t end_log_file() {
        check_file_writer();
        if (!file_writer_ptr->write("#END\n")) {
            EXCEPTION_MESSAGE("[ERROR in log_string(char *)] Unable to open file");
            return 1;
        }
        kernels_total_errors = 0;
        iteration_number = 0;
        kernel_time_acc = 0;
        return 0;
    }

    int32_t log_error_count(size_t kernel_errors) {
        update_timestamp();
        check_file_writer();

        if (kernel_errors > 0) {
            kernels_total_errors += kernel_errors;
            //"#SDC Ite:%lu KerTime:%f AccTime:%f KerErr:%lu AccErr:%lu\n",
            std::ostringstream output;
            // Decimal places for the precision, 7 places
            output.precision(7);

            output << "#SDC Ite:" << iteration_number
                   << float_output_format
                   << " KerTime:" << kernel_time
                   << " AccTime:" << kernel_time_acc
                   << " KerErr:" << kernel_errors
                   << " AccErr:" << kernels_total_errors
                   << std::endl;
            bool file_writing_outcome = file_writer_ptr->write(output.str());

            // "#ABORT amount of errors equals of the last iteration\n");
            if (kernel_errors == last_iter_errors &&
                (last_iter_with_errors + 1) == iteration_number && double_error_kill) {
                std::string abort_error = "#ABORT amount of errors equals of the last iteration";
                file_writer_ptr->write(abort_error + "\n");
                end_log_file();
                EXCEPTION_MESSAGE(abort_error);
            } else {
                // "#ABORT too many errors per iteration\n");
                if (kernel_errors > max_errors_per_iter) {
                    std::string abort_error = "#ABORT too many errors per iteration";
                    file_writer_ptr->write(abort_error + "\n");
                    end_log_file();
                    EXCEPTION_MESSAGE(abort_error);
                }
            }
            last_iter_errors = kernel_errors;
            last_iter_with_errors = iteration_number;
            return !file_writing_outcome;
        }
        return false;
    }

    int32_t log_info_count(size_t info_count) {
        update_timestamp();
        check_file_writer();

        //There is no limit to info that aborts the code
        //"#CINF Ite:%lu KerTime:%f AccTime:%f KerInfo:%lu AccInfo:%lu\n",
        if (info_count > 0) {
            //"#SDC Ite:%lu KerTime:%f AccTime:%f KerErr:%lu AccErr:%lu\n",
            std::ostringstream output;
            // Decimal places for the precision, 7 places
            output.precision(7);

            output << "#INF Ite:" << iteration_number
                   << float_output_format
                   << " KerTime:" << kernel_time
                   << " AccTime:" << kernel_time_acc
                   << " KerInf:" << info_count << std::endl;
            return !file_writer_ptr->write(output.str());
        }
        return 0;
    }

    int32_t log_error_detail(std::string error_detail) {
        check_file_writer();
        bool file_write_outcome = false;
        if (log_error_detail_counter <= max_errors_per_iter) {
            error_detail = "#ERR " + error_detail + "\n";
            file_write_outcome = file_writer_ptr->write(error_detail);
        }
        log_error_detail_counter++;
        return !file_write_outcome;
    }

    int32_t log_info_detail(std::string info_detail) {
        check_file_writer();
        bool file_write_outcome = false;
        if (log_info_detail_counter <= max_infos_per_iter) {
            info_detail = "#INF " + info_detail + "\n";
            file_write_outcome = file_writer_ptr->write(info_detail);
        }
        log_info_detail_counter++;
        return !file_write_outcome;
    }

    size_t set_max_errors_iter(size_t max_errors) {
        if (max_errors >= 1) {
            max_errors_per_iter = max_errors;
        }
        return max_errors_per_iter;
    }

    size_t set_max_infos_iter(size_t max_infos) {
        if (max_infos >= 1) {
            max_infos_per_iter = max_infos;
        }
        return max_infos_per_iter;
    }

    int32_t set_iter_interval_print(int32_t interval) {
        if (interval >= 1) {
            iter_interval_print = interval;
        }
        return iter_interval_print;
    }

    void disable_double_error_kill() {
        double_error_kill = false;
    }


    std::string get_log_file_name() {
        check_file_writer();
        return file_writer_ptr->get_file_path();
    }
} /*END NAMESPACE LOG_HELPER*/
