//
// Created by fernando on 07/03/2022.
//

#ifndef LIB_LOGHELPER_COMMON_HPP
#define LIB_LOGHELPER_COMMON_HPP

#include <string>

namespace log_helper {
    inline std::string _exception_info(const std::string &message, const std::string &file, int line) {
#ifdef DEBUG
        return message + " - FILE:" + file + ":" + std::to_string(line) + "\n";
#else
        return "";
#endif
    }

#define EXCEPTION_MESSAGE(message) std::cerr << _exception_info(message, __FILE__, __LINE__)
#define DEBUG_MESSAGE(message) std::cout << _exception_info(message, __FILE__, __LINE__)
#define THROW_EXCEPTION(message) std::throw_with_nested(std::runtime_error(_exception_info(message, __FILE__, __LINE__)))

    typedef enum {
        LOCAL_ONLY = 0,
        UDP_ONLY = 1,
        LOCAL_AND_UDP = 2
    } LoggingType;

#ifndef LOGGING_TYPE
#define LOGGING_TYPE LOCAL_ONLY
#endif
}
#endif //LIB_LOGHELPER_COMMON_HPP
