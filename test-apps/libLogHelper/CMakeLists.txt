cmake_minimum_required(VERSION 3.16)

# Common CMake settings
project(LogHelper LANGUAGES C CXX VERSION 1.1.1 DESCRIPTION "Log Helper for radiation experiments")

########################################################################################################################
#In many cases you will want to make sure that the required libraries are always found independent
#from LD_LIBRARY_PATH and the install location. use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if ("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif ("${isSystemDir}" STREQUAL "-1")
########################################################################################################################

#set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# Selects between C++ and C backend
option(WITH_CXX_BACKEND "To use C++ over C backend" ON)
if (NOT WITH_CXX_BACKEND)
    message(WARNING "Option for C backend was disabled on libLogHelper 1.1, using C++ backend")
endif ()
# Define debug feature
option(WITH_DEBUG "To print debug information" ON)
# PYTHON Wrapper
option(WITH_PYTHON_SUPPORT "To build the SWIG Python 3.8 wrapper" ON)

# Include directories for the lib
include_directories(include)
add_library(
        LogHelper
        SHARED
        src/log_helper.cpp
        src/log_helper_c_wrapper.cpp
        src/file_writer.cpp
        include/log_helper.h
        include/log_helper.hpp
)

########################################################################################################################
# Set the logging approach
# UPDATE IF THERE IS CHANGES on common.hpp
#    enum class LoggingType{
#        LOCAL_ONLY = 0,
#        UDP_ONLY,
#        LOCAL_AND_UDP
#    } ;
if (NOT LOGGING_TYPE OR LOGGING_TYPE STREQUAL "LOCAL")
    message("-- Selecting LOCAL logging approach")
    add_definitions(-DLOGGING_TYPE=0)
elseif (LOGGING_TYPE STREQUAL "UDP")
    message("-- Selecting UDP logging approach")
    add_definitions(-DLOGGING_TYPE=1)
elseif (LOGGING_TYPE STREQUAL "LOCAL_AND_UDP")
    message("-- Selecting LOCAL_AND_UPD logging approach")
    add_definitions(-DLOGGING_TYPE=2)
else ()
    message(FATAL_ERROR "INVALID LOGGING_TYPE CONFIGURATION, USE: LOCAL, UPD, LOCAL_AND_UDP")
endif ()


########################################################################################################################
# CONFIGURE INSTALLATION
# Set the pkg-config files

include(GNUInstallDirs)
set(CMAKE_BUILD_TYPE Release)

set_target_properties(LogHelper PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(LogHelper PROPERTIES PUBLIC_HEADER "include/log_helper.h;include/log_helper.hpp")

configure_file(cmake_install_scripts/LogHelper.pc.in LogHelper.pc @ONLY)
target_include_directories(LogHelper PRIVATE include)

install(TARGETS LogHelper
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_link_libraries(LogHelper)

install(FILES ${CMAKE_BINARY_DIR}/LogHelper.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

# CONFIGURE log-helper.conf file #######################################
# Path to the radiation-benchmarks repository
set(HOME_DIR $ENV{HOME})

# Set radiation-benchmarks directory to keep compatibility with older setup
if (NOT RAD_BENCHS_INSTALL_DIR)
    set(RAD_BENCHS_INSTALL_DIR ${HOME_DIR}/radiation-benchmarks)
endif ()
# Killall signal command to be sent to the SW watchdog
if (NOT WATCHDOG_COMMANDS)
    set(WATCHDOG_COMMANDS "echo -n")
endif ()
# System tmp dir
if (NOT TMP_DIR)
    set(TMP_DIR /tmp)
endif ()
# Path to file that will contains 1/0 that refers to ECC enabled or disabled respectively.
if (NOT ECC_INFO_FILE_DIR)
    set(ECC_INFO_FILE_DIR /tmp/ecc-info-file.txt)
endif ()
# Server that will receive the messages IP
if (NOT SERVER_IP)
    set(SERVER_IP 192.168.1.5)
endif ()
# Server port that will receive the messages
if (NOT SERVER_PORT)
    set(SERVER_PORT 1024)
endif ()

# Set the log dir for the benchmarks
# Default is /var/radiation-benchmarks
if (NOT LOG_DIR)
    set(LOG_DIR "/${CMAKE_INSTALL_LOCALSTATEDIR}")
endif ()
########################################################################################################################
# Installation process
set(INSTALL_SCRIPTS_DIR cmake_install_scripts)
# configuration file that will be read by the library
set(RADIATION_CONF_FILE radiation-benchmarks.conf)
configure_file(${INSTALL_SCRIPTS_DIR}/${RADIATION_CONF_FILE}.in ${RADIATION_CONF_FILE} @ONLY)
add_definitions(-DCONFIG_FILE_PATH="${PATH_TO_CONFIG_FILE}/${RADIATION_CONF_FILE}")
#install(
#        FILES ${CMAKE_BINARY_DIR}/${RADIATION_CONF_FILE}
#        DESTINATION /${CMAKE_INSTALL_SYSCONFDIR}
#)

# Destination of the atBoot.sh file
include(ProcessorCount)
ProcessorCount(NUMBER_OF_PROCESSORS)
#configure_file(${INSTALL_SCRIPTS_DIR}/atBoot.sh.in atBoot.sh @ONLY)
#install(
#        FILES ${CMAKE_BINARY_DIR}/atBoot.sh
#        DESTINATION ${HOME_DIR}
#)
#
## Destination of the rc.local file
#configure_file(${INSTALL_SCRIPTS_DIR}/rc.local.in rc.local @ONLY)
#install(
#        FILES ${CMAKE_BINARY_DIR}/rc.local
#        DESTINATION /${CMAKE_INSTALL_SYSCONFDIR}
#)

## Post install script commands
#set(SUDOERS_ADD \"$ENV{USER}\ ALL=NOPASSWD:/sbin/reboot\")
#install(CODE
#        "execute_process(COMMAND ${INSTALL_SCRIPTS_DIR}/post_install.sh
#        ${LOG_DIR} ${HOME_DIR} ${SUDOERS_ADD})")
## uninstall target
#if (NOT TARGET uninstall)
#    configure_file(
#            "${INSTALL_SCRIPTS_DIR}/Uninstall.cmake.in"
#            "${CMAKE_CURRENT_BINARY_DIR}/Uninstall.cmake"
#            IMMEDIATE @ONLY)
#
#    add_custom_target(uninstall
#            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/Uninstall.cmake)
#endif ()

########################################################################################################################

# Debug feature
if (WITH_DEBUG)
    add_definitions(-DDEBUG=1)
endif ()

# Python wrapper
if (WITH_PYTHON_SUPPORT)
    message("-- Building with python support, run the following command before run cmake:
             -- sudo apt install python3.8 python3.8-dev python3.8-distutils python3.8-venv swig")
    find_package(SWIG REQUIRED)
    include(UseSWIG)
    set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)

    ##    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    #    find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
    find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)

    set_property(SOURCE python_wrapper/log_helper.i PROPERTY CPLUSPLUS ON)
    set_property(SOURCE python_wrapper/log_helper.i PROPERTY SWIG_MODULE_NAME log_helper)

    swig_add_library(
            log_helper
            TYPE MODULE
            LANGUAGE python
            OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES python_wrapper/log_helper.i
    )
    add_library(${PROJECT_NAMESPACE}::log_helper ALIAS log_helper)

    target_include_directories(log_helper
            PRIVATE
            include
            ${Python3_INCLUDE_DIRS}
            )
    set_property(TARGET log_helper PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

    # note: Only for UNIX
#    set_target_properties(log_helper PROPERTIES
#            INSTALL_RPATH "$ORIGIN:$ORIGIN/../../${PYTHON_PROJECT}/.libs"
#            )
#    target_link_libraries(log_helper PRIVATE ${PROJECT_NAMESPACE}::log_helper)


    #    swig_add_library(log_helper
    #            TYPE SHARED
    #            LANGUAGE python
    #            OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
    #            SOURCES python_wrapper/log_helper.i
    #            )
    set_property(
            TARGET log_helper
            PROPERTY INCLUDE_DIRECTORIES ${Python_INCLUDE_DIRS}
            include
    )

#    set_property(
#            SOURCE python_wrapper/log_helper.i
#            PROPERTY SWIG_FLAGS "-py3" "-c++" "-extranative"
#    )

    TARGET_LINK_LIBRARIES(log_helper ${PYTHON_LIBRARIES})
    TARGET_LINK_LIBRARIES(log_helper LogHelper)
endif ()

########################################################################################################################
# Build examples
add_subdirectory(examples)
