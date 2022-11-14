/*log_helper.i */
%module log_helper
// Make log_helper.cxx include this header:
%{
#define SWIG_FILE_WITH_INIT

#include "log_helper.hpp"

%}

%include <std_string.i>
%include <stdint.i>

// Make SWIG look into this header:
%include "log_helper.hpp"
