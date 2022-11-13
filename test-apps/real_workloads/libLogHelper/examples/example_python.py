#!/usr/bin/python3

import random

try:
    import log_helper
except ModuleNotFoundError as e:
    print("Please add the built python wrapper to the PYTHONPATH first")
    raise e


def main():
    # Input list to be sorted
    size = 8192 << 8
    # This variable is generally set to infinity iterations, for debug we set to 100 iterations
    iterations = 100
    # Start the log with filename including "my_benchmark" to its name, and
    # header inside log file will print the detail "size:x repetition:y"
    log_helper.start_log_file("SampleQuicksortPython3", f"size:{size} repetition:{iterations}")
    # set the maximum number of errors allowed for each iteration,
    #  default is 500
    log_helper.set_max_errors_iter(800)

    # set the interval of iteration to print details of current test,
    # default is 1.
    # CAUTION: Avoid writing too much on disc/ethernet, this can break your test
    log_helper.set_iter_interval_print(5)

    print("Starting sample benchmark in Python 3, log file is", log_helper.get_log_file_name())

    # This is just an example of input. In a real scenario, we should load the data from a  file,
    # then we calculate nothing on the Device Under Test but the evaluation kernel
    gold_array = [random.uniform(-1, 1) for _ in range(size)]
    i = 0
    while i < iterations:
        # copy to the tmp vector
        input_array = gold_array[:]
        log_helper.start_iteration()
        # Execute the test (ONLY THE KERNEL), log functions will measure kernel time
        input_array.sort()
        log_helper.end_iteration()
        # insert a fake SDC to test the comparator
        if i % 20 == 0:
            input_array[333] = size

        error_count, info_count = 0, 0
        # Testing with error_count > 0 for some iterations
        #  You can call as many log_error_detail(str) as you need
        #  However, it will log only the 500 errors or the
        #  max_errors_iter set with set_max_errors_iter()
        for p in range(len(input_array) - 2):
            p0, p1 = input_array[p], input_array[p + 1]
            if p0 > p1:
                error_detail = f"position {p} val {p0} is higher than next position {p1}"
                print("Error at iteration", i, error_detail)
                log_helper.log_error_detail(error_detail)
                error_count += 1
        if i == 16:
            # only the value passed to set_max_infos_iter() will be logged (default is 500)
            log_helper.log_info_detail("info of event during iteration")
            info_count = info_count + 520

        # log how many errors the iteration had
        # if error_count is greater than 800, or the
        # max_errors_iter set with set_max_errors_iter()
        # it will terminate the execution of the benchmark
        log_helper.log_error_count(error_count)
        # the logging info function does not terminate the program
        log_helper.log_info_count(info_count)
        i += 1

    # Finish the log file
    log_helper.end_log_file()
    print("Sample benchmark has finished!")


if __name__ == '__main__':
    main()
