#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log_helper.h"

int cmp_function(const void *a, const void *b) {
    return (*(int *) a > *(int *) b);
}

int main() {
    unsigned size = 8192 << 8;
    // Start the log with filename including "my_benchmark" to its name, and
    // header inside log file will print the detail "size:x repetition:y"
    char header[50] = "";
    char test_name[] = "SampleQuicksortC";
    // This variable is generally set to infinity iterations, for debug we set to 100 iterations
    int iterations = 100;
    sprintf(header, "size:%d repetition:%d", size, iterations);
    start_log_file(test_name, header);
    // set the maximum number of errors allowed for each iteration,
    // default is 500
    set_max_errors_iter(800);

    // set the interval of iteration to print details of current test,
    // default is 1.
    // CAUTION: Avoid writing too much on disc/ethernet, this can break your test
    set_iter_interval_print(5);
    char log_file_name[1024] = "";
    get_log_file_name(log_file_name);
    printf("Starting sample benchmark in C, the log file is at %s\n", log_file_name);

    int *vector = malloc(size * sizeof(int));
    int *tmp_vector = malloc(size * sizeof(int));
    if (vector == NULL || tmp_vector == NULL) {
        log_info_detail("could not allocate the arrays");
        end_log_file();
        exit(-1);
    }

    // This is just an example of input. In a real scenario, we should load the data from a  file,
    // then we calculate nothing on the Device Under Test but the evaluation kernel
    for (int i = 0; i < size; i++) {
        vector[i] = rand() % size;
    }

    for (int i = 0; i < iterations; i++) {
        //copy to the tmp vector
        memcpy(tmp_vector, vector, size * sizeof(int));
        start_iteration();
        // Execute the test (ONLY THE KERNEL), log functions will measure kernel time
        qsort(tmp_vector, size, sizeof(int), cmp_function);
        end_iteration();

        // insert a fake SDC to test the comparator
        if (i % 20 == 0) {
            tmp_vector[333] = size;
        }

        int error_count = 0;
        int info_count = 0;

        // Testing with error_count > 0 for some iterations
        // You can call as many log_error_detail(str) as you need
        // However, it will log only the 500 errors or the
        // max_errors_iter set with set_max_errors_iter()
        for (int p = 0; p < size - 1; p++) {
            int p0 = tmp_vector[p], p1 = tmp_vector[p + 1];
            if (p0 > p1) {
                char error_detail[128];
                sprintf(error_detail, "position %d val %d is higher than next position %d", p, p0, p1);
                printf("Error at iteration %d %s\n", i, error_detail);
                log_error_detail(error_detail);
                error_count++;
            }
        }

        if (i == 85) {
            // only the value passed to set_max_infos_iter() will be logged (default is 500)
            log_info_detail("Iteration 1326");
            info_count = 520;
        }

        // log how many errors the iteration had
        // if error_count is greater than 800, or the
        // max_errors_iter set with set_max_errors_iter()
        // it will terminate the execution of the program
        log_error_count(error_count);
        // the logging info function does not terminate the program
        log_info_count(info_count);
    }

    // Finish the log file
    end_log_file();
    free(vector);
    free(tmp_vector);
    printf("Sample benchmark has finished!\n");
}
