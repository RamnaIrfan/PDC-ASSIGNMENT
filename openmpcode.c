#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 10000000  // Large dataset (10 million elements)
#define MAX_THREADS 8  // Adjust based on available CPU cores
#define RUNS 10        

void merge(int arr[], int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++) L[i] = arr[left + i];
    for (j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void parallelMergeSort(int arr[], int left, int right, int depth) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if (depth < MAX_THREADS) {
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, left, mid, depth + 1);

                #pragma omp section
                parallelMergeSort(arr, mid + 1, right, depth + 1);
            }
        } else {
            parallelMergeSort(arr, left, mid, depth + 1);
            parallelMergeSort(arr, mid + 1, right, depth + 1);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    double total_time = 0.0;

    for (int run = 1; run <= RUNS; run++) {
        int* arr = (int*)malloc(SIZE * sizeof(int));
        if (!arr) {
            printf("Memory allocation failed!\n");
            return 1;
        }

        // Generate a new random array
        for (int i = 0; i < SIZE; i++) {
            arr[i] = rand() % 100000;
        }

        double start = omp_get_wtime();
        parallelMergeSort(arr, 0, SIZE - 1, 0);
        double end = omp_get_wtime();

        double time_taken = end - start;
        total_time += time_taken;

        printf("Run %d: Time taken = %f seconds\n", run, time_taken);
        free(arr);
    }

    double average_time = total_time / RUNS;
    printf("\nAverage Parallel Execution Time: %f seconds\n", average_time);

    return 0;
}
