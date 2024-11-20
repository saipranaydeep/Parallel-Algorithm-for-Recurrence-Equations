// This is the code for finding f(z) where f(x) = Sum(bi*x^(N-i))
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <cstdlib>
using namespace std;

// Define the functions as per the problem constraints
long long f(long long b, long long gx) { // f should be associative
    return b + gx;
}

long long g(long long a, long long x) { // g should be distributive over f and semi-associative
    return a * x;
}

long long h(long long a, long long b) { // g(x, g(y, z)) = g(h(x, y), z)
    return a * b;
}

// Sequential recurrence solution
void sequentialRecurrence(const vector<long long>& a, const vector<long long>& b, long long &answer) {
    double sequential_start_time = omp_get_wtime();
    
    answer = 0;
    for (size_t i = 0; i < a.size(); i++) {
        answer = f(b[i], g(a[i], answer));
    }

    double sequential_end_time = omp_get_wtime();

    double sequential_time = sequential_end_time - sequential_start_time;
    cout << "Time taken for sequential execution is: " << sequential_time << "s\n";
}

// Parallel recurrence solution using recursive doubling
void parallelRecurrence(const vector<long long>& a, const vector<long long>& b, long long &answer) {
    size_t n = a.size();
    vector<long long> A(a), B(b);
    size_t logN = log2(n);

    double parallel_start_time = omp_get_wtime();

    for (size_t step = 0; step < logN; ++step) {
        size_t stride = 1 << step; // Stride long longs in each step
        #pragma omp parallel for 
        for (size_t i = stride - 1; i < n; i += 2 * stride) {
            if(i + stride < n) {
                B[i + stride] = f(B[i + stride], g(A[i + stride], B[i]));
                A[i + stride] = h(A[i + stride], A[i]);
            }
        }
    }

    answer = B.back();

    double parallel_end_time = omp_get_wtime();

    double parallel_time = parallel_end_time - parallel_start_time;
    cout << "Time taken for parallel execution is: " << parallel_time << "s\n";
}

int main() {
    size_t n = 10;
    int z = 3;
    size_t N = (1 << ((int) log2(n)+1));
    vector<long long> a(N, z), b(N, 1);

    for(size_t i = 0; i < n; i++) {
        b[i] = rand() % 3 + 1;
    }

    // Initialize random coefficients
    for (size_t i = n; i < N; i++) {
        a[i] = 1; 
        b[i] = 0;
    }    

    long long sequentialAnswer = 0, parallelAnswer = 0;

    // Solve sequentially
    sequentialRecurrence(a, b, sequentialAnswer);

    // Solve in parallel
    parallelRecurrence(a, b, parallelAnswer);

    // Output results
    cout << "Sequential answer: " << sequentialAnswer << "\n";
    cout << "Parallel answer: " << parallelAnswer << "\n";

    return 0;
}