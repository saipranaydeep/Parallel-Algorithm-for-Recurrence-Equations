// This is the code for reccursion problem Xn = an1*Xn-1 + an2*Xn-2 + anm*Xn-m + bn 

#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
using namespace std;

using Matrix = vector<vector<double>>;
using Vector = vector<double>;

// Matrix-vector multiplication
Vector multiply(const Matrix& mat, const Vector& vec) {
    size_t m = mat.size();
    size_t n = vec.size();
    Vector result(m, 0);
    #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

// Matrix-matrix multiplication
Matrix multiply(const Matrix& mat1, const Matrix& mat2) {
    size_t m = mat1.size();
    size_t n = mat2[0].size();
    size_t p = mat2.size();
    Matrix result(m, Vector(n, 0));
    #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < p; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

// Solve m-th order recurrence using recursive doubling
void solveMthOrder(const vector<Matrix>& A, const vector<Vector>& B, Vector& Z0, size_t m) {
    size_t n = A.size();
    vector<Matrix> Ak = A;   // Copy of A
    vector<Vector> Bk = B;   // Copy of B

    // Recursive doubling
    size_t logN = ceil(log2(n));
    for (size_t step = 0; step < logN; step++) {
        size_t stride = 1 << step;
        #pragma omp parallel for
        for (size_t i = stride; i < n; i += 2 * stride) {
            Bk[i] = multiply(Ak[i], Bk[i - stride]);  // A_i * B_{i-1}
            Ak[i] = multiply(Ak[i], Ak[i - stride]);  // A_i * A_{i-1}
        }
    }

    // Compute final state
    Z0 = multiply(Ak[n - 1], Z0); // A[n-1] * Z[0]
    // Z0 = multiply(Bk[n - 1], Z0); // + B[n-1]
}

int main() {
    size_t n = 8;   // Number of recurrence steps
    size_t m = 3;   // Order of the recurrence

    // Define matrices A[i] and vectors B[i]
    vector<Matrix> A(n, Matrix(m, Vector(m, 0)));
    vector<Vector> B(n, Vector(m, 0));
    Vector Z0(m, 0);

    // Initialize A[i], B[i], and Z0
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            for (size_t k = 0; k < m; k++) {
                A[i][j][k] = (j == 0) ? (rand() % 5 + 1) : (j - 1 == k ? 1 : 0);  // Recurrence matrix
            }
        }
        B[i][0] = rand() % 5 + 1;  // Only the first element of B[i] is non-zero
    }
    for (size_t i = 0; i < m; i++) Z0[i] = rand() % 5 + 1;  // Initial state

    // Solve the m-th order recurrence
    solveMthOrder(A, B, Z0, m);

    // Output the result
    cout << "Final state Z[n]: ";
    for (double z : Z0) {
        cout << z << " ";
    }
    cout << endl;

    return 0;
}
