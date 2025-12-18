// calculate_support_vectors.cpp
// #define EIGEN_USE_MKL
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
// #define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
const double epsilon = 1e-10;
bool AreSame(double a, double b)
{
    return fabs(a - b) < epsilon;
}
extern "C" {
    
    void calculate_support_vectors_indexes(
        double* Y_data, double* H_data, double* W_data, double* K_data,
        int rows, int k_rows, int k_cols,
        int MAX_ITERATION,
        int* non_zero_W_indices, bool* completed, double* min_M, int* zero_M_count, int* iteration_number
    ) {
        using namespace Eigen;
        
        // Map input arrays to Eigen matrices
        // Map<MatrixXd> X(X_data, rows, cols);
        Map<MatrixXd> Y(Y_data, rows, 1);
        Map<MatrixXd> H(H_data, rows, 1);
        Map<MatrixXd> W(W_data, rows, 1);
        Map<MatrixXd> K(K_data, k_rows, k_cols);

        // Initialize completed array
        std::vector<bool> completed_vec(1, false);
        
        MatrixXd M(rows, 1); 
        int iteration;
        for (iteration = 0; iteration < MAX_ITERATION; ++iteration) {
            if (std::all_of(completed_vec.begin(), completed_vec.end(), [](bool c) { return c; })) {
                break;
            }
            
            M.noalias() = Y.cwiseProduct(H);

            for (int c = 0; c < 1; ++c) {
                if (completed_vec[c]) {
                    continue;
                }
                // *iteration_number = iteration;
                int i;
                M.col(c).minCoeff(&i);
                if (M(i, c) <= 0.00) {
                    double delta;
                    // if (AreSame(Y(i, c), 1.0)) {
                    //     delta = (1.0 - H(i, c)) / K(i, i);
                    // }
                    // else {
                    //     delta = (-1.0 - H(i, c)) / K(i, i);
                    // }
                    delta = (Y(i, c) - H(i, c)) / K(i, i);
                    // b = beta^(0.5*(y+1))
                    // double b = pow(beta, 0.5 * (Y(i, c) + 1));
                    // double delta = (b - H(i, c));
                    W(i, c) += delta;
                    H.col(c) += delta * K.col(i);

                    continue;
                }
                // *iteration_number = iteration;
                ArrayXd H_col_c = H.col(c).array();
                ArrayXd W_col_c = W.col(c).array();
                ArrayXd K_diag = K.diagonal().array();
                ArrayXd M_l = Y.col(c).array() * (H_col_c - W_col_c * K_diag);

                M_l = M_l * (W_col_c.abs() > epsilon).cast<double>();

                std::vector<int> non_zero_W;
                for (int j = 0; j < rows; ++j) {
                    if (!AreSame(W(j, c),0.0)) {
                        non_zero_W.push_back(j);
                    }
                }



                if (non_zero_W.empty()) {
                    completed_vec[c] = true;
                    // *iteration_number = iteration;
                    continue;
                }
                int max_idx = *std::max_element(non_zero_W.begin(), non_zero_W.end(), [&](int a, int b) {
                    return M_l(a) < M_l(b);
                });
                
                if (M_l(max_idx) > 0.0) {
                    H.col(c) -= W(max_idx, c) * K.col(max_idx);
                    W(max_idx, c) = 0.0;
                    // *iteration_number = iteration;
                    continue;
                }
                
                // *iteration_number = iteration;
                completed_vec[c] = true;
            }
        }
        
            *iteration_number = iteration;
        int count = 0;
        for (int i = 0; i < rows; ++i) {
            if (!AreSame(W.row(i).cwiseAbs().sum(),0.0)) {
                non_zero_W_indices[count] = i;
                count +=1;
            }
        }
        // std::cout << Y.cwiseProduct(H).minCoeff() << " max M " << std::endl;
        
        *min_M = M.minCoeff();
        *zero_M_count = (M.array() <= 0.0).count();
        
        std::copy(completed_vec.begin(), completed_vec.end(), completed);
    }
}
