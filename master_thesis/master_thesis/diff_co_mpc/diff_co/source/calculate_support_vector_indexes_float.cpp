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
const float epsilon = 1e-6;
bool AreSame(float a, float b)
{
    return fabs(a - b) < epsilon;
}
extern "C" {
    
    void calculate_support_vectors_indexes(
        float* Y_data, float* H_data, float* W_data, float* K_data,
        int rows, int k_rows, int k_cols,
        int MAX_ITERATION,
        int* non_zero_W_indices, bool* completed, float* min_M, int* zero_M_count, int* iteration_number
    ) {
        using namespace Eigen;
        // std::cout << "rows " << rows << std::endl;
        
        // Map input arrays to Eigen matrices
        // Map<MatrixXf> X(X_data, rows, cols);
        Map<MatrixXf> Y(Y_data, rows, 1);
        Map<MatrixXf> H(H_data, rows, 1);
        Map<MatrixXf> W(W_data, rows, 1);
        Map<MatrixXf> K(K_data, k_rows, k_cols);

        // Initialize completed array
        std::vector<bool> completed_vec(1, false);
        MatrixXf M(rows, 1); 
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
                if (M(i, c) <= 0.00f) {
                    float delta;
                    // if (AreSame(Y(i, c), 1.0)) {
                    //     delta = (1.0 - H(i, c)) / K(i, i);
                    // }
                    // else {
                    //     delta = (-1.0 - H(i, c)) / K(i, i);
                    // }
                    delta = (Y(i, c) - H(i, c)) / K(i, i);
                    // b = beta^(0.5*(y+1))
                    // float b = pow(beta, 0.5 * (Y(i, c) + 1));
                    // float delta = (b - H(i, c));
                    W(i, c) += delta;
                    H.col(c) += delta * K.col(i);

                    continue;
                }
                // *iteration_number = iteration;
                ArrayXf H_col_c = H.col(c).array();
                ArrayXf W_col_c = W.col(c).array();
                ArrayXf K_diag = K.diagonal().array();
                ArrayXf M_l = Y.col(c).array() * (H_col_c - W_col_c * K_diag);

                M_l = M_l * (W_col_c.abs() > epsilon).cast<float>();

                std::vector<int> non_zero_W;
                for (int j = 0; j < rows; ++j) {
                    if (!AreSame(W(j, c),0.0f)) {
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
                    W(max_idx, c) = 0.0f;
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
            if (!AreSame(W.row(i).cwiseAbs().sum(),0.0f)) {
                non_zero_W_indices[count] = i;
                count +=1;
            }
        }
        // std::cout << Y.cwiseProduct(H).minCoeff() << " max M " << std::endl;
        
        *min_M = M.minCoeff();
        *zero_M_count = (M.array() <= 0.0f).count();
        
        std::copy(completed_vec.begin(), completed_vec.end(), completed);
    }
}
