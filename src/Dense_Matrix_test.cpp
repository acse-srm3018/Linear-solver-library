#include <iostream> 
#include <ctime>
#include "Matrix.cpp"
#include "CSRMatrix.cpp"
#include <chrono>
#include <assert.h> 
#include <stdio.h> 
#include <cassert>
#define NDEBUG
#include <math.h>       /* nearbyint */

using namespace std;

void dense_solvers(int n)
{
    // Generate test matrix A with 2s on k diagonal
    // and -1s on diagonals k-1 and k+1.
    // Also, 1s in top right and bottom left.
    double test_A_vals[n * n]{0};
    double diag_vals[] = {-1, 2, -1};

    // Fill three diagonals, excluding corners
    for (int i = 1; i < n-1;  i++)
    {
        test_A_vals[(i-1) + i * n] = diag_vals[0];
        test_A_vals[i + i * n] = diag_vals[1];
        test_A_vals[i+1 + i * n] = diag_vals[2];
    }

    // Enter values in four corners
    test_A_vals[0] = diag_vals[1];
    test_A_vals[1] = diag_vals[2];
    test_A_vals[n-1] = 1;
    test_A_vals[(n-1) * n] = 1;
    test_A_vals[(n * n) - 1] = diag_vals[1];
    test_A_vals[(n * n) - 2] = diag_vals[0];
    
    // Test Matrix A
    auto *test_A = new Matrix<double>(n, n, test_A_vals);

    // Generate constant vector b 
    // This vector which is all zeros final element = 1
    double test_b_vals[n]{0};
    test_b_vals[n-1] = 1;
    auto *test_b = new Matrix<double>(n, 1, test_b_vals);

    // Generate expected solution for comparison
    // solution has the form: xi = -n/4 + i/2
    double sol_vals[n];
    for (int i = 1; i < n+1; i++)
    {
        sol_vals[i-1] = (-double(n) / 4) + (double(i) / 2);
    }
    auto *sol = new Matrix<double>(n, 1, sol_vals);

    // ~~~~~~~~~~~~ Testing solvers ~~~~~~~~~~~~~
    int dp = 3;  // decimal place precision of final solutions
    double max_iter = 500; //Maximum iteration
    // ~~~~~~~~~~~~ Jacobi ~~~~~~~~~~~~~
    double A_jac_vals[n * n];
    double b_jac_vals[n];
    memcpy(A_jac_vals, test_A_vals, sizeof(test_A_vals));
    memcpy(b_jac_vals, test_b_vals, sizeof(test_b_vals));
    auto *A_jacobi = new Matrix<double>(n, n, A_jac_vals);
    auto *b_jacobi = new Matrix<double>(n, 1, b_jac_vals);

    Matrix<double> *x_jacobi = A_jacobi->Jacobi(*b_jacobi, dp, max_iter);
    for (int i = 0; i < n; i++)
    {
    assert (abs(x_jacobi->values[i] - sol_vals[i]) <= 0.1);
    }
    cout << endl << "Jacobi passed test "<< endl;

    delete A_jacobi;
    delete b_jacobi;

    
    // ~~~~~~~~~~~~ Conjugate-Gradient ~~~~~~~~~~~~~
    double A_CG_vals[n * n];
    double b_CG_vals[n];
    memcpy(A_CG_vals, test_A_vals, sizeof(test_A_vals));
    memcpy(b_CG_vals, test_b_vals, sizeof(test_b_vals));
    auto *A_CG = new Matrix<double>(n, n, A_CG_vals);
    auto *b_CG = new Matrix<double>(n, 1, b_CG_vals);

    Matrix<double> *x_CG = A_CG->ConjGrad(*b_CG, dp);
    for (int i = 0; i < n; i++)
    {
    assert (abs(x_CG->values[i] - sol_vals[i]) <= 0.1);
    }
    cout <<endl << "Conjugate Gradient method passed test  "<< endl;
    delete A_CG;
    delete b_CG;

    // ~~~~~~~~~~~~ Gaussian Elimination ~~~~~~~~~~~~~
    double A_GE_vals[n * n];
    double b_GE_vals[n];
    memcpy(A_GE_vals, test_A_vals, sizeof(test_A_vals));
    memcpy(b_GE_vals, test_b_vals, sizeof(test_b_vals));
    auto *A_GE = new Matrix<double>(n, n, A_GE_vals);
    auto *b_GE = new Matrix<double>(n, 1, b_GE_vals);

    delete A_GE;
    delete b_GE;

    // ~~~~~~~~~~~~ Gauss Seidel ~~~~~~~~~~~~~
    double A_GS_vals[n * n];
    double b_GS_vals[n];
    memcpy(A_GS_vals, test_A_vals, sizeof(test_A_vals));
    memcpy(b_GS_vals, test_b_vals, sizeof(test_b_vals));
    auto *A_GS = new Matrix<double>(n, n, A_GS_vals);
    auto *b_GS = new Matrix<double>(n, 1, b_GS_vals);

    Matrix<double> *x_GS = A_GS->GauSeidel(*b_GS, dp, max_iter);
    
    for (int i = 0; i < n; i++)
    {
    assert (abs(x_GS->values[i] - sol_vals[i]) <= 0.1);
    }
    cout << endl<< "Gauss-Seidel passed test  " << endl;

    // ~~~~~~~~~~~~ Successive Over-Relaxation ~~~~~~~~~~~~~
    double omega = 0.9;
    double A_SOR_vals[n * n];
    double b_SOR_vals[n];
    memcpy(A_SOR_vals, test_A_vals, sizeof(test_A_vals));
    memcpy(b_SOR_vals, test_b_vals, sizeof(test_b_vals));
    auto *A_SOR = new Matrix<double>(n, n, A_SOR_vals);
    auto *b_SOR = new Matrix<double>(n, 1, b_SOR_vals);

    Matrix<double> *x_SOR = A_SOR->SOR(omega, *b_SOR, dp, max_iter);
    
    for (int i = 0; i < n; i++)
    {
    assert (abs(x_SOR->values[i] - sol_vals[i]) <= 0.1);
    }
    
    cout << endl<<"SOR with 0.8 relaxtion factor passed test " << endl;
    delete A_SOR;
    delete b_SOR;


    // ~~~~~~~~~~~~ GMRES ~~~~~~~~~~~~~~
    double A_GMRES_vals[n * n];
    double b_GMRES_vals[n];
    double x0_GMRES[n] = {0};
    memcpy(A_GMRES_vals, test_A_vals, sizeof(test_A_vals));
    memcpy(b_GMRES_vals, test_b_vals, sizeof(test_b_vals));
    auto *A_GMRES = new Matrix<double>(n, n, A_GMRES_vals);

    delete A_GMRES;    


    // ~~~~~~~~~~~~ Bi-Conjugate Gradient ~~~~~~~~~~~~~~
    double A_BiCG_vals[n * n];
    double b_BiCG_vals[n];
    double x0_BiCG[n] = {0};
    memcpy(A_BiCG_vals, test_A_vals, sizeof(test_A_vals));
    memcpy(b_BiCG_vals, test_b_vals, sizeof(test_b_vals));
    auto *A_BiCG = new Matrix<double>(n, n, A_BiCG_vals);

    /*Matrix <double> *x_BiCG = A_BiCG->BiCG(x0_BiCG, b_BiCG_vals, 50, 1.0E-08);
    for (int i = 0; i < n; i++)
    {
    assert(x_BiCG->values[i]== sol_vals[i]);
    }
    cout << "passed test" << endl;
*/
    delete A_BiCG;    

    delete test_A;
    delete test_b;
    delete sol;
}

int main()
{
    cout << endl << "Testing some Linear solvers for different matrix sizes" << endl;
    cout << "Giles Matthews, Hanxiao Zhang and Raha Moosavi" << endl;
    cout << endl << "--------------------------------------------------------" << endl;

    for (int n = 10; n <= 24; n+=2)
    {
    cout << endl << "--------------------------------------------------------" << endl;
    cout << "Testing some Linear solvers for Matrix size " << n << "*" << n << endl;
    cout << endl << "--------------------------------------------------------" << endl;
    dense_solvers(n);
    }
    return 0;
}