// main_run.cpp
// This program builds example linear systems which are passed into
// the linear solvers. 
// The main() function at the end of the script asks the user which
// tests to perfrom and calls the relevant dense_solvers(n) and 
// sparse_solvers() functions. Analysis of the performance of the sparse solvers
// can also be run.

#include <iostream> 
#include <ctime>
#include <string>
#include "Matrix.cpp"
#include "CSRMatrix.cpp"
#include <chrono>

using namespace std;

// Function to run the sparse_GMRES solver with incresingly large inputs
// and measure the corressponding errors. Tridiagonal matrices of size N are 
// generated as CSRMatrix objects and passed into the solver.
void test_GMRES_performance(int N)
{
    // make tri-diagonal sparse matrix: [-1,2,-1]
    int nnzs = 3 * N - 2;   // number of entries

    auto* a = new double [nnzs];    // values
    int i;
    auto* ia = new int[N+1];    // row_index
    int itr_max;
    auto* ja = new int[nnzs];   // col_index
    int k;
    int mr;
    int n = N;
    auto* rhs = new double [N];
    double tol_abs;
    double tol_rel;
    double x_error;
    auto* x_estimate = new double [N];
    auto* x_exact = new double [N];
    k = 0;
    ia[0] = 0;

    for (i = 0; i < n; i++)
    {
        ia[i+1] = ia[i];
        if (0 < i)
        {
            ia[i+1] = ia[i+1] + 1;
            ja[k] = i-1;
            a[k] = -1.0;    // Upper diagonal
            k++;
        }

        ia[i+1] = ia[i+1] + 1;
        ja[k] = i;
        a[k] = 2.0;         // Main diagonal
        k++;

        if (i < N-1)
        {
            ia[i+1] = ia[i+1] + 1;
            ja[k] = i+1;
            a[k] = -1.0;    // Lower diagonal
            k++;
        }
    }
    // make RHS
    for (i = 0; i < n-1; i++)
    {
        rhs[i] = 0.0;
    }
    rhs[n-1] = (double) (n + 1);
    // make exact solution for comparison
    for ( i = 0; i < n; i++ )
    {
        x_exact[i] = (double) (i + 1);
    }
    // make initial guess
    for (i = 0; i < n; i++)
    {
        x_estimate[i] = 0.0;
    }
    x_error = 0.0;
    for (i = 0; i < n; i++)
    {
        x_error = x_error + pow (x_exact[i] - x_estimate[i], 2);
    }
    x_error = sqrt (x_error);
    // iteration and tolerance parameters
    itr_max = 200;
    mr = 200;
    tol_abs = 1.0E-08;
    tol_rel = 1.0E-08;

    cout << "\n";
    cout << "  Matrix order N = " << n << "\n";
    cout << "  Inner iteration limit = " << mr << "\n";
    cout << "  Outer iteration limit = " << itr_max << "\n";
    cout << "  Initial X_ERROR = " << x_error << "\n";

    // initialize the matrix
    auto* test_sparse_mat = new CSRMatrix<double>(N, N, nnzs, true);
    memcpy(test_sparse_mat->row_position, ia, (N+1)*sizeof(int));
    memcpy(test_sparse_mat->col_index, ja, nnzs*sizeof(int));
    memcpy(test_sparse_mat->values, a, nnzs*sizeof(double));
    test_sparse_mat->sparse_GMRES(x_estimate, rhs, itr_max, mr, tol_abs, tol_rel);

    // compute error
    x_error = 0.0;
    for (i = 0; i < n; i++)
    {
        x_error = x_error + pow (x_exact[i] - x_estimate[i], 2);
    }
    x_error = sqrt(x_error);

    cout << "  Final X_ERROR = " << x_error << "\n";
    delete test_sparse_mat;
}

// Function to run the sparse_BiCG solver with incresingly large inputs
// and measure the corressponding errors. Tridiagonal matrices of size N are 
// generated as CSRMatrix objects and passed into the solver.
void test_BiCG_performance(int N)
{
    // make tri-diagonal sparse matrix: [-1,4,-1]
    int nnzs = 3 * N - 2;   // number of entries

    auto* a = new double [nnzs];    // values
    int i;
    auto* ia = new int[N+1];    // row_index
    int itr_max;
    auto* ja = new int[nnzs];   // col_index
    int k;
    int mr;
    int n = N;
    auto* rhs = new double [N];
    double tol_abs;
    double tol_rel;
    double x_error;
    auto* x_estimate = new double [N];
    auto* x_exact = new double [N];
    k = 0;
    ia[0] = 0;

    for (i = 0; i < n; i++)
    {
        ia[i+1] = ia[i];
        if (0 < i)
        {
            ia[i+1] = ia[i+1] + 1;
            ja[k] = i-1;
            a[k] = -1.0;    // Upper diagonal
            k++;
        }

        ia[i+1] = ia[i+1] + 1;
        ja[k] = i;
        a[k] = 4.0;         // Main diagonal
        k++;

        if (i < N-1)
        {
            ia[i+1] = ia[i+1] + 1;
            ja[k] = i+1;
            a[k] = -1.0;    // Lower diagonal
            k++;
        }
    }

    // make exact solution for comparison
    for (i = 0; i < n; i++)
    {
        x_exact[i] = (double) (i + 1);
    }

    // make RHS
    for (i = 0; i < n-1; i++)
    {
        rhs[i] = 0.0+x_exact[i]*2;
    }
    rhs[n-1] = (double) (n + 1)+x_exact[n-1]*2;

    // make initial guess
    for (i = 0; i < n; i++)
    {
        x_estimate[i] = 0.0;
    }
    x_error = 0.0;
    for (i = 0; i < n; i++)
    {
        x_error = x_error + pow (x_exact[i] - x_estimate[i], 2);
    }
    x_error = sqrt (x_error);
    // iteration and tolerance parameters
    itr_max = 200;
    mr = 200;
    tol_abs = 1.0E-08;
    tol_rel = 1.0E-08;

    cout << "\n";
    cout << "  Matrix order N = " << n << "\n";
    cout << "  Inner iteration limit = " << mr << "\n";
    cout << "  Outer iteration limit = " << itr_max << "\n";
    cout << "  Initial X_ERROR = " << x_error << "\n";

    // initialize the matrix
    auto* test_sparse_mat = new CSRMatrix<double>(N, N, nnzs, true);
    memcpy(test_sparse_mat->row_position, ia, (N+1)*sizeof(int));
    memcpy(test_sparse_mat->col_index, ja, nnzs*sizeof(int));
    memcpy(test_sparse_mat->values, a, nnzs*sizeof(double));
    test_sparse_mat->sparse_GMRES(x_estimate, rhs, itr_max, mr, tol_abs, tol_rel);

    // compute error
    x_error = 0.0;
    for (i = 0; i < n; i++)
    {
        x_error = x_error + pow(x_exact[i] - x_estimate[i], 2);
    }
    x_error = sqrt(x_error);

    cout << "  Final X_ERROR = " << x_error << "\n";
    delete test_sparse_mat;
}

// Print dense solutions to terminal
template <class T>
void print_sol(Matrix<T> sol, int dp)
{
    for (int i = 0; i < sol.rows; i++)
    {
        cout << "x" << i+1 << " = "  << fixed << setprecision(dp) << sol.values[i] << endl;
    }
}

// Function to generate an n x n sized matrix and constant vector of size n.
// The corressponding expected solution is also calculated.
// The example system is passed into each of the solvers and the 
// calculated solutions are printed to terminal.
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
    auto* sol_vals = new double [n]();
    for (int i = 1; i < n+1; i++)
    {
        sol_vals[i-1] = (-double(n) / 4) + (double(i) / 2);
    }
    auto *sol = new Matrix<double>(n, 1, sol_vals);

    // Print test problem to terminal with expected solution
    cout << endl << "Example System";
    cout << endl << "Matrix A: ";
    test_A->printMatrix();
    cout << endl << "Vector b: ";
    test_b->printValues();
    cout << endl << "Expected solution x: ";
    sol->printValues();

    // ~~~~~~~~~~~~ Testing solvers ~~~~~~~~~~~~~
    int dp = 3;  // decimal place precision of final solutions

    // ~~~~~~~~~~~~ Jacobi ~~~~~~~~~~~~~
    string ans;
    cout << endl << "~~ Jacobi ~~";
    cout << endl << "Only suitable for matrices with non-zero elements on main diagonal";
    cout << endl << "Run test? y/n" << endl;
    cin >> ans;

    if (ans == "y")
    {
        auto start = chrono::steady_clock::now();
        int max_iter = 200;
        double A_jac_vals[n * n];
        double b_jac_vals[n];
        memcpy(A_jac_vals, test_A_vals, sizeof(test_A_vals));
        memcpy(b_jac_vals, test_b_vals, sizeof(test_b_vals));
        auto *A_jacobi = new Matrix<double>(n, n, A_jac_vals);
        auto *b_jacobi = new Matrix<double>(n, 1, b_jac_vals);

        Matrix<double> *x_jacobi = A_jacobi->Jacobi(*b_jacobi, dp, max_iter);
        cout << endl << "Solution to a precision of +/- " << pow(10, -1 * dp) << ": " << endl;
        print_sol(*x_jacobi, dp);
        auto end = chrono::steady_clock::now();
        cout << "Elapsed time in microseconds : " << chrono::duration_cast<chrono::microseconds>(end - start).count()<< endl;
        delete A_jacobi;
        delete b_jacobi;
    }

    // ~~~~~~~~~~~~ Conjugate-Gradient ~~~~~~~~~~~~~

    cout << endl << "~~ Conjugate-Gradient ~~";
    cout << endl << "Only suitable for symmetric matrices";
    cout << endl << "Run test? y/n" << endl;
    ans = to_string(n);
    cin >> ans;

    if (ans == "y")
    {
        auto start = chrono::steady_clock::now();
        double A_CG_vals[n * n];
        double b_CG_vals[n];
        memcpy(A_CG_vals, test_A_vals, sizeof(test_A_vals));
        memcpy(b_CG_vals, test_b_vals, sizeof(test_b_vals));
        auto *A_CG = new Matrix<double>(n, n, A_CG_vals);
        auto *b_CG = new Matrix<double>(n, 1, b_CG_vals);

        Matrix<double> *x_CG = A_CG->ConjGrad(*b_CG, dp);

        cout << endl << "Solution to a precision of +/- " << pow(10, -1 * dp) << ": " << endl;
        print_sol(*x_CG, dp);

        auto end = chrono::steady_clock::now();
        cout << "Elapsed time in microseconds : " << chrono::duration_cast<chrono::microseconds>(end - start).count()<< endl;
        delete A_CG;
        delete b_CG;
    }

    // ~~~~~~~~~~~~ Gaussian Elimination ~~~~~~~~~~~~~

    cout << endl << "~~ Gaussian Elimination ~~";
    cout << endl << "Run test? y/n" << endl;
    ans = n;
    cin >> ans;

    if (ans == "y")
    {
        auto start = chrono::steady_clock::now();
        double A_GE_vals[n * n];
        double b_GE_vals[n];
        memcpy(A_GE_vals, test_A_vals, sizeof(test_A_vals));
        memcpy(b_GE_vals, test_b_vals, sizeof(test_b_vals));
        auto *A_GE = new Matrix<double>(n, n, A_GE_vals);
        auto *b_GE = new Matrix<double>(n, 1, b_GE_vals);

        A_GE->GaussElim(*b_GE, dp);

        cout << endl << "Solution:" << endl;
        print_sol(*b_GE, dp);
        
        auto end = chrono::steady_clock::now();
        cout << "Elapsed time in microseconds : " << chrono::duration_cast<chrono::microseconds>(end - start).count()<< endl;
        delete A_GE;
        delete b_GE;
    }

    // ~~~~~~~~~~~~ Gauss Seidel ~~~~~~~~~~~~~
    cout << endl << "~~ Gauss Seidel ~~";
    cout << endl << "Run test? y/n" << endl;
    ans = to_string(n);
    cin >> ans;

    if (ans == "y")
    {
        int max_iter = 200;
        auto start = chrono::steady_clock::now();
        double A_GS_vals[n * n];
        double b_GS_vals[n];
        memcpy(A_GS_vals, test_A_vals, sizeof(test_A_vals));
        memcpy(b_GS_vals, test_b_vals, sizeof(test_b_vals));
        auto *A_GS = new Matrix<double>(n, n, A_GS_vals);
        auto *b_GS = new Matrix<double>(n, 1, b_GS_vals);

        Matrix<double> *x_GS = A_GS->GauSeidel(*b_GS, dp, max_iter);

        cout << endl << "Solution to a precision of +/- " << pow(10, -1 * dp) << ": " << endl;
        print_sol(*x_GS, dp);
        auto end = chrono::steady_clock::now();
        cout << "Elapsed time in microseconds : " << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;
        delete A_GS;
        delete b_GS;
    }

    // ~~~~~~~~~~~~ Successive Over-Relaxation ~~~~~~~~~~~~~
    cout << endl << "~~ Successive Over-Relaxation ~~";
    cout << endl << "Run test? y/n" << endl;
    ans = to_string(n);
    cin >> ans;

    if (ans == "y")
    {
        int max_iter = 200;
        auto start = chrono::steady_clock::now();
        double omega = 0.67;
        double A_SOR_vals[n * n];
        double b_SOR_vals[n];
        memcpy(A_SOR_vals, test_A_vals, sizeof(test_A_vals));
        memcpy(b_SOR_vals, test_b_vals, sizeof(test_b_vals));
        auto *A_SOR = new Matrix<double>(n, n, A_SOR_vals);
        auto *b_SOR = new Matrix<double>(n, 1, b_SOR_vals);

        Matrix<double> *x_SOR = A_SOR->SOR(omega, *b_SOR, dp, max_iter);

        cout << endl << "Solution to a precision of +/- " << pow(10, -1 * dp) << ": " << endl;
        print_sol(*x_SOR, dp);

                auto end = chrono::steady_clock::now();
        cout << "Elapsed time in microseconds : " << chrono::duration_cast<chrono::microseconds>(end - start).count()<< endl;
        delete A_SOR;
        delete b_SOR;
    }

    // ~~~~~~~~~~~~ GMRES ~~~~~~~~~~~~~~
    cout << endl << "~~ GMRES ~~";
    cout << endl << "Run test? y/n" << endl;
    ans = to_string(n);
    cin >> ans;

    if (ans == "y")
    {
        auto start = chrono::steady_clock::now();
        double A_GMRES_vals[n * n];
        double b_GMRES_vals[n];
        double x0_GMRES[n] = {0};
        memcpy(A_GMRES_vals, test_A_vals, sizeof(test_A_vals));
        memcpy(b_GMRES_vals, test_b_vals, sizeof(test_b_vals));
        auto *A_GMRES = new Matrix<double>(n, n, A_GMRES_vals);

        A_GMRES->dense_GMRES(x0_GMRES, b_GMRES_vals, 5, 4, 1.0E-08,1.0E-08);

        cout << endl << "Solution: " << endl;
        auto end = chrono::steady_clock::now();
        cout << "Elapsed time in microseconds : " << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;
        for (int i = 0; i < n; i++)
        {
            cout << "x" << i+1 << " = "  << fixed << setprecision(dp) << x0_GMRES[i] << endl;
        }

        delete A_GMRES;    
    }

    // ~~~~~~~~~~~~ Bi-Conjugate Gradient ~~~~~~~~~~~~~~
    cout << endl << "~~ BiCG ~~";
    cout << endl << "Run test? y/n" << endl;
    ans = to_string(n);
    cin >> ans;

    if (ans == "y")
    {
        auto start = chrono::steady_clock::now();
        double A_BiCG_vals[n * n];
        double b_BiCG_vals[n];
        double x0_BiCG[n] = {0};
        memcpy(A_BiCG_vals, test_A_vals, sizeof(test_A_vals));
        memcpy(b_BiCG_vals, test_b_vals, sizeof(test_b_vals));
        auto *A_BiCG = new Matrix<double>(n, n, A_BiCG_vals);

        A_BiCG->BiCG(x0_BiCG, b_BiCG_vals, 50, 1.0E-08);

        cout << endl << "Solution: " << endl;
        
        auto end = chrono::steady_clock::now();
        cout << "Elapsed time in microseconds : " << chrono::duration_cast<chrono::microseconds>(end - start).count()<< endl;
        for (int i = 0; i < n; i++)
        {
            cout << "x" << i+1 << " = "  << fixed << setprecision(dp) << x0_BiCG[i] << endl;
        }

        delete A_BiCG;    
    } 
    delete test_A;
    delete test_b;
    delete sol;
}

// Generate an example CSRMatrix object and pass it into the sparse sovlers.
// The expected solution and calculated solutions are printed to terminal.
void sparse_solvers()
{
    auto* test_sparse_mat = new CSRMatrix<double>(4, 4, 9, true);

    double values[9] = {7, 1, 8, 2, 2, 9, 3, 2, 6};
    int col_ind[9] = {0, 1, 1, 2, 1, 2, 3, 2, 3};
    int row_ind[5] = {0, 2, 4, 7, 9};

    for (int i = 0; i < 9; i++)
    {
        test_sparse_mat->values[i] = values[i];
        test_sparse_mat->col_index[i] = col_ind[i];
    }
    for (int i = 0; i < 5; i++)
    {
        test_sparse_mat->row_position[i] = row_ind[i];
    }

    double rhs[4] = {9, 22, 43, 30};

    cout << endl << "Matrix, A, to test sparse solvers: " << endl;
    test_sparse_mat->printMatrix();

    cout << endl << "Constant vector, b:" << endl;
    for (int i = 0; i < 4; i++)
    {
        cout << rhs[i] << endl;
    }

    cout << endl << " Expected Solution x:" << endl;
    for (int i = 0; i < 4; i++)
    {
        cout << i+1 << endl;
    }

    double x_GMRES[4] = {0, 0, 0, 0};
    double x_BiCG[4] = {0, 0, 0, 0};

    test_sparse_mat->sparse_GMRES(x_GMRES, rhs, 5, 4, 1.0E-08,1.0E-08);
    cout << endl << "GMRES Solution: " << endl;
    for (int i = 0; i < 4; i++)
    {
        cout << "x" << i+1 << " = "  << x_GMRES[i] << endl;
    }

    test_sparse_mat->BiCG(x_BiCG, rhs, 50,1.0E-08);   
    cout << endl << "BiCG Solution: " << endl;
    for (int i = 0; i < 4; i++)
    {
        cout << "x" << i+1 << " = "  << x_BiCG[i] << endl;
    }
}

// Main function to call the relevant functions to demonstrate the 
// solver library.
int main()
{
    string ans = "n";
    cout << endl << "Solving the Linear Equation Ax = b in sparse and dense systems for real positive definite matrices." << endl;
    cout << "Giles Matthews, Hanxiao Zhang and Raha Moosavi." << endl;
    cout << endl << "--------- Dense Solvers ---------" << endl;
    cout << endl << "Test dense solvers? y/n" << endl;
    cin >> ans;

    // ~~~~ Dense Solvers ~~~~
    if (ans == "y")
    {
        int n = 11;
        dense_solvers(n);
        ans = "n";
    }

    cout << endl << "--------- Sparse Solvers ---------" << endl;
    cout << endl << "Test sparse solvers? y/n" << endl;
    cin >> ans;

    // ~~~~ Sparse Solvers ~~~~
    if (ans == "y")
    {
        sparse_solvers();
        ans = "n";
    }
    
    // ~~~~~~~~~~~~~~ Test Sparse Performance ~~~~~~~~~~~~~~~~
    cout << endl << "Test performance of sparse solvers? y/n" << endl;
    cin >> ans;

    if (ans == "y")
    {
        int n = 20;
        cout << endl << "BiCG Performance:" << endl;
        for (int i = 0; i<5;i++)
        {
            test_BiCG_performance(n);
            n += 20;
        }
        cout << endl << "GMRES Performance:" << endl;
        for (int i = 0; i<5;i++)
        {
            test_GMRES_performance(n);
            n += 20;
        } 
        ans = "n";
    }

    // ~~~~~~~~~~~~~~ Read/write to file ~~~~~~~~~~~~~~~~

    ans = "n";
    cout << endl << "--------- Reading and Writing to file --------" << endl;
    cout << endl << "Read dense system from file? y/n" << endl;
    cin >> ans;

    // Demonstration of read/write functionality for dense systems
    // using Jacobi solver. Files used:
    //      dense_mat.txt   (Matrix A)
    //      dense_mat_b.txt (Vector b)
    //      solution.txt    (solution x)
    if (ans == "y")
    {
        int rows = 3;
        int cols = 3;

        char A_file[] = "./dense_mat.txt";
        char b_file[] = "./dense_mat_b.txt";
        char x_file[] = "./solution.txt";

        auto* read_A = new Matrix<double>(rows, cols, A_file);
        auto* read_b = new Matrix<double>(rows, 1, b_file);

        cout << endl << "Matrix A (dense_mat.txt):";
        read_A->printMatrix();
        cout << endl << "Vector b (dense_mat_b.txt):";
        read_b->printMatrix();

        Matrix<double> *x = read_A->Jacobi(*read_b, 3, 200);

        x->write_matrix(x_file);
        cout << endl << "Solution x (solution.txt):";
        x->printMatrix();

        ans = "n";

        delete read_A;
        delete read_b;
    }

    // Demonstration of read/write functionality for sparse systems
    // using GMRES solver. Files used:
    //      row_pos.txt   (Matrix A)
    //      col_ind.txt (Vector b)
    //      values.txt    (solution x)
    cout << endl << "Read sparse system from file? y/n" << endl;
    cin >> ans;

    if (ans == "y")
    {
        int rows = 3;
        int cols = 3;

        char row_pos[] = "./row_pos.txt";
        char col_ind[] = "./col_ind.txt";
        char values[] = "./values.txt";

        double rhs[4] = {1, 2, 3, 4};
        double x[4] = {0};

        auto read_A = CSRMatrix<double>(4, 4, 12, row_pos, col_ind, values);        

        cout << endl << "Matrix A:" << endl;;
        read_A.printMatrix();

        read_A.sparse_GMRES(x, rhs, 5, 4, 1.0E-08,1.0E-08);
        
        cout << endl << "Solution: " << endl;
        for (int i = 0; i < 4; i++)
        {
            cout << "x" << i+1 << " = "  << x[i] << endl;
        }

        ans = "n";
    }
    return 0;
}