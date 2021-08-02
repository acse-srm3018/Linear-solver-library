// Header file for Matrix class used to store matrices
// in a dense format and call methods to perfrom matrix operations
// and solve linear systems.

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include "Matrix.h"

template <class T>
Matrix<T>::Matrix(int rows, int cols, bool preallocate):rows(rows), cols(cols), 
                    size_of_values(rows * cols), preallocated(preallocate)

{
    if (this->preallocated)
    {
        this->values = new T[this->size_of_values];
    }
}

// If preallocate is true in this constructor, we own the memory.
// If not, you are responsible for setting the values pointer and deleting the memory yourself
template <class T>
Matrix<T>::Matrix(int rows, int cols, T *values_ptr):rows(rows), cols(cols),
                    size_of_values(rows * cols), values(values_ptr)
{}

// Matrix constructor from file
template <class T>
Matrix<T>::Matrix(int rows, int cols, char *file_name) :rows(rows), cols(cols), size_of_values(rows* cols), preallocated(true)
{
    std::ifstream mat_file;
    this->values = new T[this->size_of_values];
    mat_file.open(file_name);
    for (int i=0; i<this->size_of_values; i++){
        mat_file >> this->values[i];
    }
    mat_file.close();
}

// Destructor
template <class T>
Matrix<T>::~Matrix()
{
    if (this->preallocated)
    {
        delete[] this->values;
    }
}

// Print matrix object values as an array
template <class T>
void Matrix<T>::printValues()
{
    std::cout << std::endl << "Printing values: " << std::endl;
    for (int i=0; i < this->size_of_values; i++)
    {
        std::cout << this->values[i] << " ";
    }
    std::cout << std::endl;
}

// Print matrix objects elements as a Matrix 
template <class T>
void Matrix<T>::printMatrix()
{
    std::cout << std::endl << "Printing Matrix: " << std::endl;
    for (int j=0; j < this->rows; j++)
    {
        std::cout << std::endl;
        for (int i = 0; i < this->cols; i++)
        {
            std::cout << this->values[i + j * this->cols] << " ";
        }
    }   
    std::cout << std::endl;
}

// Write Matrix to file
template <class T>
void Matrix<T>::write_matrix(char* file_name){
    std::ofstream mat_file;
    mat_file.open(file_name);
    for (int j = 0; j < this->rows; j++){
        for (int i = 0; i < this->cols; i++){
            mat_file << this->values[i + j * this->cols] << " ";
        }
    }
    mat_file.close();
}

template <class T>
void Matrix<T>::matMatMult(Matrix& mat_left, Matrix& output)
{

    // Check our dimernsions match
    if(this->cols != mat_left.rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }
    // Check if our output matrix has had space allocated to it
    if (output.values != nullptr)
    {
        // Check our dimensions match
        if (this->rows != output.rows || this->cols != output.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }
    // The output hasn't been preallocated, so we are going to do that
    else
    {
        output.values = new T[this->rows * mat_left.cols];
        output.preallocated = true;
    }

    // set to zero
    for (int i = 0; i < output.size_of_values; i++)
    {
        output.values[i] = 0.0;
    }

    // Now we can perform our matrix multiplication
    for (int i=0; i < this->rows; i++)
    {
        for (int k = 0; k < this->cols; k++)
        {
            for (int j = 0; j < mat_left.cols; j++)
            {
                output.values[i * output.cols + j] +=
                    this->values[i * this->cols + k] *
                    mat_left.values[k * mat_left.cols + j];
            }
        }
    }
}

// Matrix-vector multiplication for vector stored as Matrix object
template <class T>
void Matrix<T>::matVecMult(Matrix& vect, Matrix& output)
{
    // Check if our output matrix has had space allocated to it
    if (output.values != nullptr)
    {
        // Check our dimensions match
        if (this->cols != output.rows)
        {
            std::cerr << "Input dimensions for Matrix and Vector don't match" << std::endl;
            return;
        }
    }
    // The output hasn't been preallocated, so we are going to do that
    else
    {
        output.values = new T[this->rows * vect.cols];
        output.preallocated = true;
    }

    // set output values to zero
    for (int i = 0; i < output.size_of_values; i++)
    {
        output.values[i] = 0.0;
    }

    // Solve Ax = b using matrix vector multiplication
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < output.rows; j++)
        {
            output.values[i] += this->values[i * this->cols + j] * vect.values[j];
        }
    }
}

// Matrix vector multiplication for vector stored as array
template <class T>
void Matrix<T>::matVecMult(T vect[], T output[])
{
    memset(output, T(), this->rows * sizeof(T));

    // Solve Ax = b using matrix vector multiplication
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->rows; j++)
        {
            output[i] += this->values[i * this->cols + j] * vect[j];
        }
    }
}

template <class T>
double Matrix<T>::vectordot(Matrix& b)
{
    double product = 0;
    int n = this->rows;

    if (b.rows != n)
    {
        std::cout << std::endl << "Vectors are different sizes" << std::endl;
        return 0;
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            product += this->values[i] * b.values[i];
        }
        return product;
    }
}

// inner product in R^n
template <class T>
T inner (int n, T v1[], T v2[])
{
    T value = T();
    for (int i = 0; i < n; i++)
    {
        value += v1[i] * v2[i];
    }
    return value;
}

// Givens transform. Rotation in R^n
template <class T>
void Givens(T cos0, T sin0, int k, T vec[])
{
    T v1;
    T v2;

    v1 = cos0 * vec[k] - sin0 * vec[k+1];
    v2 = sin0 * vec[k] + cos0 * vec[k+1];

    vec[k] = v1;
    vec[k+1] = v2;
}

// daxpy in BLAS level 1
// compute z=a*x+y
template<class T>
void daxpy(T a, T x[], T y[], T z[], int n)
{
    for (int i=0; i<n; i++)
    {
        z[i] = a*x[i] + y[i];
    }
}

// matvecmul
template <class T>
void my_mat_vec_mul(int n, int nnzs, int row_ind[], int col_ind[], T val[], T x[], T y[])
{
    for (int i = 0; i < n; i++)
    {
        y[i] = 0.0;
    }

    for (int k = 0; k < nnzs; k++)
    {
        y[row_ind[k]] = y[row_ind[k]] + val[k] * x[col_ind[k]];
    }
}

// GMRES - Hanxiao Zhang
// See: Iterative Methods for Sparse Linear Systems (2000) by Yousef Saad
// Code reference: https://people.sc.fsu.edu/~jburkardt/cpp_src/mgmres/mgmres.html
template <class T>
void Matrix<T>::dense_GMRES(T *x0, T *rhs, int max_itr, int max_int_itr, T tol_abs, T tol_ref)
{
    T dot_prod, h_tmp;
    T *givens_cos, *v1_tmp, *v2_tmp, *res, *givens_sin, *v3_tmp, *iter_tmp;
    T dlt = 1.0e-03;
    int i, j, k, k_temp, itr, itr_used, mu_m, rho, rho_tol;

    givens_cos = new T[max_int_itr];
    v1_tmp = new T[max_int_itr+1];
    v2_tmp = new T[(max_int_itr+1)*max_int_itr];
    res = new T[this->rows];
    givens_sin = new T[max_int_itr];
    v3_tmp = new T[this->rows*(max_int_itr+1)];
    iter_tmp = new T[max_int_itr+1];
    
    // Pre-process the matrix. Get the number of total non-zeros entries in the matrix.
    int nnzs = 0;
    for (i=0; i<this->size_of_values; i++){
        if (this->values[i] !=0){
            nnzs++;
        }
    }
    int* row_ind = new int[nnzs];
    int* col_ind = new int[nnzs];
    T* val = new T[nnzs];
    k = 0;
    // get coo form of the sparse matrix
    for (j = 0; j < this->rows; j++)
    {
        for (i = 0; i < this->cols; i++)
        {
            if (this->values[i + j * this->cols] != T())
            {
                row_ind[k] = j;
                col_ind[k] = i;
                val[k] = this->values[i + j * this->cols];
                k++;
            }
        }
    }

    itr_used = 0;
    // iteration
    for (itr = 1; itr <= max_itr; itr++)
    {
        // Arnoldi's method (FOM)
        // get residue r0=b-Ax0
        my_mat_vec_mul(this->rows, nnzs, row_ind, col_ind, val, x0, res);
        for (i = 0; i < this->rows; i++)
        {
            res[i] = rhs[i] - res[i];
        }
        // Gram–Schmidt process
        rho = sqrt(inner(this->rows, res, res));
        if (itr == 1)
        {
            rho_tol = rho * tol_ref;
        }
        // Normalization
        for (i = 0; i < this->rows; i++)
        {
            v3_tmp[i] = res[i] / rho;
        }

        v1_tmp[0] = rho;
        for (i = 1; i <= max_int_itr; i++)
        {
            v1_tmp[i] = T();
        }

        for (i = 0; i < max_int_itr+1; i++)
        {
            for (j = 0; j < max_int_itr; j++)
            {
                v2_tmp[i+j*(max_int_itr+1)] = T();
            }
        }
        // upper Hessenberg matrix for orthogonal method of solving least square
        for (k = 1; k <= max_int_itr; k++)
        {
            k_temp = k;
            // Do QR decomposition
            my_mat_vec_mul(this->rows, nnzs, row_ind, col_ind, val, v3_tmp+(k-1)*this->rows, v3_tmp+k*this->rows);
            // inner product
            dot_prod = sqrt( inner( this->rows, v3_tmp+k*this->rows, v3_tmp+k*this->rows));
            // orthogonalization
            for ( j = 1; j <= k; j++ )
            {
                v2_tmp[(j-1)+(k-1)*(max_int_itr+1)] = inner( this->rows, v3_tmp+k*this->rows, v3_tmp+(j-1)*this->rows);
                for (i = 0; i < this->rows; i++)
                {
                    v3_tmp[i+k*this->rows] -= v2_tmp[(j-1)+(k-1)*(max_int_itr+1)] * v3_tmp[i+(j-1)*this->rows];
                }
            }

            v2_tmp[k+(k-1)*(max_int_itr+1)] = sqrt(inner(this->rows, v3_tmp+k*this->rows, v3_tmp+k*this->rows));
            // Householder orthogonalization
            if ((dot_prod + dlt * v2_tmp[k+(k-1)*(max_int_itr+1)]) == dot_prod)
            {
                for (j = 1; j <= k; j++)
                {
                    h_tmp = inner(this->rows, v3_tmp+k*this->rows, v3_tmp+(j-1)*this->rows);
                    v2_tmp[(j-1)+(k-1)*(max_int_itr+1)] += h_tmp;
                    for (i = 0; i < this->rows; i++)
                    {
                        v3_tmp[i+k*this->rows] -= h_tmp * v3_tmp[i+(j-1)*this->rows];
                    }
                }
                v2_tmp[k+(k-1)*(max_int_itr+1)] = sqrt(inner(this->rows, v3_tmp+k*this->rows, v3_tmp+k*this->rows));
            }

            if (v2_tmp[k+(k-1)*(max_int_itr+1)] != T())
            {
                for (i = 0; i < this->rows; i++)
                {
                    v3_tmp[i+k*this->rows] /= v2_tmp[k+(k-1)*(max_int_itr+1)];
                }
            }
            // Givens rotation. Orthogonal method to solve the least square
            if (1 < k)
            {
                for (i = 1; i <= k+1; i++)
                {
                    iter_tmp[i-1] = v2_tmp[(i-1)+(k-1)*(max_int_itr+1)];
                }
                for (j = 1; j <= k - 1; j++)
                {
                    Givens(givens_cos[j-1], givens_sin[j-1], j-1, iter_tmp);
                }
                for (i = 1; i <= k+1; i++)
                {
                    v2_tmp[i-1+(k-1)*(max_int_itr+1)] = iter_tmp[i-1];
                }
            }
            // Givens rotation.
            mu_m = sqrt( v2_tmp[(k-1)+(k-1)*(max_int_itr+1)]*v2_tmp[(k-1)+(k-1)*(max_int_itr+1)] + v2_tmp[ k+(k-1)*(max_int_itr+1)]*v2_tmp[ k+(k-1)*(max_int_itr+1)] );
            givens_cos[k-1] =  v2_tmp[(k-1)+(k-1)*(max_int_itr+1)] / mu_m;
            givens_sin[k-1] = -v2_tmp[ k   +(k-1)*(max_int_itr+1)] / mu_m;
            v2_tmp[(k-1)+(k-1)*(max_int_itr+1)] = givens_cos[k-1] * v2_tmp[(k-1)+(k-1)*(max_int_itr+1)] - givens_sin[k-1] * v2_tmp[ k   +(k-1)*(max_int_itr+1)];
            v2_tmp[k+(k-1)*(max_int_itr+1)] = 0;
            Givens( givens_cos[k-1], givens_sin[k-1], k-1, v1_tmp );

            rho = fabs(v1_tmp[k]);

            itr_used++;

            // Check if relative and absolute tolerances are satified
            if (rho <= rho_tol && rho <= tol_abs)
            {
                break;
            }
        }
        // Do iteration
        k = k_temp - 1;
        iter_tmp[k] = v1_tmp[k] / v2_tmp[k+k*(max_int_itr+1)];

        for (i = k; 1 <= i; i--)
        {
            iter_tmp[i-1] = v1_tmp[i-1];
            for (j = i+1; j <= k+1; j++)
            {
                iter_tmp[i-1] -= v2_tmp[(i-1)+(j-1)*(max_int_itr+1)] * iter_tmp[j-1];
            }
            iter_tmp[i-1] /= v2_tmp[(i-1)+(i-1)*(max_int_itr+1)];
        }

        for (i = 1; i <= this->rows; i++)
        {
            for (j = 1; j <= k + 1; j++)
            {
                x0[i-1] += v3_tmp[(i-1)+(j-1)*this->rows] * iter_tmp[j-1];
            }
        }

        // Check if relative and absolute tolerances are satified
        if (rho <= rho_tol && rho <= tol_abs)
        {
            break;
        }
    }

    delete [] row_ind;
    delete [] col_ind;
    delete [] val;
    delete [] givens_cos;
    delete [] v1_tmp;
    delete [] v2_tmp;
    delete [] res;
    delete [] givens_sin;
    delete [] v3_tmp;
    delete [] iter_tmp;
}

template <class T>
T Matrix<T>::check_res(T x[], T rhs[]){
    T* Ax = new T[this->rows];
    T* r = new T[this->rows];
    this->matVecMult(x, Ax);
    daxpy(-1.0, Ax, rhs, r, this->rows);
    return inner(this->rows, r, r);
}

// BiCG algorithm - Hanxiao Zhang
// See: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
template <class T>
void Matrix<T>::BiCG(T x0[], T rhs[], int max_itr, T tol_abs){
    int i;
    T* r0 = new T[this->rows];
    T* Ax0 = new T[this->rows];
    T* r0_hat = new T[this->rows];
    T* v0 = new T[this->rows]();
    T* v1 = new T[this->rows]();
    T* p0 = new T[this->rows]();
    T* p1 = new T[this->rows]();
    T* tmp1 = new T[this->rows];
    T* h = new T[this->rows];
    T* s = new T[this->rows];
    T* t = new T[this->rows];
    T* x1 = new T[this->rows];

    T rho0 = 1.0;
    T rho1;
    T alp = 1.0;
    T w0 = 1.0;
    T w1;
    T beta;
    // get the initial residue
    this->matVecMult(x0, Ax0);
    daxpy(-1.0, Ax0, rhs, r0, this->rows);
    memcpy(r0_hat, r0, this->rows*sizeof(T));
    for(i=0; i<max_itr; i++){
        // Apply gram schmidt orthonormalization to get A-orthogonal sequence
        rho1 = inner(this->rows, r0_hat, r0);
        beta = (rho1/rho0)/(alp/w0);
        // Using gradient descend to minimize least square loss
        daxpy(-w0, v0, p0, tmp1, this->rows);
        daxpy(beta, tmp1, r0, p1, this->rows);
        this->matVecMult(p1, v1);
        alp = rho1/inner(this->rows, r0_hat, v1);
        // get the updated solution, check if accuracy enough
        daxpy(alp, p1, x0, h, this->rows);
        if (this->check_res(h, rhs)<tol_abs){
            memcpy(x1, h, this->rows * sizeof(T));
            break;
        }
        // Bi-orthogonal sequence
        daxpy(-alp, v1, r0, s, this->rows);
        this->matVecMult(s, t);
        w1 = inner(this->rows, t, s)/inner(this->rows, t, t);
        // get the updated solution, check if accuracy enough
        daxpy(w1, s, h, x1, this->rows);
        if (this->check_res(x1, rhs)<tol_abs){
            break;
        }
        daxpy(-w1, t, s, r0, this->rows);
        // iteration
        rho0 = rho1;
        w0 = w1;

        memcpy(p0, p1, this->rows * sizeof(T));
        memcpy(v0, v1, this->rows * sizeof(T));
        memcpy(x0, x1, this->rows * sizeof(T));
    }

    memcpy(x0, x1, this->rows * sizeof(T));
}

// Solver implementing the Jacobi method - Giles Matthews.
// Iterative method to compute problems where A has no zeros on main diagonal.
// Using an intial guess, each diagonal element is approximated 
// and solved for until a convergence tolerance is satisfied.
template <class T>
Matrix<T>* Matrix<T>::Jacobi(Matrix<T>& b, int dp, int max_iter)
{
    bool solved = false;
    int n = this->rows;

    // Initialise matrix, Dinv, to hold reciprocal diagonal values
    auto* zeros = new double[n * n]();
    auto* x = new double[n]();
    auto *Dinv = new Matrix<double>(n, n, true);
    auto *x_old = new Matrix<double>(n, 1, x);
    auto *x_new = new Matrix<double>(n, 1, false);
    auto *M = new Matrix<double>(n, 1, zeros);
    auto *temp = new Matrix<double>(n, 1, false);
    double err;
    int count_iter = 0;

    // Generate D and LU matrices
    // For each row we will copy the reciprocal of the diagonal value into Dinv
    // and remove the diagonal from A. This will leave us with a diagonal matrix, D, and
    // a Matrix, LU, composed of the strict upper and lower triangular parts of A.
    for (int i = 0; i < n; i++)
    {
        // Copy diagonal values into Dinv matrix and take reciprocals to give Dinv
        Dinv->values[i + i * n] = 1 / this->values[i + i * n];

        // Replace diagonal values with zeros to leave LU matrix.
        this->values[i + i * n] = 0;
    }
    
    // Iterate through until x(k+1) and x(k) reached a desired convergence.
    // Convergence is measured as the norm of x(k+1)-x(k), the tolerance is to 0.001.
    //     x(k+1) = Dinv * M
    // Where M = [(-L-U)*x(k) + b]
    while (solved == false)
    {
        this->matVecMult(*x_old, *temp);

        for (int i = 0; i < n; i++)
        {
            M->values[i] = b.values[i] - temp->values[i];
        }

        // ~~ Generate x_new ~~
        Dinv->matVecMult(*M, *x_new);

        // ~~ Calculate convergence error ~~
        err = 0;
        for (int i = 0; i < n; i++)
        {
            err += pow(x_new->values[i] - x_old->values[i], 2);
            x_old->values[i] = x_new->values[i];
        }

        // Check if the convergence tolerance is satisfied
        // Stop iterations and print results.
        if (sqrt(err) < pow(10, -1 * dp))
        {
            solved = true;
        }

        if (count_iter == max_iter)
        {
            std::cout << std::endl << "Iteration limit reached (" << max_iter << ")" << std::endl;
            solved = true;
        }
        count_iter += 1;
    }
    return x_new;

    delete Dinv;
    delete x_old;
    delete x_new;
    delete M;
    delete temp;
}

// Solver implementing the Conjugate Gradient method - Giles Matthews.
// Iterative method for when A is symmetrical.
// The method calculates the direction of steepest slope, s,
// from current guess to vector b and computes an appropriate
// step size, alpha, to converge on solution in the s direction.
template <class T>
Matrix<T>* Matrix<T>::ConjGrad(Matrix<T>& b, int dp)
{
    int n = this->rows;

    // Generate output vector, x with inital guess of zeros
    auto *x = new Matrix<double>(n, 1, true);
    for (int i = 0; i < n; i++)
    {
        x->values[i] = 0;
    }

    // Generate vectors to hold intermediate values
    auto *r = new Matrix<double>(n, 1, true);
    auto *s = new Matrix<double>(n, 1, true);
    auto *Ax = new Matrix<double>(n, 1, true);
    auto *As = new Matrix<double>(n, 1, true);

    // Generate constants used in iterations
    double sr, sAs, alpha, rAs, beta, err;

    this->matVecMult(*x, *Ax);

    // Calculate inital residual, r0, and search direction, s0.
    // r0 = b - Ax0
    // s0 = r0     
    for (int i = 0; i < n; i++)
    {
        r->values[i] = b.values[i] - Ax->values[i];
        s->values[i] = r->values[i];
    }

    // Iterate through solver n times where n is the number of unknowns.
    for (int iter = 0; iter < n; iter++)
    {
        sr = 0;
        sAs = 0;
        rAs = 0;
        err = 0;

        this->matVecMult(*s, *As);

        // Calcuate the step-length, alpha
        alpha = s->vectordot(*r) / s->vectordot(*As);

        // Step in direction s with step-size alpha to calculate new x
        for (int i = 0; i < n; i++)
        {
            x->values[i] += s->values[i] * alpha;
        }

        this->matVecMult(*x, *Ax);   // New value of Ax

        // Calculate new residual vector and the corressponding error
        for (int i = 0; i < n; i++)
        {
            r->values[i] = b.values[i] - Ax->values[i];
            err += pow(r->values[i], 2);
        }

        // If error tolerance is satisified then exit loop
        if (sqrt(err) < pow(10, -1 * dp))
        {
            break;
        }
        else
        {
            beta = (-1 * r->vectordot(*As)) / s->vectordot(*As);
            for (int i = 0; i < n; i++)
            {
                s->values[i] = r->values[i] + beta * s->values[i];
            }
        }
    }
    return x;

    delete x;
    delete r;
    delete s;
    delete Ax;
    delete As;
}

// Solver implementing the Gauss Elimination method - Giles Matthews.
// The solution is built into the b vector rather than
// a solution vector, x, being returned.
// There are two phases: Elimination phase and Back-substitution phase
template <class T>
void Matrix<T>::GaussElim(Matrix<T>& b, int dp)
{
    int n = this->rows;
    double lambda;

    // Elimination phase
    // Iterate through each transformation row, i.
    // Use row echelon method to compute upper triangular matrix.
    for (int i = 0; i < n-1; i++)  // Pivot row
    {
        for (int j = i+1; j < n; j++) // Transformation row
        {
            // If leading value already zero. no further calculation required.     
            if (this->values[i + j * n] != 0) 
            {
                // Calulate ratio, lambda, between leading values of
                // Transformation row and pivot row.
                lambda = this->values[i + j * n] / this->values[i + i * n];

                // For each value in subsequent rows:
                // Aik = Aik - lambda * A(i-1)k
                // b(k) = b(k) - lambda * b(k-1)
                for (int k = 0; k < n; k++)
                {
                    this->values[k + j * n] -= lambda * this->values[k + i * n];
                }
                b.values[j] -= lambda * b.values[i];
            }
        }
    }

    // Back-Substitution phase
    // Calcualte solution from triangular matrix.
    for (int i = n-1; i >= 0; i--)
    {
        double row_product = 0;

        for (int j = i + 1; j < n; j++)
        {
            row_product += this->values[i * n + j] * b.values[j];
        }
        b.values[i] = (b.values[i] - row_product) / this->values[i * n + i];
    }
}

// Gauss Seidel iterative solver - Raha Moosavi
template <class T>
Matrix<T>* Matrix<T>::GauSeidel(Matrix<T>& b, int dp, int max_iter)
{
    int n = this->rows;

    // Initialise matrix;
    auto *x = new Matrix<double>(n, 1, true);
    auto *Ex = new Matrix<double>(n, 1, true);

    // Set intial guess to zero
    for (int i = 0; i < n; i++)
    {
        x->values[i] = 0;
    }
    
    // Iterate through until x(k+1) and x(k) reached a desired convergence.
    // Convergence is measured as the norm of x(k+1)-x(k), the tolerance is to 0.001.
    //    if i = 0 , x_new[i] = x_old *(1-omega) + omega/a[i][i] *(b[i] - sum (a[i][j] * x_old[j]))
    //    if 0 < i < n-1, for (j < i) x_new[i] = x_old *(1-omega) + omega/a[i][i] *(b[i] - sum (a[i][i] * x_old[j]) 
    //    if 0 < i < n-1, for (j < i) x_new[i] = x_old *(1-omega) + omega/a[i][i] *(b[i] - sum (a[i][i] * x_new[j]) 
    //    if i = n-1 , x_new[i] = x_old *(1-omega) + omega/a[i][i] *(b[i] - sum (a[i][j] * x_new[j]))

    for (int iter = 0; iter < max_iter; iter ++)
    {
        //x->printValues();
        for (int i = 0; i < n; i++)
        {     
            double sum = 0;
            if (i==0)
            {
                //std::cout << sum << "\n";
                for (int j = 1; j < n; j++)
                {
                    sum +=  this->values[j + i *n] * x->values[j];
                    //std::cout << sum << "\n";
                }
            }    
            else if (i >= 1 && i <= n-2)
            {
                sum = 0;
                for (int j = 0; j < n ; j++)
                {
                   if  (j < i) 
                   {
                     sum += this->values[j + i * n] * x->values[j];
                   }
                   else if (j > i)
                   {
                    sum +=  this->values[j + i * n] * x->values[j];
                   }
                }               
            }
            else
            {
                sum = 0;
                for (int j = 0; j < n -1 ; j++)
                {
                    sum += this->values[j + i * n] * x->values[j]; 
                }
            }
        x->values[i] = (b.values[i] - sum)/ (this->values[i + i * n]); 
        }

        // ~~ Calculate convergence error ~~

        this->matVecMult(*x, *Ex); 

        double err = 0;
        for (int i = 0; i < n; i++)
        {
            err += pow(Ex->values[i] - b.values[i], 2);
        }

        // Check if the convergence tolerance is satisfied
        // Stop iterations and print results.
        if (sqrt(err) < pow(10, -1 * dp))
        {
             break;
        }
    }
    return x;
    delete x;
    delete Ex;
}

// Successive Over-Relaxation solver - Raha Moosavi
template <class T>
Matrix<T>* Matrix<T>::SOR(double omega, Matrix<T>& b, int dp, int max_iter)
{
    int n = this->rows;

    // Initialise matrix;
    auto *x = new Matrix<double>(n, 1, true);
    auto *Bx = new Matrix<double>(n, 1, true);

    // Set intial guess to zero
    for (int i = 0; i < n; i++)
    {
        x->values[i] = 0;
    }
    
    // Iterate through until x(k+1) and x(k) reached a desired convergence.
    // Convergence is measured as the norm of x(k+1)-x(k), the tolerance is to 0.001.
    //    if i = 0 , x_new[i] = 1/a[i][i] *(b[i] - sum (a[i][j] * x_old[j]))
    //    if 0 < i < n-1, fof (j < i) x_new[i] = 1/a[i][i] *(b[i] - sum (a[i][i] * x_old[j]) 
    //    if 0 < i < n-1, fof (j < i) x_new[i] = 1/a[i][i] *(b[i] - sum (a[i][i] * x_new[j]) 
    //    if i = n-1 , x_new[i] = 1/a[i][i] *(b[i] - sum (a[i][j] * x_new[j]))

    for (int iter = 0; iter < max_iter; iter ++)
    {
        for (int i = 0; i < n; i++)
        {     
            double sum = 0;
            if (i==0)
            {
                for (int j = 1; j < n; j++)
                {
                    sum +=  this->values[j + i *n] * x->values[j];
                }
            }    
            else if (i >= 1 && i <= n-2)
            {
                sum = 0;
                for (int j = 0; j < n ; j++)
                {
                   if  (j < i) 
                   {
                     sum += this->values[j + i * n] * x->values[j];
                   }
                   else if (j > i)
                   {
                    sum +=  this->values[j + i * n] * x->values[j];
                   }
                }               
            }
            else
            {
                sum = 0;
                for (int j = 0; j < n -1 ; j++)
                {
                    sum += this->values[j + i * n] * x->values[j]; 
                }
            }

         x->values[i] = (1-omega) * x->values[i] + ((b.values[i] - sum) * (omega/ this->values[i + i * n])); 
        }

        // ~~ Calculate convergence error ~~
        this->matVecMult(*x, *Bx); 

        double err = 0;
        for (int i = 0; i < n; i++)
        {
            err += pow(Bx->values[i] - b.values[i], 2);
        }

        // Check if the convergence tolerance is satisfied
        if (sqrt(err) < pow(10, -1 * dp))
        {
             break;
        }
    }
    return x;
    delete x;
    delete Bx;
}
