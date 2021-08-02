// Header file for Matrix class used to store matrices
// in a dense format and call methods to perfrom matrix operations
// and solve linear systems.

#pragma once
#include<fstream>

template <class T>
class Matrix
{
public:

    // ~~~~~ Constructors ~~~~~~~
    // constructor where we want the matrix class to (possibly) own the memory
    Matrix(int rows, int cols, bool preallocate);
    // constructor where the matrix class doesn't own the memory
    Matrix(int rows, int cols, T *values_ptr);
    // constructor from file
    Matrix(int rows, int cols, char* file_name);

    // destructor
    virtual ~Matrix();

    // printing out methods
    void printValues();
    virtual void printMatrix();
    // write file
    void write_matrix(char* file_name);

    // Matrix Matrix multiplication
    // Use '&' to pass by reference and avoid copying.
    virtual void matMatMult(Matrix& left, Matrix& output);

    // Matrix Vector multiplication
    // For Ax = b where A is a matrix and x is a vector. 
    // b is the output solution vector.
    virtual void matVecMult(Matrix& vect, Matrix& output);
    // matvecmul for arrays inputs
    virtual void matVecMult(T vect[], T output[]);

    // Calculate the dot product of two vectors stored as 
    // Matrix objects. 
    // Output is a scalar stored as a double
    double vectordot(Matrix<T>& b);

    // Functions to solve Ax = b for x
    void dense_GMRES(T x0[], T rhs[], int max_itr, int max_int_itr, T tol_abs, T tol_ref );

    T check_res(T x[], T rhs[]);
    void BiCG(T x0[], T rhs[], int max_itr, T tol_abs);

    Matrix<T>* Jacobi(Matrix<T>& b, int dp, int max_iter);

    Matrix<T>* ConjGrad(Matrix<T>& b, int dp);

    Matrix<T>* GauSeidel(Matrix<T>& b, int dp, int max_iter);

    Matrix<T>* SOR(const double omega, Matrix<T>& b, int dp, int max_iter);

    void GaussElim(Matrix<T>& b, int dp);

    // sizes of the matrix
    T *values = nullptr;
    int rows = -1; // nonsense default values
    int cols = -1;

protected:

    bool preallocated = false;

private:

    int size_of_values = -1;
    
};