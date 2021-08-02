// Header file for CSRMatrix class used to store matrices
// in a compressed sparse row format and call methods to perfrom matrix operations
// and solve linear systems.
#pragma once
#include "Matrix.h"

template <class T>
class CSRMatrix : public Matrix<T>
{
public:

	// constructor where we want to preallocate ourselves
	CSRMatrix(int rows, int cols, int nnzs, bool preallocate);
	// constructor where we already have allocated memory outside
	CSRMatrix(int rows, int cols, int nnzs, T* values_ptr, int* row_position, int* col_index);
	// constructor from files
	CSRMatrix(int rows, int cols, int nnzs, char* row_position_file_name, char* col_index_file_name, char* values_file_name);
	// destructor
	~CSRMatrix();

	// Print out the values in our matrix
	virtual void printMatrix();

	// Perform some operations with our matrix
	void matMatMult(CSRMatrix<T>& mat_left, CSRMatrix<T>& output);
	// Perform some operations with our matrix
	void matVecMult(T* input, T* output);
	// get element from row and col index
	T get(int row, int col) const;
	// GMRES
	void sparse_GMRES(T* x0, T* rhs, int max_itr, int max_int_itr, T tol_abs, T tol_ref);
	// iLU
    void ilu(int diag_ind[], T b[], T res[]);
    // write file
    void write_matrix(char* row_position_file_name, char* col_index_file_name, char* values_file_name);
	// Explicitly using the C++11 nullptr here
	int* row_position = nullptr;
	int* col_index = nullptr;


	// How many non-zero entries we have in the matrix
	int nnzs = -1;

	// Private variables - there is no need for other classes 
	// to know about these variables
private:

};