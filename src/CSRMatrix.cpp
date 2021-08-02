// Source file for CSRMatrix class used to store matrices
// in a compressed sparse row format and call methods to perfrom matrix operations
// and solve linear systems.

#include <iostream>
#include "CSRMatrix.h"
#include <cmath>
#include <fstream>

template <class T>
void CSRMatrix<T>::ilu(int diag_ind[], T b[], T res[]){
    int i, j;
    T *tmp = new T[this->rows];
    memcpy(tmp, b, this->rows*sizeof(T));
    for ( i = 1; i < this->rows; i++ )
    {
        for ( j = this->row_position[i]; j < diag_ind[i]; j++ )
        {
            tmp[i] -= this->values[j] * tmp[this->col_index[j]];
        }
    }
    for ( i = this->rows - 1; 0 <= i; i-- )
    {
        for ( j = diag_ind[i] + 1; j < this->row_position[i+1]; j++ )
        {
            tmp[i] -= this->values[j] * tmp[this->col_index[j]];
        }
        tmp[i] /= this->values[diag_ind[i]];
    }
    memcpy(res, tmp, this->rows*sizeof(T));
    delete [] tmp;
}

// Constructor - using an initialisation list here
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate) : Matrix<T>(rows, cols, false), nnzs(nnzs)
{
	// If we don't pass false in the initialisation list base constructor, it would allocate values to be of size
	// rows * cols in our base matrix class
	// So then we need to set it to the real value we had passed in
	this->preallocated = preallocate;

	// If we want to handle memory ourselves
	if (this->preallocated)
	{
		// Must remember to delete this in the destructor
		this->values = new T[this->nnzs];
		this->row_position = new int[this->rows + 1];
		this->col_index = new int[this->nnzs];
	}
}

// Constructor - now just setting the value of our T pointer
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T* values_ptr, int* row_position, int* col_index) : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_position), col_index(col_index)
{}
// Constructor from files
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, char* row_position_file_name, char* col_index_file_name, char* values_file_name): Matrix<T>(rows, cols, false), nnzs(nnzs)
{
    std::ifstream row_position_file, col_index_file, values_file;
    int i;

    this->preallocated = true;
    this->values = new T[this->nnzs];
    this->row_position = new int[this->rows + 1];
    this->col_index = new int[this->nnzs];

    row_position_file.open(row_position_file_name);
    col_index_file.open(col_index_file_name);
    values_file.open(values_file_name);

    for (i=0; i<nnzs; i++){
        values_file >> this->values[i];
        col_index_file >> this->col_index[i];
    }
    for (i=0; i<rows+1; i++){
        row_position_file >> this->row_position[i];
    }

    row_position_file.close();
    col_index_file.close();
    values_file.close();
}
// destructor
template <class T>
CSRMatrix<T>::~CSRMatrix()
{
	// Delete the values array
	if (this->preallocated) {
		delete[] this->row_position;
		delete[] this->col_index;
	}
	// The super destructor is called after we finish here
	// This will delete this->values if preallocated is true
}

// Explicitly print out the values in values array as if they are a matrix
template <class T>
void CSRMatrix<T>::printMatrix()
{
	std::cout << "Printing matrix" << std::endl;
	std::cout << "Values: ";
	for (int j = 0; j < this->nnzs; j++)
	{
		std::cout << this->values[j] << " ";
	}
	std::cout << std::endl;
	std::cout << "row_position: ";
	for (int j = 0; j < this->rows + 1; j++)
	{
		std::cout << this->row_position[j] << " ";
	}
	std::cout << std::endl;
	std::cout << "col_index: ";
	for (int j = 0; j < this->nnzs; j++)
	{
		std::cout << this->col_index[j] << " ";
	}
	std::cout << std::endl;
}

// Do a matrix-vector product
// output = this * input
template<class T>
void CSRMatrix<T>::matVecMult(T* input, T* output)
{
	if (input == nullptr || output == nullptr)
	{
		std::cerr << "Input or output haven't been created" << std::endl;
		return;
	}

	// Set the output to zero
	for (int i = 0; i < this->rows; i++)
	{
		output[i] = 0.0;
	}

	// Loop over each row
	for (int i = 0; i < this->rows; i++)
	{
		// Loop over all the entries in this col
		for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
		{
			// This is an example of indirect addressing
			// Can make it harder for the compiler to vectorise!
			output[i] += this->values[val_index] * input[this->col_index[val_index]];

		}
	}
}


// Do matrix matrix multiplication
// output = mat_left * this
template <class T>
void CSRMatrix<T>::matMatMult(CSRMatrix<T>& mat_left, CSRMatrix<T>& output)
{

	// Check our dimensions match
	if (this->cols != output.cols)
	{
		std::cerr << "Input dimensions for matrices don't match" << std::endl;
		return;
	}

	// Check if our output matrix has had space allocated to it
	int preal_flag = 1;
	if (output.values != nullptr)
	{
		// Check our dimensions match
		if (this->rows != mat_left.cols || mat_left.rows != output.rows)
		{
			std::cerr << "Input dimensions for matrices don't match" << std::endl;
			return;
		}
	}
	// The output hasn't been preallocated, so we are going to do that
	else
	{
		std::cerr << "OUTPUT HASN'T BEEN ALLOCATED" << std::endl;
		preal_flag = 0;
		output.row_position = new int[mat_left.rows + 1];
	}
	T* values_tmp = new T[mat_left.rows * this->cols];
	int* cols_tmp = new int[mat_left.rows * this->cols];
	int row_nz = 0;
	int col_ind = 0;
	output.nnzs = 0;
	output.row_position[0] = 0;
	for (int i = 0; i <= mat_left.rows; i++) {
		row_nz = 0;
		for (int j = 0; j <= this->cols; j++) {
			T a = T();
			for (int k = 0; k <= mat_left.cols; k++) {
				a = a + mat_left.get(i, k) * this->get(k, j);
			}
			if (a != T())
			{
                output.nnzs += 1;
			    cols_tmp[output.nnzs-1] = j;
				values_tmp[output.nnzs-1] = a;
				col_ind += 1;
				row_nz += 1;
			}
		}
		output.row_position[i + 1] = output.row_position[i] + row_nz;
	}
	if (preal_flag == 0)
	{
		output.preallocated = true;
		output.values = new T[output.nnzs];
		output.col_index = new int[output.nnzs];
	}
	for (int i = 0; i < output.nnzs; i++) {
		output.values[i] = values_tmp[i];
		output.col_index[i] = cols_tmp[i];
	}
	delete[] values_tmp;
	delete[] cols_tmp;
}

// get i-th row, j-th column element
template <class T>
T CSRMatrix<T>::get(int row, int col) const
{
	int cur_col;
	for (int pos = this->row_position[row]; pos < this->row_position[row + 1]; ++pos) {
		cur_col = this->col_index[pos];

		if (cur_col == col) {
			return this->values[pos];

		}
		else if (cur_col > col) {
			break;
		}
	}

	return T();
}

// GMRES
// See: Iterative Methods for Sparse Linear Systems (2000) by Yousef Saad
// Code reference: https://people.sc.fsu.edu/~jburkardt/cpp_src/mgmres/mgmres.html
template <class T>
void CSRMatrix<T>::sparse_GMRES(T x0[], T rhs[], int max_itr, int max_int_itr, T tol_abs, T tol_ref )
{
    int i, j, k, itr, itr_used, k_tmp, j_diff, i_temp, j1, j2, j_row, j_diag;
    T dot_prod, d_temp, mu_m, rho_i, rho_tol, l_tmp, h_tmp;
    T *givens_cos, *v1_tmp, *v2_tmp, *ele_diag, *res, *givens_sin, *v3_tmp, *iter_tmp;
    int *diag_ind, *ele_diag_tmp;
    T dlt = 1.0e-03;
    itr_used = 0;

    givens_cos = new T[max_int_itr+1];
    v1_tmp = new T[max_int_itr+1];
    v2_tmp = new T[(max_int_itr+1)*max_int_itr];
    ele_diag = new T[this->row_position[this->rows]+1];
    res = new T[this->rows];
    givens_sin = new T[max_int_itr+1];
    diag_ind = new int[this->rows];
    v3_tmp = new T[(max_int_itr+1)*this->rows];
    iter_tmp = new T[max_int_itr+1];

    // Pre-process the matrix
    for (i = 0; i < this->rows; i++ )
    {
        j1 = this->row_position[i];
        j2 = this->row_position[i+1];
        j_diff = j2 - j1;

        for (k = 1; k < j_diff; k++ )
        {
            for (j = j1; j < j2 - k; j++ )
            {
                if ( this->col_index[j+1] < this->col_index[j] )
                {
                    i_temp = this->col_index[j+1];
                    this->col_index[j+1] =  this->col_index[j];
                    this->col_index[j] =  i_temp;

                    d_temp = this->values[j+1];
                    this->values[j+1] =  this->values[j];
                    this->values[j] = d_temp;
                }
            }
        }
    }

    // Take the diagonal
    for (i = 0; i < this->rows; i++ )
    {
        diag_ind[i] = -1;
        j1 = this->row_position[i];
        j2 = this->row_position[i+1];

        for (j = j1; j < j2; j++ )
        {
            if ( this->col_index[j] == i )
            {
                diag_ind[i] = j;
            }
        }

    }
    ele_diag_tmp = new int[this->rows];
    for (k = 0; k < this->nnzs; k++ )
    {
        ele_diag[k] = this->values[k];
    }
    // get preconditioner of the matrix by the diagonal
    for (i = 0; i < this->rows; i++ )
    {
        for (j = 0; j < this->rows; j++ )
        {
            ele_diag_tmp[j] = -1;
        }

        for (k = this->row_position[i]; k <= this->row_position[i+1] - 1; k++ )
        {
            ele_diag_tmp[this->col_index[k]] = k;
        }

        j = this->row_position[i];
        do{
            j_row = this->col_index[j];
            if ( i <= j_row )
            {
                break;
            }
            l_tmp = ele_diag[j] * ele_diag[diag_ind[j_row]];
            ele_diag[j] = l_tmp;
            for (int j_tmp = diag_ind[j_row] + 1; j_tmp <= this->row_position[j_row+1] - 1; j_tmp++ )
            {
                j_diag = ele_diag_tmp[this->col_index[j_tmp]];
                if ( j_diag != -1 )
                {
                    ele_diag[j_diag] -= l_tmp * ele_diag[j_tmp];
                }
            }
            j++;
        } while (j <= this->row_position[i+1] - 1 );

        diag_ind[i] = j;

        ele_diag[j] = 1.0 / ele_diag[j];
    }

    for ( k = 0; k < this->rows; k++ )
    {
        ele_diag[diag_ind[k]] = 1.0 / ele_diag[diag_ind[k]];
    }
    // iteration
    for ( itr = 0; itr < max_itr; itr++ )
    {
        // Arnoldi's method (FOM)
        // get residue r0=b-Ax0
        this->matVecMult(x0, res);

        for ( i = 0; i < this->rows; i++ )
        {
            res[i] = rhs[i] - res[i];
        }
        // iLU precondition
        this->ilu(diag_ind, res, res);
        // Gram–Schmidt process
        rho_i = sqrt ( inner ( this->rows, res, res ) );

        if ( itr == 0 )
        {
            rho_tol = rho_i * tol_ref;
        }
        // Normalization
        for ( i = 0; i < this->rows; i++ )
        {
            v3_tmp[i] = res[i] / rho_i;
        }

        v1_tmp[0] = rho_i;
        for ( i = 1; i < max_int_itr + 1; i++ )
        {
            v1_tmp[i] = T();
        }

        for ( i = 0; i < max_int_itr + 1; i++ )
        {
            for ( j = 0; j < max_int_itr; j++ )
            {
                v2_tmp[i*(max_int_itr)+j] = T();
            }
        }
        // upper Hessenberg matrix for orthogonal method of solving least square
        for ( k = 0; k < max_int_itr; k++ )
        {
            k_tmp = k;
            // Do QR decomposition
            this->matVecMult(v3_tmp+k*this->rows, v3_tmp+(k+1)*this->rows);
            // iLU precondition
            this->ilu(diag_ind, v3_tmp+(k+1)*this->rows, v3_tmp+(k+1)*this->rows);
            // inner product
            dot_prod = sqrt ( inner ( this->rows, v3_tmp+(k+1)*this->rows, v3_tmp+(k+1)*this->rows ) );
            // orthogonalization
            for ( j = 0; j <= k; j++ )
            {
                v2_tmp[j*max_int_itr+k] = inner ( this->rows, v3_tmp+(k+1)*this->rows, v3_tmp+j*this->rows );
                for ( i = 0; i < this->rows; i++ )
                {
                    v3_tmp[(k+1)*this->rows+i] -= v2_tmp[j*max_int_itr+k] * v3_tmp[j*this->rows+i];
                }
            }
            v2_tmp[(k+1)*max_int_itr+k] = sqrt ( inner ( this->rows, v3_tmp+(k+1)*this->rows, v3_tmp+(k+1)*this->rows ) );
            // Householder orthogonalization
            if ( ( dot_prod + dlt * v2_tmp[(k+1)*max_int_itr+k]) == dot_prod ){
                for ( j = 0; j < k + 1; j++ )
                {
                    h_tmp = inner ( this->rows, v3_tmp+(k+1)*this->rows, v3_tmp+j*this->rows );
                    v2_tmp[j*max_int_itr+k] += h_tmp;
                    for ( i = 0; i < this->rows; i++ )
                    {
                        v3_tmp[(k+1)*this->rows+i] -= h_tmp * v3_tmp[j*this->rows+i];
                    }
                }
                v2_tmp[(k+1)*max_int_itr+k] = sqrt ( inner ( this->rows, v3_tmp+(k+1)*this->rows, v3_tmp+(k+1)*this->rows ) );
            }

            if ( v2_tmp[(k+1)*max_int_itr+k] != T() )
            {
                for ( i = 0; i < this->rows; i++ )
                {
                    v3_tmp[(k+1)*this->rows+i] /= v2_tmp[(k+1)*max_int_itr+k];
                }
            }
            // Givens rotation. Orthogonal method to solve the least square
            if ( 0 < k )
            {
                for ( i = 0; i < k + 2; i++ )
                {
                    iter_tmp[i] = v2_tmp[i*max_int_itr+k];
                }
                for ( j = 0; j < k; j++ )
                {
                    Givens ( givens_cos[j], givens_sin[j], j, iter_tmp );
                }
                for ( i = 0; i < k + 2; i++ )
                {
                    v2_tmp[i*max_int_itr+k] = iter_tmp[i];
                }
            }
            // Givens rotation.
            mu_m = sqrt ( v2_tmp[k*max_int_itr+k] * v2_tmp[k*max_int_itr+k] + v2_tmp[(k+1)*max_int_itr+k] * v2_tmp[(k+1)*max_int_itr+k] );
            givens_cos[k] = v2_tmp[k*max_int_itr+k] / mu_m;
            givens_sin[k] = -v2_tmp[(k+1)*max_int_itr+k] / mu_m;
            v2_tmp[k*max_int_itr+k] = givens_cos[k] * v2_tmp[k*max_int_itr+k] - givens_sin[k] * v2_tmp[(k+1)*max_int_itr+k];
            v2_tmp[(k+1)*max_int_itr+k] = T();
            Givens ( givens_cos[k], givens_sin[k], k, v1_tmp );

            rho_i = fabs ( v1_tmp[k+1] );

            itr_used ++;

            if ( rho_i <= rho_tol && rho_i <= tol_abs )
            {
                break;
            }
        }
        // Do iteration
        k = k_tmp;

        iter_tmp[k] = v1_tmp[k] / v2_tmp[k*max_int_itr+k];
        for ( i = k - 1; 0 <= i; i-- )
        {
            iter_tmp[i] = v1_tmp[i];
            for ( j = i + 1; j < k + 1; j++ )
            {
                iter_tmp[i] -= v2_tmp[i*max_int_itr+j] * iter_tmp[j];
            }
            iter_tmp[i] /= v2_tmp[i*max_int_itr+i];
        }
        for ( i = 0; i < this->rows; i++ )
        {
            for ( j = 0; j < k + 1; j++ )
            {
                x0[i] += v3_tmp[j*this->rows+i] * iter_tmp[j];
            }
        }

        if ( rho_i <= rho_tol && rho_i <= tol_abs )
        {
            break;
        }
    }
    delete [] ele_diag_tmp;
    delete [] givens_cos;
    delete [] v1_tmp;
    delete [] v2_tmp;
    delete [] ele_diag;
    delete [] res;
    delete [] givens_sin;
    delete [] diag_ind;
    delete [] v3_tmp;
    delete [] iter_tmp;
}

template <class T>
void CSRMatrix<T>::write_matrix(char* row_position_file_name, char* col_index_file_name, char* values_file_name){
    int i;
    std::ofstream row_position_file, col_index_file, values_file;
    row_position_file.open(row_position_file_name);
    col_index_file.open(col_index_file_name);
    values_file.open(values_file_name);

    for(i=0; i<this->nnzs-1; i++){
        col_index_file << this->col_index[i] << " ";
        values_file << this->values[i] << " ";
    }
    col_index_file << this->col_index[this->nnzs-1];
    values_file << this->values[this->nnzs-1];

    for (i = 0; i < this->rows; i++)
    {
        row_position_file << this->row_position[i] << " ";
    }
    row_position_file << this->row_position[this->rows];

    row_position_file.close();
    col_index_file.close();
    values_file.close();
}