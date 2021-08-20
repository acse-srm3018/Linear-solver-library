# C++ Linear Solver library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* [Basic Information](#basic-information)
    * [Abstract](#abstract)


# Basic Information

This project aims to develop an efficient C++ solver for linear systems, or the Ax = b problems. 

An overview of the files is provided below.

- `src/` contains the header and source files for the library Matrix.
- `.gitignore` is the file specifying which files should be ignored in Git.
- `LICENSE.txt` is the MIT license.
- `README.md` contains basic information for the repository and detailed information for how to compile and reproduce the results.

## Abstract
Almost every topic in engineering, requires the solution of a linear (or matrix) system of the form

𝐴𝒙=𝒃,
 
where for a given  𝑚×𝑛  matrix  𝐴  and a given  𝑚×1  right hand side vector  𝒃  we want to find an  𝑛×1  vector  𝒙 .

[Remember that the product of an  𝑚×𝑛  and an  𝑛×𝑝  matrix (in that order) results in an  𝑚×𝑝  matrix]

Often, but not always, we will be considering a square problem, i.e. where  𝑚=𝑛 .

For many methods or algorithms the solution of linear systems such as this also turns out to be the most costly component of that method. And also where sensible vs stupid algorithm choices can lead to terrible run times and even instability (e.g. not being able to obtain a solution at all).

Hence, it is vitally important for us to understand the methods are available to us to solve this problem accurately and efficiently.

## How to Use Library
A demonstration of the solvers can be run by compiling and running the main_run.cpp file in the src directory. This script generates example linear systems and passes them into the different linear solvers. 

## Summary of Solver Methods
Solvers for systems stored in a dense format are methods in the Matrix class. Input parameters in brackets.
- Jacobi(Matrix A, decimal point precision, max iterations)
- Conjugate Gradient(Matrix A, decimal point precision)
- Gauss Elimination(Matrix A, decimal point precision)
- Gauss-Seidel(Matrix A, decimal point precision, max iterations)
- SOR(omega, Matrix A, decimal point precision, max iterations)
- GMRES(x0, rhs, max iteration, max internal iteration, total absolute error, total reflexive error)
- BiCG(x0, rhs, max iteration, total absolute error)

Solvers for systems stored in a compressed sparse row format are methods in the CSRMatrix class. Input parameters in brackets.
- GMRES(x0, rhs, max iteration, max internal iteration, total absolute error, total reflexive error)
