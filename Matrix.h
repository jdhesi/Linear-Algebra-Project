#ifndef MATRIXDEF
#define MATRIXDEF

#include <cmath>
#include "Exception.h"
#include <tuple>
class Vector; //forward declaration 
class Matrix
{
protected: 
	double* matrixVal = nullptr;
	int rows;
	int columns;
public:

	// Used to allocate space 
	void createMatrix(int rows, int columns);
	// Constructors
	Matrix();								  // Default constructor 
	Matrix(int r, int c);					  // Constructor given rows and columns
	Matrix(int size) :Matrix(size, size) {};  // Constructor given a size
	Matrix(const Matrix& m);				  // Copy constructor
	// Destructor 
	~Matrix();								

	// Operators
	Matrix& operator=(const Matrix &m);	// equality
	double& operator()(int i, int j);	// indexing
	friend Matrix operator+(const Matrix& m1, const Matrix& m2);
	friend Matrix operator-(const Matrix& m1, const Matrix& m2);
	friend Matrix operator*(const Matrix& m1, const Matrix& m2);
	friend Matrix operator*(const double& a, const Matrix& m);
	friend Matrix operator*(const Matrix& m, const double& a);
	friend Matrix operator/(const Matrix& m, const double& a);
	friend Matrix operator-(const Matrix& m);
	friend Vector operator*(const Matrix& m, const Vector& v);
	friend Vector operator/(const Matrix& A, const Vector& b);
	// printing
	friend std::ostream& operator<<(std::ostream& output, const Matrix& m);	// printing
	// Functions 
	friend Matrix eye(int n);
	friend Vector getColumn(Matrix A, int c);	
	friend Matrix getRow(Matrix A, int c);	
	friend Matrix transpose(const Matrix& m);
	friend Vector gmres(const Matrix& A, const Vector& b, const Vector& x0, double tol, int maxits);
	friend std::tuple<Matrix, Matrix> qr(const Matrix& A);
	friend Matrix eigMatrix(const Matrix& A);
	friend Vector eig(const Matrix& A);
	//friend Vector eigP(const Matrix& A);
	//friend Matrix eigPractical (const Matrix& A);
	friend Matrix inverse(const Matrix& A);
	friend Matrix hessenberg(const Matrix& A);
	friend Vector convertMatrixToVector(const Matrix& A); //backend func
	friend Matrix rand(int n);
	Matrix& swap_rows(int a, int b);
	Matrix& setColumn(int col, Vector& v);
	Matrix& addRows(int r);
	Matrix& addCols(int c);

};
// Accesible functions
Matrix rand(int n);
Vector eigP(const Matrix& A);
Matrix eigPractical(const Matrix& A);
Matrix hessenberg(const Matrix& A);
Matrix inverse(const Matrix& A);
Matrix eigMatrix(const Matrix& A);
std::tuple<Matrix, Matrix> qr(const Matrix& A);
Vector gmres(const Matrix& A, const Vector& b, const Vector& x0, double tol = 1.0e-10, int maxits = 10000);
Matrix eye(int n);
Vector eig(const Matrix& A);
Vector getColumn(Matrix A, int c);
#endif