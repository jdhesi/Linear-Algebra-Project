#include <iostream>
#include "Matrix.h"
#include "Vector.h"
#include <time.h> // for rand function
void Matrix::createMatrix(int rows, int columns)
{
	if (matrixVal != nullptr) // Delete the contents of any non-empty matrix
	{
		delete[] matrixVal;
	}
	matrixVal = new double[rows * columns];	// Allocate memory for single array format
	this->rows = rows;
	this->columns = columns;
	if (rows == 0 || columns == 0)
	{
		matrixVal = NULL;	// Assign an empty matrix NULL value
	}
	else
	{
		for (int i = 0; i < rows * columns; i++)
		{
			matrixVal[i] = 0; // Assign the matrix with zeros 
		}
	}
}

Matrix::Matrix()
{
	createMatrix(0, 0);
}

Matrix::Matrix(int r, int c)
{

	createMatrix(r, c);
}


Matrix::Matrix(const Matrix& m)
{
	createMatrix(m.rows, m.columns);
	for (int i = 0; i < m.rows * m.columns; i++)
	{
		matrixVal[i] = m.matrixVal[i];
	}
}
// ---------------------------------------------------------------------------
Matrix::~Matrix()
{
	if (matrixVal != nullptr)
	{
		delete[] matrixVal;
		matrixVal = nullptr;
	}
}
// ---------------------------------------------------------------------------
Matrix& Matrix::operator=(const Matrix& m)
{
	if (m.rows != rows || m.columns != columns)
	{
		createMatrix(m.rows, m.columns);
	}
	for (int i = 0; i < rows * columns; i++)
	{
		matrixVal[i] = m.matrixVal[i];
	}
	return *this;
}


double& Matrix::operator()(int i, int j)
{
	if ((i < 1) || (j < 1))	// exception if indices are too low
	{
		throw Exception("out of range", "index too small");
	}
	else if ((i > rows) || (j > columns)) // exception if indices are too high
	{
		throw Exception("out of range", "index too large");
	}
	return matrixVal[i - 1 + (j - 1) * rows];
}

Matrix& Matrix::swap_rows(int a, int b)
{
	// swap row a and row b (input with indexing starting at 1)
	a -= 1;
	b -= 1; // for indexing :/
	double temp;
	// loop through columns of row 
	for (int i = 0; i < this->columns; i++)
	{
		temp = this->matrixVal[a + this->rows * i];

		this->matrixVal[a + this->rows * i] = this->matrixVal[b + this->rows * i];
		this->matrixVal[b + this->rows * i] = temp;
	}
	return *this;
}

Matrix& Matrix::setColumn(int col, Vector& v)
{
	for (int i = 0; i < v.rows; i++)
	{
		this->matrixVal[i + (col - 1) * this->rows] = v.matrixVal[i];
	}
	return *this;
}

Matrix& Matrix::addRows(int r)
{
	int old_r = this->rows;
	int old_c = this->columns;
	Matrix temp(*this); // copy contents
	Matrix m_resize(old_r + r, old_c);
	//std::cout << "old r: " << m_resize.rows << "\n";
	*this = m_resize;
	//std::cout << "current " << *this << "\n";
	for (int i = 0; i < old_r; i++)
	{
		for (int j = 0; j < old_c; j++)
		{
			this->matrixVal[i + j * this->rows] = temp.matrixVal[i + j * old_r];
		}
	}
	return *this;
}

Matrix& Matrix::addCols(int c)
{
	int old_r = this->rows;
	int old_c = this->columns;
	Matrix temp(*this); // copy contents
	Matrix m_resize(old_r, old_c + c); // new matrix with additional columns
	*this = m_resize;
	for (int i = 0; i < old_r; i++)
	{
		for (int j = 0; j < old_c; j++)
		{
			this->matrixVal[i + j * this->rows] = temp.matrixVal[i + j * old_r];
		}
	}
	return *this;
}



Matrix operator+(const Matrix& m1, const Matrix& m2)
{
	// for now we don't want to add matrices of different size
	const int r = m1.rows;
	const int c = m1.columns;

	Matrix sum(m1); // output matrix
	if ((m2.rows != r) || (m2.columns != c))
	{
		throw Exception("incompatible sizes", "matrices must be the same size for addition");
	}
	else
	{

		for (int i = 0; i < r * c; i++)
		{
			sum.matrixVal[i] = m1.matrixVal[i] + m2.matrixVal[i];
		}
	}
	return sum;
}


Matrix operator-(const Matrix& m1, const Matrix& m2)
{
	// for now we don't want to add matrices of different size
	const int r = m1.rows;
	const int c = m1.columns;

	Matrix sum(m1); // output matrix
	if ((m2.rows != r) || (m2.columns != c))
	{
		throw Exception("incompatible sizes", "matrices must be the same size for addition");
	}
	else
	{

		for (int i = 0; i < r * c; i++)
		{
			sum.matrixVal[i] = m1.matrixVal[i] - m2.matrixVal[i];
		}
	}
	return sum;
}

Matrix operator*(const Matrix& m1, const Matrix& m2)
{
	const int r1 = m1.rows;
	const int c1 = m1.columns;
	const int r2 = m2.rows;
	const int c2 = m2.columns;
	Matrix m(r1, c2);
	if (c1 != r2)
	{
		throw Exception("incompatible sizes", "wrong sizes for matrix multiplication");
	}
	else
	{

		for (int i = 0; i < r1; i++)
		{
			for (int j = 0; j < c2; j++)
			{
				for (int k = 0; k < c1; k++)
				{
					m.matrixVal[i + j * r1] += m1.matrixVal[i + k * r1] * m2.matrixVal[k + j * r2];
				}
			}
		}
	}
	return m;
}

Matrix operator*(const double& a, const Matrix& m)
{
	//Matrix w(m);
	//for (int i = 0; i < m.rows * m.columns; i++)
	//{
	//	w.matrixVal[i] *= a;
	//}
	return m*a;
}

Matrix operator*(const Matrix& m, const double& a)
{
	Matrix w(m);
	for (int i = 0; i < m.rows * m.columns; i++)
	{
		w.matrixVal[i] *= a;
	}
	return w;
}

Matrix operator/(const Matrix& m, const double& a)
{
	if (a == 0)
	{
		throw Exception("dividing by 0", "attempting to divide matrix by 0");
	}
	else 
	{
		return m * (1.0 / a);
	}
	
}

Matrix operator-(const Matrix& m)
{
	Matrix A(m);
	for (int i = 0; i < m.rows * m.columns; i++)
	{
		A.matrixVal[i] = -m.matrixVal[i];
	}
	return A;
}

Vector operator*(const Matrix& m, const Vector& v)
{
	const int r1 = m.rows;
	const int c1 = m.columns;
	const int r2 = v.rows;
	const int c2 = 1;
	Vector prod(m.rows);
	if (c1 != r2)
	{
		throw Exception("incompatible sizes", "wrong sizes for matrix multiplication");
	}
	else
	{

		for (int i = 0; i < r1; i++)
		{
			for (int j = 0; j < c2; j++)
			{
				for (int k = 0; k < c1; k++)
				{
					prod.matrixVal[i + j * r1] += m.matrixVal[i + k * r1] * v.matrixVal[k + j * r2];
				}
			}
		}
	}
	return prod;

}
// ******************************************************************************************************************************************
// backslash operator

Vector operator/(const Matrix& A, const Vector& b)
{
	int m = A.rows;
	int n = m + 1; // Columns of Augmented matrix
	// We only accept square matrix input
	if (m != A.columns)
	{
		throw Exception(
			"Bad input", "A/b requires a sqaure matrix input");
	}
	// Form augmented matrix
	Vector bcopy(b);
	Matrix A_aug(A);
	A_aug.addCols(1);
	A_aug.setColumn(m + 1, bcopy);

	// solution vector
	Vector x(m); 

	// Initialise variables
	double largest_pivot = 0;
	int largest_pivot_index = 0;
	double q;	// variable to store a quotient in the loop

	int h = 0; // row index	   (0 --> m - 1)
	int k = 0; // column index (0 --> n - 1 = m)

	// loop through the augmented matrix
	while ((h < m) && (k < n))
	{
		// clear previos pivot values
		largest_pivot = 0;
		largest_pivot_index = 0;
		// findng pivot in column k - loop through all subdiagonal entries
		for (int i = h; i < m; i++)
		{
			// check if entry > largest pivot
			if (abs(A_aug.matrixVal[i + k * m]) > abs(largest_pivot))
			{
				// if entry > largest pivot, set largest pivot = entry
				largest_pivot = A_aug.matrixVal[i + k * m]; 
				largest_pivot_index = i;

			}

		}
		// if largest pivot is 0 we can move onto the next column
		if (largest_pivot == 0)
		{
			k += 1; 
		}
		// otherwise we bring the pivot to the sub diagonal and elimnate lower entries
		else
		{
			A_aug.swap_rows(h + 1, largest_pivot_index + 1);
			for (int i = h + 1; i < m; i++)
			{
				// find the ratio of each value to the pivot
				q = A_aug.matrixVal[i + k * m] / A_aug.matrixVal[h + k * m];
				// elimnate entries below the pivot
				A_aug.matrixVal[i + k * m] = 0;
				for (int j = k + 1; j < n; j++)
				{
					// perform corresponding operation on the rest of the row
					A_aug.matrixVal[i + j * m] -= A_aug.matrixVal[h + j * m] * q;
				}
			}
			// move to the next column, repeat the process
			h += 1;
			k += 1;
		}

	}

	// Now we find x using back substitution
	double sum = 0;

	// final value of x is simply the ratio of the final two non zero elements in the reduced 
	// augmented matrix 
	x.matrixVal[m - 1] = A_aug.matrixVal[m - 1 + (n - 1) * m] / A_aug.matrixVal[m - 1 + (n - 2) * m]; 
	// we loop through the rest of the rows, each time gaining another value in x
	for (int i = m - 2; i >= 0; i--)
	{
		sum = 0;
		for (int j = i + 1; j <= m - 1; j++)
		{
			// sum the known values of x * coefficients. -= since these will be subtracted from the final column value
			sum -= A_aug.matrixVal[i + j * m] * x.matrixVal[j];
		}
		// next x value is given by adding the sum of -(row elements * known x values) and the final coefficient 
		x.matrixVal[i] = (A_aug.matrixVal[i + m * (n - 1)] + sum) / (A_aug.matrixVal[i + i * m]); 
	}
	// return the solution
	return x;
}

/*
Vector operator/(const Matrix& A, const Vector& b)
{
	int m = A.rows;
	int n = A.columns + 1; 

	Vector x(m); //solution vector

	double largest_pivot = 0;
	int largest_pivot_index = 0;
	double q;
	Matrix tempA(m, n);

	for (int i = 0; i < A.rows; i++)
	{
		for (int j = 0; j < A.rows; j++)
		{
			tempA.matrixVal[i + j * m] = A.matrixVal[i + j * (m)];
		}
	}

	for (int i = 0; i < m; i++)
	{
		tempA.matrixVal[i + (n - 1) * m] = b.matrixVal[i]; // sets augmented matrix 
	}

	int h = 0;
	int k = 0;

	while ((h < m) && (k < n))
	{
		// find kth pivot i.e. pivot in column k
		largest_pivot = 0;
		largest_pivot_index = 0;
		for (int i = h; i < m; i++)
		{
			// loop over rows of column k
			if (abs(tempA.matrixVal[i + k * m]) > abs(largest_pivot))
			{
				largest_pivot = tempA.matrixVal[i + k * m];
				largest_pivot_index = i; // then (i,k) gives index of largest pivot

			}

		}

		if (largest_pivot == 0)
		{
			k += 1; // move onto the next column if this one has no pivot
		}
		else
		{
			tempA.swap_rows(h + 1, largest_pivot_index + 1);
			for (int i = h + 1; i < m; i++)
			{
				q = tempA.matrixVal[i + k * m] / tempA.matrixVal[h + k * m];
				tempA.matrixVal[i + k * m] = 0;
				for (int j = k + 1; j < n; j++)
				{
					tempA.matrixVal[i + j * m] -= tempA.matrixVal[h + j * m] * q;
				}
			}
			h += 1;
			k += 1;
		}

	}
	double sum = 0;
	x.matrixVal[m - 1] = tempA.matrixVal[m - 1 + (n - 1) * m] / tempA.matrixVal[m - 1 + (n - 2) * m]; //final value of x

	// final value is filled in
	// assuming square matrix now idk
	// back substitution: 
	for (int i = m - 2; i >= 0; i--)
	{
		sum = 0;
		for (int j = i + 1; j <= m - 1; j++)
		{
			sum -= tempA.matrixVal[i + j * m] * x.matrixVal[j];
		}
		x.matrixVal[i] = (tempA.matrixVal[i + m * (n - 1)] + sum) / (tempA.matrixVal[i + i * m]); // b(i) --> tempA(i,end) final column
	}

	return x;

}
// ******************************************************************************************************************************************
*/
std::ostream& operator<<(std::ostream& output, const Matrix& m) {
	output << "\n";
	for (int r = 0; r < m.rows; r++)
	{
		//std::cout << "|";
		for (int c = 0; c < m.columns; c++)
		{
			output << m.matrixVal[r + c * m.rows] << " ";
		}
		output << "\n";
	}
	output << "\n";
	return output;
}

Matrix eye(const int n) 
{
	Matrix m(n); 
	for (int i = 0; i < n; i += 1)
	{
		m.matrixVal[i + i * n] = 1;
	}
	return m;
}



Vector getColumn(Matrix A, int c)
{
	// want to return column c of matrix A 
	Vector col(A.rows);
	for (int i = 0; i < A.rows; i++)
	{
		col.matrixVal[i] = A.matrixVal[i + c * A.rows];
	}
	return col;
}

Matrix getRow(Matrix A, int c)
{
	Vector colOfTranspose = getColumn(transpose(A), c);
	return transpose(colOfTranspose);
}

Matrix transpose(const Matrix& m)
{
	Matrix mt(m.columns, m.rows);
	for (int i = 0; i < mt.rows; i++)
	{
		for (int j = 0; j < mt.columns; j++)
		{
			mt.matrixVal[i + j * mt.rows] = m.matrixVal[j + i * m.rows];
		}
	}
	return mt;
}

Vector gmres(const Matrix& A, const Vector& b, const Vector& x0, double tol, int maxits)
{

	int m = A.rows;
	if (m != A.columns)
	{
		throw Exception(
			"Bad input", "GMRES requires a sqaure matrix input");
	}
	// Residual given initial guess 
	Vector r = b - A * x0;
	// We initialise Q as a matrix instead of a vector since we shall add columns
	Matrix Q(m, 1);
	Vector q = r * (1.0 / norm(r));	// first entry of Q
	Q.setColumn(1, q); // q now added as first column of Q


	// Initialise the Hessenberg matrix obtained from Arnoldi
	Matrix H(2, 1); // starts as a 2x1

	/// Givens rotations values
	Matrix GivensTotal = eye(2);
	Matrix GivensCurrent;

	// variables used for Givens rotations
	Matrix Hcopy(H);
	double c;
	double s;
	double top;
	double bot;
	double div;
	Matrix UT(2);
	Vector errors(maxits);
	errors.matrixVal[0] = norm(r);
	double error;
	double bnorm = norm(b);

	// we will often require a temporary matrix
	Matrix temp;

	// variables for Arnoldi
	Matrix qt;
	Vector qi;
	Vector q_previous;


	// for gmres demonstration
	//Vector residuals(maxits);
	//int itsreq = 1;

	int k = 1; // errors iterations.
	// we define this outside the scope  of the loop so we can access it afterwards
	for (k; k < maxits; k++)
	{
		//std::cout<< " ~~~~~~~~~~~~~~~~ \n Start Val: " << A*Q << " ~~~~~~~~~~~~~~~~ \n";

		if (k > 1)
		{
			// on first iteration: Q is mx1 and H is 2x1, on further iterations we
			// increase the size of H to retain Upper Hessenberg form
			H.addRows(1);   // new rows are initialised with zero entries
			H.addCols(1);   // new columns are initialised with zero entries
		}


		// Arnoldi starts here -----------------------------------------------------------------
		// 
		// q_previous stores the previous column of q
		q_previous = getColumn(Q, k - 1);
		Q.addCols(1);
		// At this point H is (k+1)x(k) and Q is mxk 
		// Initially set next column as A*(previous column)
		q = A * q_previous;
		// Initialise next column in the Hessenberg matrix
		Vector h(k + 1);
		// Main Arnoldi Loop -------------------------------
		for (int i = 0; i < k; i++)
		{
			// This follows the standard Arnoldi algorithm 
			qi = getColumn(Q, i);
			qt = transpose(q);
			double hi = (qt * qi).matrixVal[0];
			h.matrixVal[i] = hi;
			q = q - hi * qi;
		}
		// -------------------------------------------------
		// Assign the final value of new H column
		h.matrixVal[k] = norm(q);
		// Final value of the newest column of Q
		q = q * (1.0 / norm(q));
		// add the new basis vector to the matrix
		Q.setColumn(k + 1, q); // --------------> Q_{k+1}
		// add new column to Hessenberg matrix
		H.setColumn(k, h);     // -------------->  H_{k}

		// Arnoldi ends here ---------------------------------------------------------------------



		// Uncommenting the below code gives a check for orthonormality between basis vectors:
		/*
		// q1 and q1 are columns of Q
		Matrix q1 = getRow(transpose(Q), k);
		Vector q2 = getColumn(Q, k+1);
		std::cout << "Q is " << Q << " and " << q1 << " x " << q2 << " = " << q1 * q2 << "\n"; // shows that they are roughly orthonormal
		// We have checked orthogonality and orthonormality - Q looks good
		*/

		// We now solve the minimisation problem ------------------------------------------------


		// Notes on implementation:
		// Need to compute the Givens rotation for the input H, this depends on the final two non zero entries of column k...
		// ... once we have multiplied the current H by the total of the previous Givens rotations. This will give a matrix...
		// ... which is upper triangular with an extra row of zeros
		// Need to store the value of the previous (total) Givens as this is required for the update of current givens
		// Inital value of total givens is 2x2 identity as initial H is 2x1.
		// - need to change H and keep a copy 
		// - want the current givens to be kxk

		// Initialise the givens matrix that will zero sub diagonal entries of the newest column of H
		// top left corner of givens is identity. Bottom right 2x2 is the interesting part
		GivensCurrent = eye(k - 1);
		GivensCurrent.addCols(2);
		GivensCurrent.addRows(2);

		// we will change the lower right 2x2 submatrix in current givens matrix to hold values [ c s ; -s c]...
		// ... where c and s are dependent on the final two non zero entries of the kth column of the current upper triangular form.

		// We want to store the H of this iteration as it is used for the next Arnoldi step, therefore make a copy that we change instead
		Hcopy = GivensTotal * H;

		// Need to multiply this by the current Givens matrix to obtain the current upper triangular matrix
		// So we need to form the givens matrix. This depends on the element on the ,...
		//...   diagonal and the one below it. Call these top and bot 

		// Equivalent to QR factorising but since it is Hessenberg initially we need to zero much fewer values

		// calculate givens matrix constants for this iteration
		top = Hcopy.matrixVal[(k - 1) + Hcopy.rows * (k - 1)];
		bot = Hcopy.matrixVal[(k)+Hcopy.rows * (k - 1)];
		div = pow(pow(top, 2) + pow(bot, 2), 0.5);
		c = top / div;
		s = bot / div;



		// now we have the values of the 2x2 we can add these to current givens to give our full transformation matrix

		GivensCurrent.matrixVal[(k - 1) + GivensCurrent.rows * (k - 1)] = c;
		GivensCurrent.matrixVal[(k)+GivensCurrent.rows * (k - 1)] = -s;
		GivensCurrent.matrixVal[(k - 1) + GivensCurrent.rows * (k)] = s;
		GivensCurrent.matrixVal[(k)+GivensCurrent.rows * (k)] = c;

		// we track the composition of all givens rotations
		temp = GivensCurrent * GivensTotal;
		GivensTotal = temp;

		// Current upper triangular matrix --> Note it is of the form | [U] |
		UT = GivensTotal * H;                                       //|--0--|


		// append newest residual based on previous residual and the givens transformation 
		errors.matrixVal[k] = errors.matrixVal[k - 1] * (-s);
		errors.matrixVal[k - 1] = errors.matrixVal[k - 1] * (c);
		error = fabs(errors.matrixVal[k] / bnorm);
		// for gmres demo
		//residuals.matrixVal[k] = error;


		// check termination condition
		if (error < tol)
		{
			std::cout << "GMRES CONVERGED IN " << k << " ITERATIONS WITH RESIDUAL " << error << "\n";

			//for gmres demo
			//itsreq = k;
			// 
			// exit loop
			k = maxits;
		}
		else
		{
			// Increase the size of the Givens total 
			// so it is ready for the next loop
			GivensTotal.addCols(1);
			GivensTotal.addRows(1);
			// New givens total = | |----------------|  0 |
			//                    | |old givens total| ...|
			//                    | |________________|  0 |
			//                    |  0      ...     0   1 |
			GivensTotal.matrixVal[(k + 1) + GivensTotal.rows * (k + 1)] = 1;

		}
		// Uncomment this to print the current iteration and error
		std::cout << " Current iteration: " << k << " Error : " << error << "\n";

	}
	//std::cout << "GMRES CONVERGED IN " << k << " ITERATIONS WITH RESIDUAL " << error << "\n";
	// need to remove final row of temp * H to give UT matrix
	Matrix U(UT.rows - 1, UT.columns);
	Vector beta(U.rows);
	Matrix Qf(Q.rows, Q.columns - 1);
	// remove final column of Q (since we terminate with 
	// Q_{k+1} but only the upper triangular matrix
	// gained from the QR decomposition of H_{k})
	for (int i = 0; i < Qf.rows; i++)
	{
		for (int j = 0; j < Qf.columns; j++)
		{
			Qf.matrixVal[i + Qf.rows * j] = Q.matrixVal[i + Q.rows * j];
		}
	}


	// Extract the  upper triangular matrix from UT
	// since it has an extra row of zeros
	// simultaneously extract all but the last rows
	// of the vector of residuals
	for (int i = 0; i < U.rows; i++)
	{
		//yy.matrixVal[i] = y.matrixVal[i];
		beta.matrixVal[i] = errors.matrixVal[i];
		for (int j = 0; j < U.columns; j++)
		{
			U.matrixVal[i + U.rows * j] = UT.matrixVal[i + UT.rows * j];
		}
	}
	// solve the minimisation problem using Gaussian elimination for 
	// the upper triangular matrix U. This is cheap since it only uses
	// back substitution
	Vector yu = U / beta;

	// convert back to x
	Vector kk = Qf * yu;
	return x0 + kk;

	/* Used for testing gmres convergence
	Vector residuals1(100);
	for (int j = 0; j < itsreq; j++ )
	{
		residuals1.matrixVal[j] = residuals.matrixVal[j+1];
	}
	return residuals1;
	*/

}

std::tuple<Matrix, Matrix> qr(const Matrix& A)
{
	// We assume A is square
	int m = A.rows;
	// Throw exception if A is non square
	if (m != A.columns) 
	{
		throw Exception(
			"Bad input", "QR requires a sqaure matrix input");
	}	
	// Initialise R as A andf Q as mxm identity
	Matrix R(A);
	Matrix Q = eye(m);

	// Initialise givens rotataion matrix
	Matrix givens;

	// Initialise variables for givens rotations 
	double top, bot, div, c, s;

	// top and bot represent the top and bottom of the final two entries in a column
	// these govern the 2x2 givens submatrix that zeros the bot entry
	// the 2x2 givens matrix is [c s ; -s c ] where c and s depend on top and bot.

	// Loop through each column
	for (int j = 0; j < m; j++)
	{
		// Loop from the final entry to the first subdiagonal entry in each column
		for (int i = m - 1; i > j; i--)
		{
			// define top and bot to be the pair of adjacent entries on the given ...
			// ... column at the given row index 
			top = R.matrixVal[(i - 1) + m * j];
			bot = R.matrixVal[(i)+m * j];

			// we only want to zero the bottom entry if it is non zero
			if (bot != 0)
			{
				// clear previous givens matrix, initialise new one
				givens = eye(m);
				// compute givens matrix constants for this loop
				div = pow(pow(top, 2) + pow(bot, 2), 0.5);
				c = top / div;
				s = bot / div;
				// assign values to givens matrix
				givens.matrixVal[(i)+m * (i)] = c;
				givens.matrixVal[(i)+m * (i - 1)] = -s;
				givens.matrixVal[(i - 1) + m * (i)] = s;
				givens.matrixVal[(i - 1) + m * (i - 1)] = c;
				// apply givens matrix to zero subdiagonal entries
				R = givens * R;
				// track the orthogonal vectors generated
				Q = Q * transpose(givens);
			}
		}
	}

	// FOR TESTING 
	// we can uncomment below to make it more clear when printing R that it is correct ...
	// ... since often rounding error adds up. 
	/*
	double tol = 1e-10;
	for (int i = 0; i < R.rows * R.columns; i++)
	{
		if (fabs(R.matrixVal[i]) < tol)
		{
			R.matrixVal[i] = 0;
		}
	}
	*/ 

	// return tuple (Q,R)
	return std::make_tuple(Q, R);
}

Matrix eigMatrix(const Matrix& A)
{

	int n = A.rows;
	// Initialise QR matrices
	Matrix Q, R;
	// Initialise Schur form matrix
	Matrix E(n);
	// QR already throws for non square input
	// compute QR
	std::tie(Q, R) = qr(A);
	// We impose a hidden maximum iterations for QR
	int maxits = 1000;
	// termination condition counter for subdiagonal
	// zeros
	int ctr = 0;
	int i = 0;
	// tolerence of finding zeros
	double tol = 1.0e-10;
	while (i < maxits && ctr < n - 1)
	{
		std::cout << "it: " << i << "\n";
		// mutliply the RxQ
		E = R * Q;
		//evecs = evecs * Q;
		for (int j = 0; j < n-1 ; j++)
		{
			// check subdiagonal sizes as termination 
			// condition
			if (fabs(E.matrixVal[(j + 1) + n * j]) < tol) 
			{
				ctr++;
			}
		}
		// compute next QR
		std::tie(Q, R) = qr(E);
		i++;
	}
	// We need to compute one final multiplication
	E = R * Q;
	/* 
	* Uncomment for testing - zeros elements close to zero
	* easier to track
	for (int i = 0; i < A.rows * A.columns; i++)
	{
		if (fabs(E.matrixVal[i]) < tol)
		{
			E.matrixVal[i] = 0;
		}
	}
	*/
	return E;
}

Vector eig(const Matrix& A)
{
	int n = A.rows;
	Matrix eigenvalueMatrix = eigMatrix(A);
	Vector eigVec(n);
	for (int i = 0; i < n; i++)
	{
		eigVec.matrixVal[i] = eigenvalueMatrix.matrixVal[i + n * i]; // get diagonal elements
	}
	return eigVec;
}

Matrix rand(int n)
{
	Matrix A(n);
	srand(time(NULL)); // random seed
	for (int i = 0; i < n * n; i++)
	{
		A.matrixVal[i] =  ((double) rand() / (RAND_MAX));
	}
	return A;
}

Vector eigP(const Matrix& A)
{
	int n = A.rows;
	Matrix eigenvalueMatrix = eigPractical(A);
	Vector eigVec(n);
	for (int i = 0; i < n; i++)
	{
		eigVec.matrixVal[i] = eigenvalueMatrix.matrixVal[i + n * i]; // get diagonal elements
	}
	return eigVec;
}


Matrix eigPractical(const Matrix& A)
{
	int n = A.rows;
	Matrix Q, R;
	Matrix E(n);
	std::tie(Q, R) = qr(A);
	int maxits = 10000;
	int ctr = 0;
	int i = 0;
	double tol = 0.0001;
	Vector trackeigs(n);
	double mu;
	while (i < maxits && ctr < n - 1)
	{
		std::cout << "it: " << i << "\n";
		ctr = 0;
		E = R * Q;

		for (int j = 0; j < n - 1; j++)
		{
			//std::cout << "E" << E << "\n";
			if (fabs(E.matrixVal[(j + 1) + n * j]) < tol) //check subdiagonal sizes
			{
				ctr++;
			}
		}
		mu = A.matrixVal[(n - 1) + n * (n - 1)];
		std::tie(Q, R) = qr(E+mu*eye(n));
		i++;
	}
	std::cout << "final its: " << i << "\n";
	E = R * Q;
	for (int i = 0; i < A.rows * A.columns; i++)
	{
		if (fabs(E.matrixVal[i]) < tol)
		{
			E.matrixVal[i] = 0;
		}
	}

	std::cout << "QR finished, E = " << E << "\n";
	return E;
}

Matrix inverse(const Matrix& A)
{
	// uses Gaussian elimination to
	// construct inverse by solving 
	// Ax=e_i
	int n = A.rows;
	Matrix Ainv(n);
	Vector col;
	for (int i = 0; i < n; i++)
	{
		Vector b(n);
		b.matrixVal[i] = 1;
		col = A / b;
		Ainv.setColumn(i+1, col);
	}
	return Ainv;
}

Matrix hessenberg(const Matrix& Atemp)
{
	// computes Hessenberg form of A using householder reflections
	int m = Atemp.rows;
	Matrix A(Atemp);
	Matrix v;
	double x0;
	for (int k = 0; k < m - 2; k++)
	{

		Matrix x(m - k - 1, 1); // all subdiag elements
		Matrix e(m - k - 1, 1);
		e.matrixVal[0] = 1;
		for (int i = 0; i < m - k - 1; i++)
		{
			x.matrixVal[i] = A.matrixVal[(i + 1 + k) + k * (m)]; // assign x
		}
		x0 = x.matrixVal[0];
		if (x0 == 0)
		{
			v = 0 * x;
		}
		else
		{
			v = (fabs(x0) / x0) * norm(convertMatrixToVector(x)) * e + x;
		}
		v = (1.0 / norm(convertMatrixToVector(v))) * v;
		Matrix subMatrix1((m - k - 1), (m - k));
		Matrix subMatrix2((m), (m - k - 1));
		for (int i = 0; i < subMatrix1.rows; i++)
		{
			for (int j = 0; j < subMatrix1.columns; j++)
			{
				subMatrix1.matrixVal[i + subMatrix1.rows * j] = A.matrixVal[(i + k + 1) + m * (j + k)];
			}
		}
		for (int i = 0; i < subMatrix2.rows; i++)
		{
			for (int j = 0; j < subMatrix2.columns; j++)
			{
				subMatrix2.matrixVal[i + subMatrix2.rows * j] = A.matrixVal[i + m * (j + k + 1)]; 
			}
		}
		Matrix reflectedSubMatrix1 = subMatrix1 - 2 * v * (transpose(v) * subMatrix1);
		Matrix tmp = subMatrix2 * v;
		Matrix tmp2 = transpose(v);
		Matrix tmp3 = 2 * (tmp * tmp2);
		Matrix reflectedSubMatrix2 = subMatrix2 - tmp3;
		for (int i = 0; i < reflectedSubMatrix1.rows; i++)
		{
			for (int j = 0; j < reflectedSubMatrix1.columns; j++)
			{
				A.matrixVal[(i + k + 1) + m * (j + k)] = reflectedSubMatrix1.matrixVal[i + reflectedSubMatrix1.rows * j];// assigning first sub matrix
			}
		}
		for (int i = 0; i < reflectedSubMatrix2.rows; i++)
		{
			for (int j = 0; j < reflectedSubMatrix2.columns; j++)
			{
				A.matrixVal[i + m * (j + k + 1)] = reflectedSubMatrix2.matrixVal[i + reflectedSubMatrix2.rows * j]; // assigning first sub matrix
			}
		}

	}
	return A;
}
	

Vector convertMatrixToVector(const Matrix& A)
{
	int n = A.rows;
	int m = A.columns;
	if (m != 1)
	{
		std::cout << "BAD CONVERSION TO VECTOR";
		return Vector();
	}
	Vector x(n);
	for (int i = 0; i < n; i++)
	{
		x.matrixVal[i] = A.matrixVal[i];
	}
	return x;
}
