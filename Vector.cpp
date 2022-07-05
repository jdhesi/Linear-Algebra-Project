#include "Vector.h"
#include <iostream>

// Constructors -----------------------------------------------
Vector::Vector()
	:Matrix()
{
}
Vector::Vector(int size)
	:Matrix(size,1)
{

}
Vector::Vector(const Vector& v)
	:Matrix(v)
{

}

// --------------------------------------------------------------

// Operators ----------------------------------------------------

Vector& Vector::operator=(const Vector& v)
{
	if (v.rows != rows)
	{
		createMatrix(v.rows, 1);
	}
	for (int i = 0; i < rows; i++)
	{
		matrixVal[i] = v.matrixVal[i];
	}
	return *this;
}


double& Vector::operator()(int i)
{
	return Matrix::operator()(i, 1);
}


// --------------------------------------------------------------

Vector operator+(const Vector& m1, const Vector& m2)
{
	// for now we don't want to add matrices of different size
	const int r = m1.rows;

	Vector sum(m1); // output matrix
	if ((m2.rows != r))
	{
		throw Exception("incompatible sizes", "vectors must be the same size for addition");
	}
	else
	{

		for (int i = 0; i < r; i++)
		{
			sum.matrixVal[i] = m1.matrixVal[i] + m2.matrixVal[i];
		}
	}
	return sum;
}

Vector operator-(const Vector& m1, const Vector& m2)
{
	// for now we don't want to add matrices of different size
	const int r = m1.rows;

	Vector sum(m1); // output matrix
	if ((m2.rows != r))
	{
		throw Exception("incompatible sizes", "vectors must be the same size for addition");
	}
	else
	{

		for (int i = 0; i < r; i++)
		{
			sum.matrixVal[i] = m1.matrixVal[i] - m2.matrixVal[i];
		}
	}
	return sum;
}


Vector operator*(const double& a, const Vector& m)
{
	Vector w(m);
	for (int i = 0; i < m.rows * m.columns; i++)
	{
		w.matrixVal[i] *= a;
	}
	return w;
}

double norm(Vector v, int p)
{
	double norm_val = 0.0;
	double temp;
	for (int i = 0; i < v.rows; i++)
	{
		temp = fabs(v.matrixVal[i]);
		norm_val += pow(temp, p);
	}
	return pow(norm_val, 1.0 / ((double)(p)));
}




Vector operator*(const Vector& m, const double& a)
{
	Vector w(m);
	for (int i = 0; i < m.rows * m.columns; i++)
	{
		w.matrixVal[i] *= a;
	}
	return w;
}

Vector linspace(double a, double b, int n)
{
	Vector v(n);
	double length = b - a;
	double h = length / (n-1);
	for (int i = 0; i < n ; i++)
	{
		v.matrixVal[i] = a + h * i;
	}
	return v;
}

double length(Vector v)
{
	return v.rows;
}
