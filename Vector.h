#include "Matrix.h"

class Vector : public Matrix
{
public:

	// Constructors
	Vector();
	Vector(int size);
	Vector(const Vector& v);

	// Operators
	Vector& operator=(const Vector& v);	// equality
	double& operator()(int i);			// indexing
	friend Vector operator-(const Vector& m1, const Vector& m2);
	friend Vector operator+(const Vector& m1, const Vector& m2);
	friend Vector operator*(const Vector& m, const double& a);
	friend Vector operator*(const double& a, const Vector& m);

	friend double norm(Vector v, int p); 
	friend Vector linspace(double a, double b, int n);
	friend double length(Vector v);

};
Vector linspace(double a, double b, int n);
double norm(Vector v, int p = 2);
double length(Vector v);