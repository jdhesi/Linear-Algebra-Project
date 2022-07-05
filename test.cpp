#include <iostream>
#include  <stdlib.h>
# include <cassert>
#include "Exception.h"
#include "Matrix.h"
#include "Vector.h"
#include <fstream>
#include <tuple>
# define M_PI 3.14159265358979323846



int main()
{
	Matrix A = rand(5);
	std::cout << A << -A << "\n";
}


/*
// 1D basis functions ( and derivatives ): 
double phi0(double x);
double phi1(double x);
double gradphi0(double x);
double gradphi1(double x);

// xi -> x coordinates caluclator
double masterToLocal(double x, double xl, double xr);

// c function
double c(double x);
int main()
{
	int meshpts = 50;
	Vector mesh = linspace(0.0, M_PI, meshpts);
	Vector dofs = linspace(1, meshpts, meshpts);
	double h = M_PI / ((double)meshpts - 1.0);

	// initialise problem matrices and vector 
	Matrix A(meshpts);
	Matrix M(meshpts);
	Vector F(meshpts);
	// initialise dofs
	int dof1, dof2;
	double Aii, Aij, Ajj;
	double Mii, Mij, Mjj;
	double vec1, vec2;
	double x1, x2, x3;

	// M can be calculated prior to the loop
	Mii = 1.0 / 6.0 * (h * phi0(0.0) * phi0(0.0))
		+ 4.0 / 6.0 * (h * phi0(0.5) * phi0(0.5))
		+ 1.0 / 6.0 * (h * phi0(1.0) * phi0(1.0));

	Mij = 1.0 / 6.0 * (h * phi1(0.0) * phi0(0.0))
		+ 4.0 / 6.0 * (h * phi1(0.5) * phi0(0.5))
		+ 1.0 / 6.0 * (h * phi1(1.0) * phi0(1.0));

	Mjj = 1.0 / 6.0 * (h * phi1(0.0) * phi1(0.0))
		+ 4.0 / 6.0 * (h * phi1(0.5) * phi1(0.5))
		+ 1.0 / 6.0 * (h * phi1(1.0) * phi1(1.0));
	for (int element = 1; element < dofs(meshpts); element++)
	{
		dof1 = element;
		dof2 = element + 1;
		x1 = masterToLocal(0.0, mesh(dof1), mesh(dof2));
		x2 = masterToLocal(0.5, mesh(dof1), mesh(dof2));
		x3 = masterToLocal(1.0, mesh(dof1), mesh(dof2));


		// calculate local elements 
		Aii = (double)
			1.0 / 6.0 * ((1.0 / h) * (gradphi0(0.0)) * (gradphi0(0.0)) + h * c(x1) * phi0(0.0) * phi0(0.0))
			+ 4.0 / 6.0 * ((1.0 / h) * (gradphi0(0.5)) * (gradphi0(0.5)) + h * c(x2) * phi0(0.5) * phi0(0.5))
			+ 1.0 / 6.0 * ((1.0 / h) * (gradphi0(1.0)) * (gradphi0(1.0)) + h * c(x3) * phi0(1.0) * phi0(0.0));
		Aij = (double)
			1.0 / 6.0 * ((1.0 / h) * (gradphi1(0.0)) * (gradphi0(0.0)) + h * c(x1) * phi1(0.0) * phi0(0.0))
			+ 4.0 / 6.0 * ((1.0 / h) * (gradphi1(0.5)) * (gradphi0(0.5)) + h * c(x2) * phi1(0.5) * phi0(0.5))
			+ 1.0 / 6.0 * ((1.0 / h) * (gradphi1(1.0)) * (gradphi0(1.0)) + h * c(x3) * phi1(1.0) * phi0(1.0));
		Ajj = (double)
			1.0 / 6.0 * ((1.0 / h) * (gradphi1(0.0)) * (gradphi1(0.0)) + h * c(x1) * phi1(0.0) * phi1(0.0))
			+ 4.0 / 6.0 * ((1.0 / h) * (gradphi1(0.5)) * (gradphi1(0.5)) + h * c(x2) * phi1(0.5) * phi1(0.5))
			+ 1.0 / 6.0 * ((1.0 / h) * (gradphi1(1.0)) * (gradphi1(1.0)) + h * c(x3) * phi1(1.0) * phi1(0.0));
		
		// append local elements to global matrices
		A(dof1, dof1) += Aii;
		A(dof1, dof2) += Aij;
		A(dof2, dof1) += Aij;
		A(dof2, dof2) += Ajj;

		M(dof1, dof1) += Mii;
		M(dof1, dof2) += Mij;
		M(dof2, dof1) += Mij;
		M(dof2, dof2) += Mjj;
	}
	// apply boundary conditions
	A(1, 1) = 0;
	A(1, 2) = 0;
	A(meshpts, meshpts) = 0;
	A(meshpts, meshpts - 1) = 0;
	M(1, 1) = 1;
	M(1, 2) = 0;
	M(meshpts, meshpts) = 1;
	M(meshpts, meshpts - 1) = 0;

	// compute the matrix we want the eigenvalues of
	Matrix B = inverse(M) * A;
	// print eigenvalues
	std::cout << "eigs: " << eig(B) << "\n";

}
double masterToLocal(double x, double xl, double xr)
{
	return xl + x * (xr - xl);
}
double c(double x)
{
	return 0.0;
}

double phi0(double x)
{
	return 1.0 - x;
}
double phi1(double x) 
{ 
	return x; 
}
double gradphi0(double x)
{
	return -1.0;
}
double gradphi1(double x)
{
	return 1.0;
}
*/




















// ********** Poisson ********** \\

/*
// 1D basis functions ( and derivatives ): 
double phi1(double x);
double phi2(double x);
double gradphi1(double x);
double gradphi2(double x);
// forcing functions: 
double f(double x);

int main()
{
	int meshpts = 1000;
	Vector mesh = linspace(0, 1, meshpts);
	Vector dofs = linspace(1, meshpts, meshpts);
	double h = 1.0 / ((double)meshpts - 1.0);

	// Initialise A matrix (stiffness matrix) - square of size meshpts
	Matrix A(meshpts);
	// Initialise F vector (forcing vector) - size meshpts
	Vector F(meshpts);

	// Initialise DOFS
	int dof1, dof2;

	// Given basis functions 1-x and x, the local matrix will be symmetric so need 3 distinct values:
	double A11, A12, A22;
	A11 = (double) 1.0 / 6.0 * ((1.0 / h) * (gradphi1(0.0)) * (gradphi1(0.0)))
				 + 4.0 / 6.0 * ((1.0 / h) * (gradphi1(0.5)) * (gradphi1(0.5)))
		         + 1.0 / 6.0 * ((1.0 / h) * (gradphi1(1.0)) * (gradphi1(1.0)));
	A12 = (double) 1.0 / 6.0 * ((1.0 / h) * (gradphi1(0.0)) * (gradphi2(0.0)))
				 + 4.0 / 6.0 * ((1.0 / h) * (gradphi1(0.5)) * (gradphi2(0.5)))
		         + 1.0 / 6.0 * ((1.0 / h) * (gradphi1(1.0)) * (gradphi2(1.0)));
	A22 = (double) 1.0 / 6.0 * ((1.0 / h) * (gradphi2(0.0)) * (gradphi2(0.0))) +
				 + 4.0 / 6.0 * ((1.0 / h) * (gradphi2(0.5)) * (gradphi2(0.5))) +
				 + 1.0 / 6.0 * ((1.0 / h) * (gradphi2(1.0)) * (gradphi2(1.0)));
	// Vector values
	double F1, F2;
	F1 = (double) 1.0 / 6.0 * (h * f(0.0) * phi1(0.0)) +
				+ 4.0 / 6.0 * (h * f(0.5) * phi1(0.5)) +
				+ 1.0 / 6.0 * (h * f(1.0) * phi1(1.0));
	F2 = (double) 1.0 / 6.0 * (h * f(0.0) * phi2(0.0)) +
		        + 4.0 / 6.0 * (h * f(0.5) * phi2(0.5)) +
				+ 1.0 / 6.0 * (h * f(1.0) * phi2(1.0));
	// loop through the elements in the domain,... 
	// ... adding the submatrix to A in the appropriate locations
	for (int element = 1; element < dofs(meshpts); element++)
	{
		dof1 = element;		// dofs give the index in A that we add to
		dof2 = element + 1;

		// Matrix and vector assembly
		A(dof1, dof1) += A11;
		A(dof1, dof2) += A12;
		A(dof2, dof1) += A12;
		A(dof2, dof2) += A22;
		F(dof1) += F1;
		F(dof2) += F2;
	}
	// Apply boundary conditions
	A(1, 1) = 1;
	A(1, 2) = 0;
	A(meshpts, meshpts) = 1;
	A(meshpts, meshpts - 1) = 0;
	F(1) = 0;
	F(meshpts) = 0;

	// Printing the computed matrix and vector
	//std::cout << "A: " << A << "\n";
	//std::cout << "F: " << F << "\n";

	// Solving the problem AU = f

	// Using Gaussian elimination
	Vector U_ge(meshpts);
	U_ge = A / F;

	// Using GMRES 
	Vector U_gm(meshpts);
	Vector guess = 0 * U_gm;
	int maxits = meshpts;
	double tol = 1.0e-10;
	U_gm = gmres(A, F, guess, tol, maxits);

	// Write results in columns | x | u(x) | to .dat file for MATLAB processing

	/*
	// Write Guassian elimination solution for 100 mesh points
	std::ofstream outfile1("ge_solution_100.dat");
	assert(outfile1.is_open());
	outfile1 << mesh << " " << U_ge << "\n";
	outfile1.close();
	
	// Write GMRES solution for 100 mesh points
	std::ofstream outfile2("gm_solution_100.dat");
	assert(outfile2.is_open());
	outfile2 << mesh << " " << U_gm << "\n";
	outfile2.close();
	*/
//}
/*
// all functions
double phi1(double x){return 1.0 - x;}
double phi2(double x){return x;}
double gradphi1(double x){return -1.0;}
double gradphi2(double x){return 1.0;}
double f(double x){return -1;}
*/

/*
int M = 100;
// Create random matrix of size M
Matrix A = rand(M)*10.0;
// Create RHS TO Ax=b
Vector b(M);
// Assign 1s to b
for (int i = 1; i < M + 1; i++)
{
	b(i) = 1;
}
// set gmres parameters
int maxits = 1000;;
double tol = 1.0e-10;
Vector guess = 0.0 * b;
// initialise results vector,
// gmres has currently been modified to return
// the vector of residuals norm(r_k)/norm(b)
Vector results;
// stream out results to .dat file for MATLAB
// processing
std::ofstream outfile("gmres_convergence.dat");
assert(outfile.is_open());
for (int i = 0; i <= 100; i += 10)
{
	// we consider convergence for A + multiples of the identity
	results = gmres(A + i * eye(M), b, guess, tol, maxits);
	outfile << results;
}
outfile.close();
*/