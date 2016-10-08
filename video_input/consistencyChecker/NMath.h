// NMath
// A collection of mathematical functions and numerical algorithms
//
// Author: Thomas Brox

#ifndef NMathH
#define NMathH

#include <math.h>
#include <stdlib.h>
#include <CVector.h>
#include <CMatrix.h>

namespace NMath {
  // Returns the faculty of a number
  int faculty(int n);
  // Computes the binomial coefficient of two numbers
  int binCoeff(const int n, const int k);
  // Returns the angle of the line connecting (x1,y1) with (y1,y2)
  float tangent(const float x1, const float y1, const float x2, const float y2);
  // Absolute for floating points
  inline float abs(const float aValue);
  // Computes min or max value of two numbers
  inline float min(float aVal1, float aVal2);
  inline float max(float aVal1, float aVal2);
  inline int min(int aVal1, int aVal2);
  inline int max(int aVal1, int aVal2);
  // Computes the sign of a value
  inline float sign(float aVal);
  // minmod function (see description in implementation)
  inline float minmod(float a, float b, float c);
  // Computes the difference between two angles respecting the cyclic property of an angle
  // The result is always between 0 and Pi
  float absAngleDifference(const float aFirstAngle, const float aSecondAngle);
  // Computes the difference between two angles aFirstAngle - aSecondAngle
  // respecting the cyclic property of an angle
  // The result ist between -Pi and Pi
  float angleDifference(const float aFirstAngle, const float aSecondAngle);
  // Computes the sum of two angles respecting the cyclic property of an angle
  // The result is between -Pi and Pi
  float angleSum(const float aFirstAngle, const float aSecondAngle);
  // Rounds to the nearest integer
  int round(const float aValue);
  // Computes the arctan with results between 0 and 2*Pi
  inline float arctan(float x, float y);

  // Computes [0,1] uniformly distributed random number
  inline float random();
  // Computes N(0,1) distributed random number
  inline float randomGauss();

  extern const float Pi;

  // Computes a principal axis transformation
  // Eigenvectors are in the rows of aEigenvectors
  void PATransformation(const CMatrix<float>& aMatrix, CVector<float>& aEigenvalues, CMatrix<float>& aEigenvectors, bool aOrdering = true);
  // Computes the principal axis backtransformation
  void PABacktransformation(const CMatrix<float>& aEigenVectors, const CVector<float>& aEigenValues, CMatrix<float>& aMatrix);
  // Computes a singular value decomposition A=USV^T
  // Input: U MxN matrix
  // Output: U MxN matrix, S NxN diagonal matrix, V NxN diagonal matrix
  void svd(CMatrix<float>& U, CMatrix<float>& S, CMatrix<float>& V, bool aOrdering = true, int aIterations = 20);
  // Reassembles A = USV^T, Result in U
  void svdBack(CMatrix<float>& U, const CMatrix<float>& S, const CMatrix<float>& V);
  // Applies the Householder method to A and b, i.e., A is transformed into an upper triangular matrix
  void householder(CMatrix<float>& A, CVector<float>& b);
  // Computes least squares solution of an overdetermined linear system Ax=b using the Householder method
  CVector<float> leastSquares(CMatrix<float>& A, CVector<float>& b);
  // Inverts a square matrix by eigenvalue decomposition,
  // eigenvalues smaller than aReg are replaced by aReg
  void invRegularized(CMatrix<float>& A, int aReg);
  // Given a positive-definite symmetric matrix A, this routine constructs A = LL^T.
  // Only the upper triangle of A need be given. L is returned in the lower triangle.
  void cholesky(CMatrix<float>& A);
  // Solves L*aOut = aIn when L is a lower triangular matrix (e.g. result from cholesky)
  void triangularSolve(CMatrix<float>& L, CVector<float>& aIn, CVector<float>& aOut);
  void triangularSolve(CMatrix<float>& L, CMatrix<float>& aIn, CMatrix<float>& aOut);
  // Solves L^T*aOut = aIn when L is a lower triangular matrix (e.g. result from cholesky)
  void triangularSolveTransposed(CMatrix<float>& L, CVector<float>& aIn, CVector<float>& aOut);
  void triangularSolveTransposed(CMatrix<float>& L, CMatrix<float>& aIn, CMatrix<float>& aOut);
  // Computes the inverse of a matrix, given its cholesky decomposition L (lower triangle)
  void choleskyInv(const CMatrix<float>& L, CMatrix<float>& aInv);
  // Creates the rotation matrix RzRyRx and extends it to a 4x4 RBM matrix with translation 0
  void eulerAngles(float rx, float ry, float rz, CMatrix<float>& A);
  // Transforms a rigid body motion in matrix representation to a twist representation
  void RBM2Twist(CVector<float> &T, CMatrix<float>& RBM); 
}

// I M P L E M E N T A T I O N -------------------------------------------------
// Inline functions have to be implemented directly in the header file

namespace NMath {

  // abs
  inline float abs(const float aValue) {
    if (aValue >= 0) return aValue;
    else return -aValue;
  }

  // min
  inline float min(float aVal1, float aVal2) {
    if (aVal1 < aVal2) return aVal1;
    else return aVal2;
  }

  // max
  inline float max(float aVal1, float aVal2) {
    if (aVal1 > aVal2) return aVal1;
    else return aVal2;
  }

  // min
  inline int min(int aVal1, int aVal2) {
    if (aVal1 < aVal2) return aVal1;
    else return aVal2;
  }

  // max
  inline int max(int aVal1, int aVal2) {
    if (aVal1 > aVal2) return aVal1;
    else return aVal2;
  }

  // sign
  inline float sign(float aVal) {
    if (aVal > 0) return 1.0;
    else return -1.0;
  }

  // minmod function:
  //     0,                       if any of the a, b, c are 0 or of opposite sign
  //     sign(a) min(|a|,|b|,|c|) else
  inline float minmod(float a, float b, float c) {
    if ((sign(a) == sign(b)) && (sign(b) == sign(c)) && (a != 0.0)) {
      float aMin = fabs(a);
      if (fabs(b) < aMin) aMin = fabs(b);
      if (fabs(c) < aMin) aMin = fabs(c);
      return sign(a)*aMin;
    }
    else return 0.0;
  }

  // arctan
  inline float arctan(float x, float y) {
    if (x == 0.0)
      if (y >= 0.0) return 0.5 * 3.1415926536;
      else return 1.5 * 3.1415926536;
    else if (x > 0.0)
      if (y >= 0.0) return atan (y/x);
      else return 2.0 * 3.1415926536 + atan (y/x);
    else return 3.1415926536 + atan (y/x);
  }

  // random
  inline float random() {
    return (float)rand()/RAND_MAX;
  }

  // randomGauss
  inline float randomGauss() {
    // Draw two [0,1]-uniformly distributed numbers a and b
    float a = random();
    float b = random();
    // assemble a N(0,1) number c according to Box-Muller */
    if (a > 0.0) return sqrt(-2.0*log(a)) * cos(2.0*3.1415926536*b);
    else return 0;
  }

}
#endif
