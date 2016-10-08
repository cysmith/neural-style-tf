// Copyright: Thomas Brox

#include <math.h>
#include <stdlib.h>
#include <NMath.h>

namespace NMath {

  const float Pi = 3.1415926536;

  // faculty
  int faculty(int n) {
    int aResult = 1;
    for (int i = 2; i <= n; i++)
      aResult *= i;
    return aResult;
  }

  // binCoeff
  int binCoeff(const int n, const int k) {
    if (k > (n >> 1)) return binCoeff(n,n-k);
    int aResult = 1;
    for (int i = n; i > (n-k); i--)
      aResult *= i;
    for (int j = 2; j <= k; j++)
      aResult /= j;
    return aResult;
  }

  // tangent
  float tangent(const float x1, const float y1, const float x2, const float y2) {
    float alpha;
    float xDiff = x2-x1;
    float yDiff = y2-y1;
    if (yDiff > 0) {
      if (xDiff == 0) alpha = 0.5*Pi;
      else if (xDiff > 0) alpha = atan(yDiff/xDiff);
      else alpha = Pi+atan(yDiff/xDiff);
    }
    else {
      if (xDiff == 0) alpha = -0.5*Pi;
      else if (xDiff > 0) alpha = atan(yDiff/xDiff);
      else alpha = -Pi+atan(yDiff/xDiff);
    }
    return alpha;
  }

  // absAngleDifference
  float absAngleDifference(const float aFirstAngle, const float aSecondAngle) {
    float aAlphaDiff = abs(aFirstAngle - aSecondAngle);
    if (aAlphaDiff > Pi) aAlphaDiff = 2*Pi-aAlphaDiff;
    return aAlphaDiff;
  }

  // angleDifference
  float angleDifference(const float aFirstAngle, const float aSecondAngle) {
    float aAlphaDiff = aFirstAngle - aSecondAngle;
    if (aAlphaDiff > Pi) aAlphaDiff = -2*Pi+aAlphaDiff;
    else if (aAlphaDiff < -Pi) aAlphaDiff = 2*Pi+aAlphaDiff;
    return aAlphaDiff;
  }

  // angleSum
  float angleSum(const float aFirstAngle, const float aSecondAngle) {
    float aSum = aFirstAngle + aSecondAngle;
    if (aSum > Pi) aSum = -2*Pi+aSum;
    else if (aSum < -Pi) aSum = 2*Pi+aSum;
    return aSum;
  }

  // round
  int round(const float aValue) {
    float temp1 = floor(aValue);
    float temp2 = ceil(aValue);
    if (aValue-temp1 < 0.5) return (int)temp1;
    else return (int)temp2;
  }

  // PATransformation
  // Cyclic Jacobi method for determining the eigenvalues and eigenvectors
  // of a symmetric matrix.
  // Ref.:  H.R. Schwarz: Numerische Mathematik. Teubner, Stuttgart, 1988.
  //        pp. 243-246.
  void PATransformation(const CMatrix<float>& aMatrix, CVector<float>& aEigenvalues, CMatrix<float>& aEigenvectors, bool aOrdering) {
    static const float eps = 0.0001;
    static const float delta = 0.000001;
    static const float eps2 = eps*eps;
    float sum,theta,t,c,r,s,g,h;
    // Initialization
    CMatrix<float> aCopy(aMatrix);
    int n = aEigenvalues.size();
    aEigenvectors = 0;
    for (int i = 0; i < n; i++)
      aEigenvectors(i,i) = 1;
    // Loop
    do {
      // check whether accuracy is reached
      sum = 0.0;
      for (int i = 1; i < n; i++)
        for (int j = 0; j <= i-1; j++)
          sum += aCopy(i,j)*aCopy(i,j);
      if (sum+sum > eps2) {
        for (int p = 0; p < n-1; p++)
          for (int q = p+1; q < n; q++)
            if (fabs(aCopy(q,p)) >= eps2) {
              theta = (aCopy(q,q) - aCopy(p,p)) / (2.0 * aCopy(q,p));
              t = 1.0;
              if (fabs(theta) > delta) t = 1.0 / (theta + theta/fabs(theta) * sqrt (theta*theta + 1.0));
              c = 1.0 / sqrt (1.0 + t*t);
              s = c*t;
              r = s / (1.0 + c);
              aCopy(p,p) -= t * aCopy(q,p);
              aCopy(q,q) += t * aCopy(q,p);
              aCopy(q,p) = 0;
              for (int j = 0; j <= p-1; j++) {
                g = aCopy(q,j) + r * aCopy(p,j);
                h = aCopy(p,j) - r * aCopy(q,j);
                aCopy(p,j) -= s*g;
                aCopy(q,j) += s*h;
              }
              for (int i = p+1; i <= q-1; i++) {
                g = aCopy(q,i) + r * aCopy(i,p);
                h = aCopy(i,p) - r * aCopy(q,i);
                aCopy(i,p) -= s * g;
                aCopy(q,i) += s * h;
              }
              for (int i = q+1; i < n; i++) {
                g = aCopy(i,q) + r * aCopy(i,p);
                h = aCopy(i,p) - r * aCopy(i,q);
                aCopy(i,p) -= s * g;
                aCopy(i,q) += s * h;
              }
              for (int i = 0; i < n; i++) {
                g = aEigenvectors(i,q) + r * aEigenvectors(i,p);
                h = aEigenvectors(i,p) - r * aEigenvectors(i,q);
                aEigenvectors(i,p) -= s * g;
                aEigenvectors(i,q) += s * h;
              }
            }
      }
    }
    // Return eigenvalues
    while (sum+sum > eps2);
    for (int i = 0; i < n; i++)
      aEigenvalues(i) = aCopy(i,i);
    if (aOrdering) {
      // Order eigenvalues and eigenvectors
      for (int i = 0; i < n-1; i++) {
        int k = i;
        for (int j = i+1; j < n; j++)
          if (fabs(aEigenvalues(j)) > fabs(aEigenvalues(k))) k = j;
        if (k != i) {
          // Switch eigenvalue i and k
          float help = aEigenvalues(k);
          aEigenvalues(k) = aEigenvalues(i);
          aEigenvalues(i) = help;
          // Switch eigenvector i and k
          for (int j = 0; j < n; j++) {
            help = aEigenvectors(j,k);
            aEigenvectors(j,k) = aEigenvectors(j,i);
            aEigenvectors(j,i) = help;
          }
        }
      }
    }
  }

  // PABackTransformation
  void PABacktransformation(const CMatrix<float>& aEigenvectors, const CVector<float>& aEigenvalues, CMatrix<float>& aMatrix) {
    for (int i = 0; i < aEigenvalues.size(); i++)
      for (int j = 0; j <= i; j++) {
         float sum = aEigenvalues(0) * aEigenvectors(i,0) * aEigenvectors(j,0);
         for (int k = 1; k < aEigenvalues.size(); k++)
           sum += aEigenvalues(k) * aEigenvectors(i,k) * aEigenvectors(j,k);
         aMatrix(i,j) = sum;
      }
    for (int i = 0; i < aEigenvalues.size(); i++)
      for (int j = i+1; j < aEigenvalues.size(); j++)
        aMatrix(i,j) = aMatrix(j,i);
  }

  // svd (nach Numerical Recipes in C basierend auf Forsythe et al.: Computer Methods for
  // Mathematical Computations (Englewood Cliffs, NJ: Prentice-Hall), Chapter 9, 1977,
  // Code übernommen von Bodo Rosenhahn)
  void svd(CMatrix<float>& U, CMatrix<float>& S, CMatrix<float>& V, bool aOrdering, int aIterations) {
    static float at,bt,ct;
    static float maxarg1,maxarg2;
    #define PYTHAG(a,b) ((at=fabs(a)) > (bt=fabs(b)) ?  (ct=bt/at,at*sqrt(1.0+ct*ct)) : (bt ? (ct=at/bt,bt*sqrt(1.0+ct*ct)): 0.0))
    #define MAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?	(maxarg1) : (maxarg2))
    #define MIN(a,b) ((a) >(b) ? (b) : (a))
    #define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
    int flag,i,its,j,jj,k,l,nm;
	  float c,f,h,s,x,y,z;
	  float anorm=0.0,g=0.0,scale=0.0;
    int aXSize = U.xSize();
    int aYSize = U.ySize();
	  CVector<float> aBuffer(aXSize);
    for (i = 0; i < aXSize; i++) {
      l=i+1;
      aBuffer(i)=scale*g;
      g=s=scale=0.0;
      if (i < aYSize) {
        for (k = i; k < aYSize; k++)
          scale += fabs(U(i,k));
        if (scale) {
          for (k = i; k < aYSize; k++) {
            U(i,k) /= scale;
	   				s += U(i,k)*U(i,k);
          }
		  		f = U(i,i);
          g = -SIGN(sqrt(s),f);
		  		h = f*g-s;
          U(i,i) = f-g;
          for (j = l; j < aXSize; j++) {
  	   			for (s = 0.0, k = i; k < aYSize; k++)
              s += U(i,k)*U(j,k);
            f = s/h;
            for (k = i; k < aYSize; k++)
				     	U(j,k) += f*U(i,k);
          }
			    for ( k = i; k < aYSize; k++)
			     	U(i,k) *= scale;
        }
      }
   	  S(i,i) = scale*g;
      g=s=scale=0.0;
      if (i < aYSize && i != aXSize-1) {
     	  for (k = l; k < aXSize; k++)
          scale += fabs(U(k,i));
        if (scale != 0)	{
          for (k = l; k < aXSize; k++) {
            U(k,i) /= scale;
            s += U(k,i)*U(k,i);
          }
	   		  f = U(l,i);
          g = -SIGN(sqrt(s),f);
	   		  h = f*g-s;
          U(l,i) = f-g;
          for (k = l; k < aXSize; k++)
            aBuffer(k) = U(k,i)/h;
          for (j = l; j < aYSize; j++) {
            for (s = 0.0, k = l; k < aXSize; k++)
              s += U(k,j)*U(k,i);
	   			  for (k = l; k < aXSize; k++)
              U(k,j) += s*aBuffer(k);
          }
	   		  for (k = l; k < aXSize; k++)
            U(k,i) *= scale;
        }
      }
	    anorm = MAX(anorm,(fabs(S(i,i))+fabs(aBuffer(i))));
    }
   	for (i = aXSize-1; i >= 0; i--)	{
   		if (i < aXSize-1)	{
        if (g != 0)	{
          for (j = l; j < aXSize; j++)
            V(i,j) = U(j,i)/(U(l,i)*g);
          for (j = l; j < aXSize; j++) {
            for (s = 0.0, k = l; k < aXSize; k++)
              s += U(k,i)*V(j,k);
            for (k = l; k < aXSize; k++)
              V(j,k) += s*V(i,k);
          }
        }
        for (j = l; j < aXSize; j++)
          V(j,i) = V(i,j) = 0.0;
      }
  		V(i,i) = 1.0;
      g = aBuffer(i);
  		l = i;
    }
	  for (i = MIN(aYSize-1,aXSize-1); i >= 0; i--)	{
	  	l = i+1;
	  	g = S(i,i);
      for (j = l; j < aXSize; j++)
	  		U(j,i) = 0.0;
      if (g != 0) {
        g = 1.0/g;
        for (j = l; j < aXSize; j++) {
          for (s = 0.0, k = l; k < aYSize; k++)
            s += U(i,k)*U(j,k);
		   		f = (s/U(i,i))*g;
          for (k = i; k < aYSize; k++)
            U(j,k) += f*U(i,k);
        }
   			for (j = i; j < aYSize; j++)
          U(i,j) *= g;
      }
		  else {
		   	for (j = i; j < aYSize; j++)
			  	U(i,j) = 0.0;
      }
   		++U(i,i);
    }
   	for (k = aXSize-1; k >= 0; k--)	{
      for (its = 1; its <= aIterations; its++)	{
	   		flag = 1;
        for (l = k; l > 0; l--) {
          nm = l - 1;
          if (fabs(aBuffer(l))+anorm == anorm)	{
            flag = 0; break;
          }
				  if (fabs(S(nm,nm))+anorm == anorm)	break;
        }
   			if (flag)	{
	  		  c = 0.0;
          s = 1.0;
          for (i = l; i <= k; i++) {
            f = s*aBuffer(i);
            aBuffer(i) = c*aBuffer(i);
            if (fabs(f)+anorm == anorm)	break;
            g = S(i,i);
				   	h = PYTHAG(f,g);
            S(i,i) = h;
            h = 1.0/h;
            c = g*h;
            s=-f*h;
            for (j = 0; j < aYSize; j++) {
              y = U(nm,j);
		   				z = U(i,j);
              U(nm,j) = y*c + z*s;
              U(i,j) = z*c - y*s;
            }
          }
        }
	  		z = S(k,k);
	   		if (l == k)	{
          if (z < 0.0) {
            S(k,k) = -z;
            for (j = 0; j < aXSize; j++)
              V(k,j) = -V(k,j);
          }
	   			break;
        }
		   	if (its == aIterations) std::cerr << "svd: No convergence in " << aIterations << " iterations" << std::endl;
			  x = S(l,l);
  			nm = k-1;
        y = S(nm,nm);
        g = aBuffer(nm);
        h = aBuffer(k);
		  	f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
        g = PYTHAG(f,1.0);
        f = ((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
        c = s = 1.0;
   			for (j = l; j <= nm; j++)	{
	  			i = j+1;
          g = aBuffer(i);
          y = S(i,i);
          h = s*g;
          g = c*g;
          z = PYTHAG(f,h);
          aBuffer(j) = z;
          float invZ = 1.0/z;
          c = f*invZ;
          s = h*invZ;
          f = x*c+g*s;
          g = g*c-x*s;
          h = y*s;
          y *= c;
          for (jj = 0; jj < aXSize; jj++)	{
            x = V(j,jj);
		   			z = V(i,jj);
            V(j,jj) = x*c + z*s;
			   		V(i,jj) = z*c - x*s;
          }
  				z = PYTHAG(f,h);
          S(j,j) = z;
          if (z != 0)	{
            z = 1.0/z;
            c = f*z;
            s = h*z;
          }
  				f = (c*g)+(s*y);
          x = (c*y)-(s*g);
          for (jj = 0; jj < aYSize; jj++)	{
            y = U(j,jj);
		  			z = U(i,jj);
            U(j,jj) = y*c + z*s;
            U(i,jj) = z*c - y*s;
          }
        }
   			aBuffer(l) = 0.0;
        aBuffer(k) = f;
	   		S(k,k) = x;
      }
    }
    // Order singular values
    if (aOrdering) {
      for (int i = 0; i < aXSize-1; i++) {
        int k = i;
        for (int j = i+1; j < aXSize; j++)
          if (fabs(S(j,j)) > fabs(S(k,k))) k = j;
        if (k != i) {
          // Switch singular value i and k
          float help = S(k,k);
          S(k,k) = S(i,i);
          S(i,i) = help;
          // Switch columns i and k in U and V
          for (int j = 0; j < aYSize; j++) {
            help = U(k,j);
            U(k,j) = U(i,j);
            U(i,j) = help;
          }
          for (int j = 0; j < aXSize; j++) {
            help = V(k,j);
            V(k,j) = V(i,j);
            V(i,j) = help;
          }
        }
      }
    }
  }

  #undef PYTHAG
  #undef MAX
  #undef MIN
  #undef SIGN

  // svdBack
  void svdBack(CMatrix<float>& U, const CMatrix<float>& S, const CMatrix<float>& V) {
    for (int y = 0; y < U.ySize(); y++)
      for (int x = 0; x < U.xSize(); x++)
        U(x,y) = S(x,x)*U(x,y);
    U *= trans(V);
  }

  // Householder-Verfahren (nach Stoer), uebernommen von Bodo Rosenhahn
  // Bei dem Verfahren wird die Matrix A (hier:*this und die rechte Seite (b)
  // mit unitaeren Matrizen P multipliziert, so dass A in eine
  // obere Dreiecksmatrix umgewandelt wird.
  // Dabei ist P = I + beta * u * uH
  // Die Vektoren u werden bei jeder Transformation in den nicht
  // benoetigten unteren Spalten von A gesichert.

  void householder(CMatrix<float>& A, CVector<float>& b) {
	  int i,j,k;
	  float sigma,s,beta,sum;
	  CVector<float> d(A.xSize());
    for (j = 0; j < A.xSize(); ++j) {
      sigma = 0;
	    for (i = j; i < A.ySize(); ++i)
    	  sigma += A(j,i)*A(j,i);
      if (sigma == 0) {
	      std::cerr << "NMath::householder(): matrix is singular!" << std::endl;
 	      break;
      }
	    // Choose sign to avoid elimination
	    s = d(j) = A(j,j)<0 ? sqrt(sigma) : -sqrt(sigma);
	    beta = 1.0/(s*A(j,j)-sigma);
      A(j,j) -= s;
      // Transform submatrix of A with P
	    for (k = j+1; k < A.xSize(); ++k)	{
	      sum = 0.0;
	      for (i = j; i < A.ySize(); ++i)
		      sum += (A(j,i)*A(k,i));
	      sum *= beta;
	      for (i = j; i < A.ySize(); ++i)
          A(k,i) += A(j,i)*sum;
      }
      // Transform right hand side of linear system with P
	    sum = 0.0;
	    for (i = j; i < A.ySize(); ++i)
	      sum += A(j,i)*b(i);
	    sum *= beta;
	    for (i = j; i < A.ySize(); ++i)
	      b(i) += A(j,i)*sum;
    }
    for (i = 0; i < A.xSize(); ++i)
	    A(i,i) = d(i);
  }

  // leastSquares
  CVector<float> leastSquares(CMatrix<float>& A, CVector<float>& b) {
    CVector<float> aResult(A.xSize());
    householder(A,b);
    for (int i = A.xSize()-1; i >= 0; i--) {
      float s = 0;
	    for (int k = i+1; k < A.xSize(); k++)
	      s += A(k,i)*aResult(k);
      aResult(i) = (b(i)-s)/A(i,i);
    }
    return aResult;
  }

  // invRegularized
  void invRegularized(CMatrix<float>& A, int aReg) {
    if (A.xSize() != A.ySize()) throw ENonquadraticMatrix(A.xSize(),A.ySize());
    CVector<float> eVals(A.xSize());
    CMatrix<float> eVecs(A.xSize(),A.ySize());
    PATransformation(A,eVals,eVecs);
    for (int i = 0 ; i < A.xSize(); i++)
      if (eVals(i) < aReg) eVals(i) = 1.0/aReg;
      else eVals(i) = 1.0/eVals(i);
    PABacktransformation(eVecs,eVals,A);
  }

  // cholesky
  void cholesky(CMatrix<float>& A) {
    if (A.xSize() != A.ySize()) throw ENonquadraticMatrix(A.xSize(),A.ySize());
    CVector<float> d(A.xSize());
    for (int i = 0; i < A.xSize(); i++)
      for (int j = i; j < A.ySize(); j++) {
        float sum = A(j,i);
        for (int k = i-1; k >= 0; k--)
          sum -= A(k,i)*A(k,j);
        if (i == j) {
          if (sum <= 0.0) return;//throw ENonPositiveDefinite();
          d(i) = sqrt(sum);
        }
        else A(i,j) = sum/d(i);
      }
    for (int i = 0; i < A.xSize(); i++)
      A(i,i) = d(i);
  }

  // triangularSolve
  void triangularSolve(CMatrix<float>& L, CVector<float>& aIn, CVector<float>& aOut) {
    for (int i = 0; i < aIn.size(); i++) {
      float sum = aIn(i);
      for (int j = 0; j < i; j++)
        sum -= L(j,i)*aOut(j);
      aOut(i) = sum/L(i,i);
    }
  }

  void triangularSolve(CMatrix<float>& L, CMatrix<float>& aIn, CMatrix<float>& aOut) {
    CVector<float> invLii(aIn.xSize());
    for (int i = 0; i < aIn.xSize(); i++)
      invLii(i) = 1.0/L(i,i);
    for (int k = 0; k < aIn.ySize(); k++)
      for (int i = 0; i < aIn.xSize(); i++) {
        float sum = aIn(i,k);
        for (int j = 0; j < i; j++)
          sum -= L(j,i)*aOut(j,k);
        aOut(i,k) = sum*invLii(i);
      }
  }

  // triangularSolveTransposed
  void triangularSolveTransposed(CMatrix<float>& L, CVector<float>& aIn, CVector<float>& aOut) {
    for (int i = aIn.size()-1; i >= 0; i--) {
      float sum = aIn(i);
      for (int j = aIn.size()-1; j > i; j--)
        sum -= L(i,j)*aOut(j);
      aOut(i) = sum/L(i,i);
    }
  }

  void triangularSolveTransposed(CMatrix<float>& L, CMatrix<float>& aIn, CMatrix<float>& aOut) {
    CVector<float> invLii(aIn.xSize());
    for (int i = 0; i < aIn.xSize(); i++)
      invLii(i) = 1.0/L(i,i);
    for (int k = 0; k < aIn.ySize(); k++)
      for (int i = aIn.xSize()-1; i >= 0; i--) {
        float sum = aIn(i,k);
        for (int j = aIn.xSize()-1; j > i; j--)
          sum -= L(i,j)*aOut(j,k);
        aOut(i,k) = sum*invLii(i);
      }
  }

  // choleskyInv
  void choleskyInv(const CMatrix<float>& L, CMatrix<float>& aInv) {
    aInv = 0;
    // Compute the inverse of L
    CMatrix<float> invL(L.xSize(),L.ySize());
    for (int i = 0; i < L.xSize(); i++)
      invL(i,i) = 1.0/L(i,i);
    for (int i = 0; i < L.xSize(); i++)
      for (int j = i+1; j < L.ySize(); j++) {
        float sum = 0.0;
        for (int k = i; k < j; k++)
          sum -= L(k,j)*invL(i,k);
        invL(i,j) = sum*invL(j,j);
      }
    // Compute lower triangle of aInv = invL^T * invL
    for (int i = 0; i < aInv.xSize(); i++)
      for (int j = i; j < aInv.ySize(); j++) {
        float sum = 0.0;
        for (int k = j; k < aInv.ySize(); k++)
          sum += invL(j,k)*invL(i,k);
        aInv(i,j) = sum;
      }
    // Complete aInv
    for (int i = 0; i < aInv.xSize(); i++)
      for (int j = i+1; j < aInv.ySize(); j++)
        aInv(j,i) = aInv(i,j);
  }

  // eulerAngles
  void eulerAngles(float rx, float ry, float rz, CMatrix<float>& A) {
    CMatrix<float> Rx(4,4,0);
    CMatrix<float> Ry(4,4,0);
    CMatrix<float> Rz(4,4,0);
    Rx(0,0)=1.0;Rx(1,1)=cos(rx);Rx(2,2)=cos(rx);Rx(3,3)=1.0;
    Rx(2,1)=-sin(rx);Rx(1,2)=sin(rx);
    Ry(1,1)=1.0;Ry(0,0)=cos(ry);Ry(2,2)=cos(ry);Ry(3,3)=1.0;
    Ry(0,2)=-sin(ry);Ry(2,0)=sin(ry);
    Rz(2,2)=1.0;Rz(0,0)=cos(rz);Rz(1,1)=cos(rz);Rz(3,3)=1.0;
    Rz(1,0)=-sin(rz);Rz(0,1)=sin(rz);
    A=Rz*Ry*Rx;
  }

  // RBM2Twist
  void RBM2Twist(CVector<float>& T, CMatrix<float>& fRBM) {
    T.setSize(6);
    CMatrix<double> dRBM(4,4);
    for (int i = 0; i < 16; i++)
      dRBM.data()[i] = fRBM.data()[i];
    CVector<double> omega;
    double theta;
    CVector<double> v;
    CMatrix<double> R(3,3);
    double sum = 0.0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        if (i != j) sum += dRBM(i,j)*dRBM(i,j);
        else sum += (dRBM(i,i)-1.0)*(dRBM(i,i)-1.0);
    if (sum < 0.0000001) {
      T(0)=fRBM(3,0); T(1)=fRBM(3,1); T(2)=fRBM(3,2);
      T(3)=0.0; T(4)=0.0; T(5)=0.0;
    }
    else {
      double diag = (dRBM(0,0)+dRBM(1,1)+dRBM(2,2)-1.0)*0.5;
      if (diag < -1.0) diag = -1.0;
      else if (diag > 1.0) diag = 1.0;
      theta = acos(diag);
      if (sin(theta)==0) theta += 0.0000001;
      omega.setSize(3);
      omega(0)=(dRBM(1,2)-dRBM(2,1));
      omega(1)=(dRBM(2,0)-dRBM(0,2));
      omega(2)=(dRBM(0,1)-dRBM(1,0));
      omega*=(1.0/(2.0*sin(theta)));
      CMatrix<double> omegaHat(3,3);
      omegaHat.data()[0] = 0.0;       omegaHat.data()[1] = -omega(2); omegaHat.data()[2] = omega(1);
      omegaHat.data()[3] = omega(2);  omegaHat.data()[4] = 0.0;       omegaHat.data()[5] = -omega(0);
      omegaHat.data()[6] = -omega(1); omegaHat.data()[7] = omega(0);  omegaHat.data()[8] = 0.0;
      CMatrix<double> omegaT(3,3);
      for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
          omegaT(i,j) = omega(i)*omega(j);
      R = (omegaHat*(double)sin(theta))+((omegaHat*omegaHat)*(double)(1.0-cos(theta)));
      R(0,0) += 1.0; R(1,1) += 1.0; R(2,2) += 1.0;
      CMatrix<double> A(3,3);
      A.fill(0.0);
      A(0,0)=1.0; A(1,1)=1.0; A(2,2)=1.0;
      A -= R;  A*=omegaHat;  A+=omegaT*theta;
      CVector<double> p(3);
      p(0)=dRBM(3,0);
      p(1)=dRBM(3,1);
      p(2)=dRBM(3,2);
      A.inv();
      v=A*p;
      T(0) = (float)(v(0)*theta);
      T(1) = (float)(v(1)*theta);
      T(2) = (float)(v(2)*theta);
      T(3) = (float)(theta*omega(0));
      T(4) = (float)(theta*omega(1));
      T(5) = (float)(theta*omega(2));
    }
  }

}

