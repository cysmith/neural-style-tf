// - Classes for 1D and 2D convolution stencils
// - Pre-defined convolution stencils for binomial filters
// - Pre-defined convolution stencils for 1st, 2nd, 3rd and 4th derivatives up to order 10
// - Functions for convolution
//
// Author: Thomas Brox

#ifndef CFILTER
#define CFILTER

#include <math.h>
#include <NMath.h>
#include <CVector.h>
#include <CMatrix.h>
#include <CTensor.h>
#include <CTensor4D.h>

// CFilter is an extention of CVector. It has an additional property Delta
// which shifts the data to the left (a vector always begins with index 0).
// This enables a filter's range to go from A to B where A can also
// be less than zero.
//
// Example:
// CFilter<double> filter(3,1);
// filter = 1.0;
// cout << filter(-1) << ", " << filter(0) << ", " << filter(1) << endl;
//
// CFilter2D behaves the same way as CFilter but is an extension of CMatrix

template <class T>
class CFilter : public CVector<T> {
public:
  // constructor
  inline CFilter(const int aSize, const int aDelta = 0);
  // copy constructor
  CFilter(const CFilter<T>& aCopyFrom);
  // constructor initialized by a vector
  CFilter(const CVector<T>& aCopyFrom, const int aDelta = 0);

  // Access to the filter's values
  inline T& operator()(const int aIndex) const;
  inline T& operator[](const int aIndex) const;
  // Copies a filter into this filter
  CFilter<T>& operator=(const CFilter<T>& aCopyFrom);

  // Access to the filter's delta
  inline int delta() const;
  // Access to the filter's range A<=i<B
  inline int A() const;
  inline int B() const;
  // Returns the sum of all filter co-efficients (absolutes)
  T sum() const;
  // Shifts the filter
  inline void shift(int aDelta);
protected:
  int mDelta;
};

template <class T>
class CFilter2D : public CMatrix<T> {
public:
  // constructor
  inline CFilter2D();
  inline CFilter2D(const int aXSize, const int aYSize, const int aXDelta = 0, const int aYDelta = 0);
  // copy contructor
  CFilter2D(const CFilter2D<T>& aCopyFrom);
  // constructor initialized by a matrix
  CFilter2D(const CMatrix<T>& aCopyFrom, const int aXDelta = 0, const int aYDelta = 0);
  // Normalize sum of values to 1.0
  void normalizeSum();
  // Moves the filter's center
  void shift(int aXDelta, int aYDelta);

  // Access to filter's values
  inline T& operator()(const int ax, const int ay) const;
  // Copies a filter into this filter
  CFilter2D<T>& operator=(const CFilter2D<T>& aCopyFrom);

  // Access to the filter's delta
  inline int deltaX() const;
  inline int deltaY() const;
  // Access to the filter's range A<=i<B
  inline int AX() const;
  inline int BX() const;
  inline int AY() const;
  inline int BY() const;
  // Returns the sum of all filter co-efficients (absolutes)
  T sum() const;
protected:
  int mDeltaX;
  int mDeltaY;
};

namespace NFilter {

  // Linear 1D filtering

  // Convolution of the vector aVector with aFilter
  // The result will be written into aVector, so its initial values will get lost
  template <class T> inline void filter(CVector<T>& aVector, const CFilter<T>& aFilter);
  // Convolution of the vector aVector with aFilter, the initial values of aVector will persist.
  template <class T> void filter(const CVector<T>& aVector, CVector<T>& aResult, const CFilter<T>& aFilter);

  // Convolution with a rectangle -> approximation of Gaussian
  template <class T> inline void boxFilter(CVector<T>& aVector, int aWidth);
  template <class T> void boxFilter(const CVector<T>& aVector, CVector<T>& aResult, int aWidth);

  // Linear 2D filtering

  // Convolution of the matrix aMatrix with aFilter, aFilter should be a separable filter
  // The result will be written into aMatrix, so its initial values will get lost
  template <class T> inline void filter(CMatrix<T>& aMatrix, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY);
  template <class T> inline void filterMin(CMatrix<T>& aMatrix, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY);
  // Convolution of the matrix aMatrix with aFilter, aFilter must be separable
  // The initial values of aMatrix will persist.
  template <class T> inline void filter(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY);
  template <class T> inline void filterMin(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY);

  // Convolution of the matrix aMatrix with aFilter only in x-direction, aDummy can be set to 1
  // The result will be written into aMatrix, so its initial values will get lost
  template <class T> inline void filter(CMatrix<T>& aMatrix, const CFilter<T>& aFilter, const int aDummy);
  template <class T> inline void filterMin(CMatrix<T>& aMatrix, const CFilter<T>& aFilter, const int aDummy);
  // Convolution of the matrix aMatrix with aFilter only in x-direction, aDummy can be set to 1
  // The initial values of aMatrix will persist.
  template <class T> void filter(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const CFilter<T>& aFilter, const int aDummy);
  template <class T> void filterMin(const CMatrix<T>& aMatrix, const CMatrix<T>& aOrig, CMatrix<T>& aResult, const CFilter<T>& aFilter, const int aDummy);
  // Convolution of the matrix aMatrix with aFilter only in y-direction, aDummy can be set to 1
  // The result will be written into aMatrix, so its initial values will get lost
  template <class T> inline void filter(CMatrix<T>& aMatrix, const int aDummy, const CFilter<T>& aFilter);
  template <class T> inline void filterMin(CMatrix<T>& aMatrix, const int aDummy, const CFilter<T>& aFilter);
  // Convolution of the matrix aMatrix with aFilter only in y-direction, aDummy can be set to 1
  // The initial values of aMatrix will persist.
  template <class T> void filter(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const int aDummy, const CFilter<T>& aFilter);
  template <class T> void filterMin(const CMatrix<T>& aMatrix, const CMatrix<T>& aOrig, CMatrix<T>& aResult, const int aDummy, const CFilter<T>& aFilter);

  // Convolution of the matrix aMatrix with aFilter
  // The result will be written to aMatrix, so its initial values will get lost
  template <class T> inline void filter(CMatrix<T>& aMatrix, const CFilter2D<T>& aFilter);
  // Convolution of the matrix aMatrix with aFilter, the initial values of aMatrix will persist
  template <class T> void filter(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const CFilter2D<T>& aFilter);

  // Convolution with a rectangle -> approximation of Gaussian
  template <class T> inline void boxFilterX(CMatrix<T>& aMatrix, int aWidth);
  template <class T> void boxFilterX(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, int aWidth);
  template <class T> inline void boxFilterY(CMatrix<T>& aMatrix, int aWidth);
  template <class T> void boxFilterY(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, int aWidth);

  // Recursive filter -> approximation of Gaussian
  template <class T> void recursiveSmoothX(CMatrix<T>& aMatrix, float aSigma);
  template <class T> void recursiveSmoothY(CMatrix<T>& aMatrix, float aSigma);
  template <class T> inline void recursiveSmooth(CMatrix<T>& aMatrix, float aSigma);

  // Linear 3D filtering

  // Convolution of the 3D Tensor aTensor with aFilter, aFilter must be separable
  // The result will be written back to aTensor so its initial values will get lost
  template <class T> inline void filter(CTensor<T>& aTensor, const CFilter<T>& aFilterX,  const CFilter<T>& aFilterY, const CFilter<T>& aFilterZ);
  // Convolution of the 3D Tensor aTensor with aFilter, aFilter must be separable
  // The initial values of aTensor will persist
  template <class T> inline void filter(const CTensor<T>& aTensor, CTensor<T>& aResult, const CFilter<T>& aFilterX,  const CFilter<T>& aFilterY, const CFilter<T>& aFilterZ);

  // Convolution of the 3D Tensor aTensor with aFilter only in x-Direction
  template <class T> inline void filter(CTensor<T>& aTensor, const CFilter<T>& aFilter, const int aDummy1, const int aDummy2);
  template <class T> void filter(const CTensor<T>& aTensor, CTensor<T>& aResult, const CFilter<T>& aFilter, const int aDummy1, const int aDummy2);
  // Convolution of the 3D Tensor aTensor with aFilter only in y-Direction
  template <class T> inline void filter(CTensor<T>& aTensor, const int aDummy1, const CFilter<T>& aFilter, const int aDummy2);
  template <class T> void filter(const CTensor<T>& aTensor, CTensor<T>& aResult, const int aDummy1, const CFilter<T>& aFilter, const int aDummy2);
  // Convolution of the 3D Tensor aTensor with aFilter only in z-Direction
  template <class T> inline void filter(CTensor<T>& aTensor, const int aDummy1, const int aDummy2, const CFilter<T>& aFilter);
  template <class T> void filter(const CTensor<T>& aTensor, CTensor<T>& aResult, const int aDummy1, const int aDummy2, const CFilter<T>& aFilter);

    // Convolution with a rectangle -> approximation of Gaussian
  template <class T> inline void boxFilterX(CTensor<T>& aTensor, int aWidth);
  template <class T> void boxFilterX(const CTensor<T>& aTensor, CTensor<T>& aResult, int aWidth);
  template <class T> inline void boxFilterY(CTensor<T>& aTensor, int aWidth);
  template <class T> void boxFilterY(const CTensor<T>& aTensor, CTensor<T>& aResult, int aWidth);
  template <class T> inline void boxFilterZ(CTensor<T>& aTensor, int aWidth);
  template <class T> void boxFilterZ(const CTensor<T>& aTensor, CTensor<T>& aResult, int aWidth);

  // Recursive filter -> approximation of Gaussian
  template <class T> void recursiveSmoothX(CTensor<T>& aTensor, float aSigma);
  template <class T> void recursiveSmoothY(CTensor<T>& aTensor, float aSigma);
  template <class T> void recursiveSmoothZ(CTensor<T>& aTensor, float aSigma);

  // Linear 4D filtering

  // Convolution of the 4D Tensor aTensor with aFilter, aFilter must be separable
  // The result will be written back to aTensor so its initial values will get lost
  template <class T> inline void filter(CTensor4D<T>& aTensor, const CFilter<T>& aFilterX,  const CFilter<T>& aFilterY, const CFilter<T>& aFilterZ, const CFilter<T>& aFilterA);

  // Convolution of the 4D Tensor aTensor with aFilter only in x-Direction
  template <class T> inline void filter(CTensor4D<T>& aTensor, const CFilter<T>& aFilter, const int aDummy1, const int aDummy2, const int aDummy3);
  template <class T> void filter(const CTensor4D<T>& aTensor, CTensor4D<T>& aResult, const CFilter<T>& aFilter, const int aDummy1, const int aDummy2, const int aDummy3);
  // Convolution of the 4D Tensor aTensor with aFilter only in y-Direction
  template <class T> inline void filter(CTensor4D<T>& aTensor, const int aDummy1, const CFilter<T>& aFilter, const int aDummy2, const int aDummy3);
  template <class T> void filter(const CTensor4D<T>& aTensor, CTensor4D<T>& aResult, const int aDummy1, const CFilter<T>& aFilter, const int aDummy2, const int aDummy3);
  // Convolution of the 4D Tensor aTensor with aFilter only in z-Direction
  template <class T> inline void filter(CTensor4D<T>& aTensor, const int aDummy1, const int aDummy2, const CFilter<T>& aFilter, const int aDummy3);
  template <class T> void filter(const CTensor4D<T>& aTensor, CTensor4D<T>& aResult, const int aDummy1, const int aDummy2, const CFilter<T>& aFilter, const int aDummy3);
  // Convolution of the 4D Tensor aTensor with aFilter only in a-Direction
  template <class T> inline void filter(CTensor4D<T>& aTensor, const int aDummy1, const int aDummy2, const int aDummy3, const CFilter<T>& aFilter);
  template <class T> void filter(const CTensor4D<T>& aTensor, CTensor4D<T>& aResult, const int aDummy1, const int aDummy2, const int aDummy3, const CFilter<T>& aFilter);

  // Recursive filter -> approximation of Gaussian
  template <class T> void recursiveSmoothX(CTensor4D<T>& aTensor, float aSigma);
  template <class T> void recursiveSmoothY(CTensor4D<T>& aTensor, float aSigma);
  template <class T> void recursiveSmoothZ(CTensor4D<T>& aTensor, float aSigma);
  template <class T> void recursiveSmoothA(CTensor4D<T>& aTensor, float aSigma);

  // Nonlinear filtering: Osher shock filter
  template <class T> void osher(CMatrix<T>& aData, int aIterations = 20);
  template <class T> inline void osher(const CMatrix<T>& aData, CMatrix<T>& aResult, int aIterations = 20);
}

// Common filters

template <class T>
class CGauss : public CFilter<T> {
public:
  CGauss(const int aSize, const int aDegreeOfDerivative);
};

template <class T>
class CSmooth : public CFilter<T> {
public:
  CSmooth(float aSigma, float aPrecision);
};

template <class T>
class CGaussianFirstDerivative : public CFilter<T> {
public:
  CGaussianFirstDerivative(float aSigma, float aPrecision);
};

template <class T>
class CGaussianSecondDerivative : public CFilter<T> {
public:
  CGaussianSecondDerivative(float aSigma, float aPrecision);
};

template <class T>
class CDerivative : public CFilter<T> {
public:
  CDerivative(const int aSize);
};

template <class T>
class CHighOrderDerivative : public CFilter<T> {
public:
  CHighOrderDerivative(int aOrder, int aSize);
};

template <class T>
class CGaborReal : public CFilter2D<T> {
public:
  CGaborReal(float aFrequency, float aAngle, float aSigma1 = 3.0, float aSigma2 = 3.0);
};

template <class T>
class CGaborImaginary : public CFilter2D<T> {
public:
  CGaborImaginary(float aFrequency, float aAngle, float aSigma1 = 3.0, float aSigma2 = 3.0);
};

// Exceptions -----------------------------------------------------------------

// Thrown if one tries to access an element of a filter which is out of the filter's bounds
struct EFilterRangeOverflow {
  EFilterRangeOverflow(const int aIndex, const int aA, const int aB) {
    using namespace std;
    cerr << "Exception EFilterRangeOverflow: i = " << aIndex;
    cerr << "  Allowed Range: " << aA << " <= i < " << aB << endl;
  }
  EFilterRangeOverflow(const int ax, const int ay, const int aAX, const int aBX, const int aAY, const int aBY) {
    using namespace std;
    cerr << "Exception EFilterRangeOverflow: (x,y) = (" << ax << "," << ay << ")  ";
    cerr << "Allowed Range: " << aAX << " <= x < " << aBX << "  " << aAY << " <= y < " << aBY << endl;
  }
};

// Thrown if the resulting container has not the same size as the initial container
struct EFilterIncompatibleSize {
  EFilterIncompatibleSize(const int aSize1, const int aSize2) {
    using namespace std;
    cerr << "Exception EFilterIncompatibleSize: Initial container size: " << aSize1;
    cerr << "  Resulting container size: " << aSize2 << endl;
  }
};

// Thrown if the demanded filter is not available
struct EFilterNotAvailable {
  EFilterNotAvailable(int aSize, int aOrder) {
    using namespace std;
    cerr << "Exception EFilterNotAvailable: Mask size: " << aSize;
    if (aOrder >= 0) cerr << "  Derivative order: " << aOrder;
    cerr << endl;
  }
};

// I M P L E M E N T A T I O N ------------------------------------------------
//
// You might wonder why there is implementation code in a header file.
// The reason is that not all C++ compilers yet manage separate compilation
// of templates. Inline functions cannot be compiled separately anyway.
// So in this case the whole implementation code is added to the header
// file.
// Users should ignore everything that's beyond this line :)
// ----------------------------------------------------------------------------

// C F I L T E R --------------------------------------------------------------
// P U B L I C ----------------------------------------------------------------
// constructor
template <class T>
inline CFilter<T>::CFilter(const int aSize, const int aDelta)
  : CVector<T>(aSize),mDelta(aDelta) {
}

// copy constructor
template <class T>
CFilter<T>::CFilter(const CFilter<T>& aCopyFrom)
  : CVector<T>(aCopyFrom.mSize),mDelta(aCopyFrom.mDelta) {
  for (register int i = 0; i < this->mSize; i++)
    this->mData[i] = aCopyFrom.mData[i];
}

// constructor initialized by a vector
template <class T>
CFilter<T>::CFilter(const CVector<T>& aCopyFrom, const int aDelta)
  : CVector<T>(aCopyFrom.size()),mDelta(aDelta) {
  for (register int i = 0; i < this->mSize; i++)
    this->mData[i] = aCopyFrom(i);
}

// operator()
template <class T>
inline T& CFilter<T>::operator()(const int aIndex) const {
  #ifdef DEBUG
    if (aIndex < A() || aIndex >= B())
      throw EFilterRangeOverflow(aIndex,A(),B());
  #endif
  return this->mData[aIndex+mDelta];
}

// operator[]
template <class T>
inline T& CFilter<T>::operator[](const int aIndex) const {
  return operator()(aIndex);
}

// operator=
template <class T>
CFilter<T>& CFilter<T>::operator=(const CFilter<T>& aCopyFrom) {
  if (this != &aCopyFrom) {
    delete[] this->mData;
    this->mSize = aCopyFrom.mSize;
    mDelta = aCopyFrom.mDelta;
    this->mData = new T[this->mSize];
    for (register int i = 0; i < this->mSize; i++)
      this->mData[i] = aCopyFrom.mData[i];
  }
  return *this;
}

// delta
template <class T>
inline int CFilter<T>::delta() const {
  return mDelta;
}

// A
template <class T>
inline int CFilter<T>::A() const {
  return -mDelta;
}

// B
template <class T>
inline int CFilter<T>::B() const {
  return this->mSize-mDelta;
}

// sum
template <class T>
T CFilter<T>::sum() const {
  T aResult = 0;
  for (int i = 0; i < this->mSize; i++)
    aResult += fabs(this->mData[i]);
  return aResult;
}

// shift
template <class T>
inline void CFilter<T>::shift(int aDelta) {
  mDelta += aDelta;
}

// C F I L T E R 2 D -----------------------------------------------------------
// P U B L I C ----------------------------------------------------------------
// constructor
template <class T>
inline CFilter2D<T>::CFilter2D()
  : CMatrix<T>(),mDeltaX(0),mDeltaY(0) {
}

template <class T>
inline CFilter2D<T>::CFilter2D(const int aXSize, const int aYSize, const int aDeltaX, const int aDeltaY)
  : CMatrix<T>(aXSize,aYSize),mDeltaX(aDeltaX),mDeltaY(aDeltaY) {
}

// copy constructor
template <class T>
CFilter2D<T>::CFilter2D(const CFilter2D<T>& aCopyFrom)
  : CMatrix<T>(aCopyFrom.mXSize,aCopyFrom.mYSize),mDeltaX(aCopyFrom.mDeltaX,aCopyFrom.mDeltaY) {
  for (int i = 0; i < this->mXSize*this->mYSize; i++)
    this->mData[i] = aCopyFrom.mData[i];
}

// constructor initialized by a matrix
template <class T>
CFilter2D<T>::CFilter2D(const CMatrix<T>& aCopyFrom, const int aDeltaX, const int aDeltaY)
  : CMatrix<T>(aCopyFrom.xSize(),aCopyFrom.ySize()),mDeltaX(aDeltaX),mDeltaY(aDeltaY) {
  for (register int i = 0; i < this->mXSize*this->mYSize; i++)
    this->mData[i] = aCopyFrom.data()[i];
}

// normalizeSum
template <class T>
void CFilter2D<T>::normalizeSum() {
  int aSize = this->size();
  T aSum = 0;
  for (int i = 0; i < aSize; i++)
    aSum += this->mData[i];
  T invSum = 1.0/aSum;
  for (int i = 0; i < aSize; i++)
    this->mData[i] *= invSum;
}

// shift
template <class T>
void CFilter2D<T>::shift(int aXDelta, int aYDelta) {
  mDeltaX = aXDelta;
  mDeltaY = aYDelta;
}

// operator()
template <class T>
inline T& CFilter2D<T>::operator()(const int ax, const int ay) const {
  #ifdef DEBUG
    if (ax < AX() || ax >= BX() || ay < AY() || ay >= BY)
      throw EFilterRangeOverflow(ax,ay,AX(),BX(),AY(),BY());
  #endif
  return this->mData[(ay+mDeltaY)*this->mXSize+ax+mDeltaX];
}

// operator=
template <class T>
CFilter2D<T>& CFilter2D<T>::operator=(const CFilter2D<T>& aCopyFrom) {
  if (this != &aCopyFrom) {
    delete[] this->mData;
    this->mXSize = aCopyFrom.mXSize;
    this->mYSize = aCopyFrom.mYSize;
    mDeltaX = aCopyFrom.mDeltaX;
    mDeltaY = aCopyFrom.mDeltaY;
    this->mData = new T[this->mXSize*this->mYSize];
    for (register int i = 0; i < this->mXSize*this->mYSize; i++)
      this->mData[i] = aCopyFrom.mData[i];
  }
  return *this;
}

// deltaX
template <class T>
inline int CFilter2D<T>::deltaX() const {
  return mDeltaX;
}

// deltaY
template <class T>
inline int CFilter2D<T>::deltaY() const {
  return mDeltaY;
}

// AX
template <class T>
inline int CFilter2D<T>::AX() const {
  return -mDeltaX;
}

// AY
template <class T>
inline int CFilter2D<T>::AY() const {
  return -mDeltaY;
}

// BX
template <class T>
inline int CFilter2D<T>::BX() const {
  return this->mXSize-mDeltaX;
}

// BY
template <class T>
inline int CFilter2D<T>::BY() const {
  return this->mYSize-mDeltaY;
}

// sum
template <class T>
T CFilter2D<T>::sum() const {
  T aResult = 0;
  for (int i = 0; i < this->mXSize*this->mYSize; i++)
    aResult += abs(this->mData[i]);
  return aResult;
}

// C G A U S S -----------------------------------------------------------------
template <class T>
CGauss<T>::CGauss(const int aSize, const int aDegreeOfDerivative)
  : CFilter<T>(aSize,aSize >> 1) {
  CVector<int> *oldData;
  CVector<int> *newData;
  CVector<int> *temp;
  oldData = new CVector<int>(aSize);
  newData = new CVector<int>(aSize);

  (*oldData)(0) = 1;
  (*oldData)(1) = 1;

  for (int i = 2; i < aSize-aDegreeOfDerivative; i++) {
    (*newData)(0) = 1;
    for (int j = 1; j < i; j++)
      (*newData)(j) = (*oldData)(j)+(*oldData)(j-1);
    (*newData)(i) = 1;
    temp = oldData;
    oldData = newData;
    newData = temp;
  }
  for (int i = aSize-aDegreeOfDerivative; i < aSize; i++) {
    (*newData)(0) = 1;
    for (int j = 1; j < i; j++)
      (*newData)(j) = (*oldData)(j)-(*oldData)(j-1);
    (*newData)(i) = -(*oldData)(i-1);
    temp = oldData;
    oldData = newData;
    newData = temp;
  }

  int aSum = 0;
  for (int i = 0; i < aSize; i++)
    aSum += abs((*oldData)(i));
  double aInvSum = 1.0/aSum;
  for (int i = 0; i < aSize; i++)
    this->mData[aSize-1-i] = (*oldData)(i)*aInvSum;

  delete newData;
  delete oldData;
}

// C S M O O T H ---------------------------------------------------------------
template <class T>
CSmooth<T>::CSmooth(float aSigma, float aPrecision)
  : CFilter<T>(2*(int)ceil(aPrecision*aSigma)+1,(int)ceil(aPrecision*aSigma)) {
  float aSqrSigma = aSigma*aSigma;
  for (int i = 0; i <= (this->mSize >> 1); i++) {
    T aTemp = exp(i*i/(-2.0*aSqrSigma))/(aSigma*sqrt(2.0*NMath::Pi));
    this->operator()(i) = aTemp;
    this->operator()(-i) = aTemp;
  }
  T invSum = 1.0/this->sum();
  for (int i = 0; i < this->mSize; i++)
    this->mData[i] *= invSum;
}

template <class T>
CGaussianFirstDerivative<T>::CGaussianFirstDerivative(float aSigma, float aPrecision)
  : CFilter<T>(2*(int)ceil(aPrecision*aSigma)+1,(int)ceil(aPrecision*aSigma)) {
  float aSqrSigma = aSigma*aSigma;
  float aPreFactor = 1.0/(aSqrSigma*aSigma*sqrt(2.0*NMath::Pi));
  for (int i = 0; i <= (this->mSize >> 1); i++) {
    T aTemp = exp(i*i/(-2.0*aSqrSigma))*i*aPreFactor;
    this->operator()(i) = aTemp;
    this->operator()(-i) = -aTemp;
  }
}

template <class T>
CGaussianSecondDerivative<T>::CGaussianSecondDerivative(float aSigma, float aPrecision) 
  : CFilter<T>(2*(int)ceil(aPrecision*aSigma)+1,(int)ceil(aPrecision*aSigma)) {
  float aSqrSigma = aSigma*aSigma;
  float aPreFactor = 1.0/(aSqrSigma*aSigma*sqrt(2.0*NMath::Pi));
  for (int i = 0; i <= (this->mSize >> 1); i++) {
    T aTemp = exp(i*i/(-2.0*aSqrSigma))*(i*i/aSqrSigma-1.0)*aPreFactor;
    this->operator()(i) = aTemp;
    this->operator()(-i) = aTemp;
  }
}

// C D E R I V A T I V E -------------------------------------------------------
template <class T>
CDerivative<T>::CDerivative(const int aSize)
  : CFilter<T>(aSize,(aSize-1) >> 1) {
  switch (aSize) {
    case 2:
      this->mData[0] = -1;
      this->mData[1] =  1;
      break;
    case 3:
      this->mData[0] = -0.5;
      this->mData[1] = 0;
      this->mData[2] = 0.5;
      break;
    case 4:
      this->mData[0] =   0.041666666666666666666666666666667;
      this->mData[1] =  -1.125;
      this->mData[2] =   1.125;
      this->mData[3] =  -0.041666666666666666666666666666667;
      break;
    case 5:
      this->mData[0] =  0.083333333333;
      this->mData[1] = -0.66666666666;
      this->mData[2] =  0;
      this->mData[3] =  0.66666666666;
      this->mData[4] = -0.083333333333;
      break;
    case 6:
      this->mData[0] = -0.0046875;
      this->mData[1] =  0.0651041666666666666666666666666667;
      this->mData[2] = -1.171875;
      this->mData[3] =  1.171875;
      this->mData[4] = -0.0651041666666666666666666666666667;
      this->mData[5] =  0.0046875;
      break;
    case 7:
      this->mData[0] = -0.016666666666666666666666666666667;
      this->mData[1] =  0.15;
      this->mData[2] = -0.75;
      this->mData[3] =  0;
      this->mData[4] =  0.75;
      this->mData[5] = -0.15;
      this->mData[6] =  0.016666666666666666666666666666667;
      break;
    case 8:
      this->mData[0] =  6.9754464285714285714285714285714e-4;
      this->mData[1] = -0.0095703125;
      this->mData[2] =  0.079752604166666666666666666666667;
      this->mData[3] = -1.1962890625;
      this->mData[4] =  1.1962890625;
      this->mData[5] = -0.079752604166666666666666666666667;
      this->mData[6] =  0.0095703125;
      this->mData[7] = -6.9754464285714285714285714285714e-4;
      break;
    case 9:
      this->mData[0] =  0.0035714285714285714285714285714286;
      this->mData[1] = -0.038095238095238095238095238095238;
      this->mData[2] =  0.2;
      this->mData[3] = -0.8;
      this->mData[4] =  0;
      this->mData[5] =  0.8;
      this->mData[6] = -0.2;
      this->mData[7] =  0.038095238095238095238095238095238;

      this->mData[8] = -0.0035714285714285714285714285714286;
      break;
    case 10:
      this->mData[0] = -1.1867947048611111111111111111111e-4;
      this->mData[1] =  0.0017656598772321428571428571428571;
      this->mData[2] = -0.0138427734375;
      this->mData[3] =  0.0897216796875;
      this->mData[4] = -1.21124267578125;
      this->mData[5] =  1.21124267578125;
      this->mData[6] = -0.0897216796875;
      this->mData[7] =  0.0138427734375;
      this->mData[8] = -0.0017656598772321428571428571428571;
      this->mData[9] =  1.1867947048611111111111111111111e-4;
      break;
    default:
      throw EFilterNotAvailable(aSize,-1);
  }
}

// C H I G H O R D E R D E R I V A T I V E -------------------------------------
template <class T>
CHighOrderDerivative<T>::CHighOrderDerivative(int aOrder, int aSize)
  : CFilter<T>(aSize,(aSize-1) >> 1) {
  switch (aSize) {
    case 3:
      switch (aOrder) {
        case 2:
          this->mData[0] =  1;
          this->mData[1] =  -2;
          this->mData[2] =  1;
          break;
        default:
          throw EFilterNotAvailable(aSize,aOrder);
      }
      break;
    case 4:
      switch (aOrder) {
        case 2:
          this->mData[0] =   0.25;
          this->mData[1] =  -0.25;
          this->mData[2] =  -0.25;
          this->mData[3] =   0.25;
          break;
        case 3:
          this->mData[0] =  -0.25;
          this->mData[1] =   0.75;
          this->mData[2] =  -0.75;
          this->mData[3] =   0.25;
          break;
        default:
          throw EFilterNotAvailable(aSize,aOrder);
      }
      break;
    case 5:
      switch (aOrder) {
        case 2:
          this->mData[0] = -0.083333333333333333333333333333333;
          this->mData[1] =  1.3333333333333333333333333333333;
          this->mData[2] = -2.5;
          this->mData[3] =  1.3333333333333333333333333333333;
          this->mData[4] = -0.083333333333333333333333333333333;
          break;
        case 3:
          this->mData[0] = -0.5;
          this->mData[1] =  1;
          this->mData[2] =  0;
          this->mData[3] = -1;
          this->mData[4] =  0.5;
          break;
        case 4:
          this->mData[0] =  1;
          this->mData[1] = -4;
          this->mData[2] =  6;
          this->mData[3] = -4;
          this->mData[4] =  1;
          break;
        default:
          throw EFilterNotAvailable(aSize,aOrder);
      }
      break;
    case 6:
      switch (aOrder) {
        case 2:
          this->mData[0] = -0.052083333333333333333333333333333;
          this->mData[1] =  0.40625;
          this->mData[2] = -0.35416666666666666666666666666667;
          this->mData[3] = -0.35416666666666666666666666666667;
          this->mData[4] =  0.40625;
          this->mData[5] = -0.052083333333333333333333333333333;
          break;
        case 3:
          this->mData[0] =  0.03125;
          this->mData[1] = -0.40625;
          this->mData[2] =  1.0625;
          this->mData[3] = -1.0625;
          this->mData[4] =  0.40625;
          this->mData[5] = -0.03125;
          break;
        case 4:
          this->mData[0] =  0.0625;
          this->mData[1] = -0.1875;
          this->mData[2] =  0.125;
          this->mData[3] =  0.125;
          this->mData[4] = -0.1875;
          this->mData[5] =  0.0625;
          break;
        default:
          throw EFilterNotAvailable(aSize,aOrder);
      }
      break;
    case 7:
      switch (aOrder) {
        case 2:
          this->mData[0] =  0.011111111111111111111111111111111;
          this->mData[1] = -0.15;
          this->mData[2] =  1.5;
          this->mData[3] = -2.6666666666666666666666666666667;
          this->mData[4] =  1.5;
          this->mData[5] = -0.15;
          this->mData[6] =  0.011111111111111111111111111111111;
          break;
        case 3:
          this->mData[0] =  0.125;
          this->mData[1] = -1;
          this->mData[2] =  1.625;
          this->mData[3] =  0;
          this->mData[4] = -1.625;
          this->mData[5] =  1;
          this->mData[6] = -0.125;
          break;
        case 4:
          this->mData[0] = -0.16666666666666666666666666666667;
          this->mData[1] =  2;
          this->mData[2] = -6.5;
          this->mData[3] =  9.3333333333333333333333333333333;
          this->mData[4] = -6.5;
          this->mData[5] =  2;
          this->mData[6] = -0.16666666666666666666666666666667;
          break;
        default:
          throw EFilterNotAvailable(aSize,aOrder);
      }
      break;
    case 8:
      switch (aOrder) {
        case 2:
          this->mData[0] =  0.011241319444444444444444444444444;
          this->mData[1] = -0.10828993055555555555555555555556;
          this->mData[2] =  0.507421875;
          this->mData[3] = -0.41037326388888888888888888888889;
          this->mData[4] = -0.41037326388888888888888888888889;
          this->mData[5] =  0.507421875;
          this->mData[6] = -0.10828993055555555555555555555556;
          this->mData[7] =  0.011241319444444444444444444444444;
          break;
        case 3:
          this->mData[0] = -0.0048177083333333333333333333333333;
          this->mData[1] =  0.064973958333333333333333333333333;
          this->mData[2] = -0.507421875;
          this->mData[3] =  1.2311197916666666666666666666667;
          this->mData[4] = -1.2311197916666666666666666666667;
          this->mData[5] =  0.507421875;
          this->mData[6] = -0.064973958333333333333333333333333;
          this->mData[7] =  0.0048177083333333333333333333333333;
          break;
        case 4:
          this->mData[0] = -0.018229166666666666666666666666667;
          this->mData[1] =  0.15364583333333333333333333333333;
          this->mData[2] = -0.3515625;
          this->mData[3] =  0.21614583333333333333333333333333;
          this->mData[4] =  0.21614583333333333333333333333333;
          this->mData[5] = -0.3515625;
          this->mData[6] =  0.15364583333333333333333333333333;
          this->mData[7] = -0.018229166666666666666666666666667;
          break;
        default:
          throw EFilterNotAvailable(aSize,aOrder);
      }
      break;
    case 9:
      switch (aOrder) {
        case 2:
          this->mData[0] = -0.0017857142857142857142857142857143;
          this->mData[1] =  0.025396825396825396825396825396825;
          this->mData[2] = -0.2;
          this->mData[3] =  1.6;
          this->mData[4] = -2.8472222222222222222222222222222;
          this->mData[5] =  1.6;
          this->mData[6] = -0.2;
          this->mData[7] =  0.025396825396825396825396825396825;
          this->mData[8] = -0.0017857142857142857142857142857143;
          break;
        case 3:
          this->mData[0] = -0.029166666666666666666666666666667;
          this->mData[1] =  0.3;
          this->mData[2] = -1.4083333333333333333333333333333;
          this->mData[3] =  2.0333333333333333333333333333333;
          this->mData[4] =  0;
          this->mData[5] = -2.0333333333333333333333333333333;
          this->mData[6] =  1.4083333333333333333333333333333;
          this->mData[7] = -0.3;
          this->mData[8] =  0.029166666666666666666666666666667;
          break;
        case 4:
          this->mData[0] =  0.029166666666666666666666666666667;
          this->mData[1] = -0.4;
          this->mData[2] =  2.8166666666666666666666666666667;
          this->mData[3] = -8.1333333333333333333333333333333;
          this->mData[4] =  11.375;
          this->mData[5] = -8.1333333333333333333333333333333;
          this->mData[6] =  2.8166666666666666666666666666667;
          this->mData[7] = -0.4;
          this->mData[8] =  0.029166666666666666666666666666667;
          break;
        default:
          throw EFilterNotAvailable(aSize,aOrder);
      }
      break;
    case 10:
      switch (aOrder) {
        case 2:
          this->mData[0] = -0.0025026351686507936507936507936508;
          this->mData[1] =  0.028759765625;
          this->mData[2] = -0.15834263392857142857142857142857;
          this->mData[3] =  0.57749565972222222222222222222222;
          this->mData[4] = -0.44541015625;
          this->mData[5] = -0.44541015625;
          this->mData[6] =  0.57749565972222222222222222222222;
          this->mData[7] = -0.15834263392857142857142857142857;
          this->mData[8] =  0.028759765625;
          this->mData[9] = -0.0025026351686507936507936507936508;
          break;
        case 3:
          this->mData[0] =  0.0008342117228835978835978835978836;
          this->mData[1] = -0.012325613839285714285714285714286;
          this->mData[2] =  0.095005580357142857142857142857143;
          this->mData[3] = -0.57749565972222222222222222222222;
          this->mData[4] =  1.33623046875;
          this->mData[5] = -1.33623046875;
          this->mData[6] =  0.57749565972222222222222222222222;
          this->mData[7] = -0.095005580357142857142857142857143;
          this->mData[8] =  0.012325613839285714285714285714286;
          this->mData[9] = -0.0008342117228835978835978835978836;
          break;
        case 4:
          this->mData[0] =  0.00458984375;
          this->mData[1] = -0.050358072916666666666666666666667;
          this->mData[2] =  0.24544270833333333333333333333333;
          this->mData[3] = -0.480078125;
          this->mData[4] =  0.28040364583333333333333333333333;
          this->mData[5] =  0.28040364583333333333333333333333;
          this->mData[6] = -0.480078125;
          this->mData[7] =  0.24544270833333333333333333333333;
          this->mData[8] = -0.050358072916666666666666666666667;
          this->mData[9] =  0.00458984375;
          break;
        default:
          throw EFilterNotAvailable(aSize,aOrder);
      }
      break;
    default:
      throw EFilterNotAvailable(aSize,aOrder);
  }
}

// C G A B O R -----------------------------------------------------------------
template <class T>
CGaborReal<T>::CGaborReal(float aFrequency, float aAngle, float aSigma1, float aSigma2)
  : CFilter2D<T>() {
  // sqrt(2.0*log(2.0))/(2.0*NMath::Pi) = 0.18739
  float sigma1Sqr2 = aSigma1*0.18739/aFrequency;
  sigma1Sqr2 = 0.5/(sigma1Sqr2*sigma1Sqr2);
  float sigma2Sqr2 = aSigma2*0.18739/aFrequency;
  sigma2Sqr2 = 0.5/(sigma2Sqr2*sigma2Sqr2);
  float aCos = cos(aAngle);
  float aSin = sin(aAngle);
  float a = 0.6*aSigma1/aFrequency;
  float b = 0.6*aSigma2/aFrequency;
  float aXSize = fabs(a*aCos)+fabs(b*aSin);
  float aYSize = fabs(b*aCos)+fabs(a*aSin);
  this->setSize(1+2.0*floor(aXSize),1+2.0*floor(aYSize));
  this->shift(floor(aXSize),floor(aYSize));
  for (int y = this->AY(); y < this->BY(); y++)
    for (int x = this->AX(); x < this->BX(); x++) {
      float a = x*aCos+y*aSin;
      float b = y*aCos-x*aSin;
      float aGauss = exp(-sigma1Sqr2*a*a-sigma2Sqr2*b*b);
      float aHelp = 2.0*NMath::Pi*aFrequency*(x*aCos+y*aSin);
      this->operator()(x,y) = aGauss*cos(aHelp);
    }
}

template <class T>
CGaborImaginary<T>::CGaborImaginary(float aFrequency, float aAngle, float aSigma1, float aSigma2)
  : CFilter2D<T>() {
  // sqrt(2.0*log(2.0))/(2.0*NMath::Pi) = 0.18739
  float sigma1Sqr2 = aSigma1*0.18739/aFrequency;
  sigma1Sqr2 = 0.5/(sigma1Sqr2*sigma1Sqr2);
  float sigma2Sqr2 = aSigma2*0.18739/aFrequency;
  sigma2Sqr2 = 0.5/(sigma2Sqr2*sigma2Sqr2);
  float aCos = cos(aAngle);
  float aSin = sin(aAngle);
  float a = 0.6*aSigma1/aFrequency;
  float b = 0.6*aSigma2/aFrequency;
  float aXSize = fabs(a*aCos)+fabs(b*aSin);
  float aYSize = fabs(b*aCos)+fabs(a*aSin);
  this->setSize(1+2.0*floor(aXSize),1+2.0*floor(aYSize));
  this->shift(floor(aXSize),floor(aYSize));
  for (int y = this->AY(); y < this->BY(); y++)
    for (int x = this->AX(); x < this->BX(); x++) {
      float a = x*aCos+y*aSin;
      float b = y*aCos-x*aSin;
      float aGauss = exp(-sigma1Sqr2*a*a-sigma2Sqr2*b*b);
      float aHelp = 2.0*NMath::Pi*aFrequency*(x*aCos+y*aSin);
      this->operator()(x,y) = aGauss*sin(aHelp);
    }
}

// F I L T E R -----------------------------------------------------------------

namespace NFilter {

// 1D linear filtering ---------------------------------------------------------

template <class T>
inline void filter(CVector<T>& aVector, const CFilter<T>& aFilter) {
  CVector<T> oldVector(aVector);
  filter(oldVector,aVector,aFilter);
}

template <class T>
void filter(const CVector<T>& aVector, CVector<T>& aResult, const CFilter<T>& aFilter) {
  if (aResult.size() != aVector.size()) throw EFilterIncompatibleSize(aVector.size(),aResult.size());
  int x1 = -aFilter.A();
  int x2 = aVector.size()-aFilter.B();
  int a2Size = 2*aVector.size()-1;
  // Left rim
  for (int i = 0; i < x1; i++) {
    aResult[i] = 0;
    for (int j = aFilter.A(); j < aFilter.B(); j++)
      if (j+i < 0) aResult(i) += aFilter(j)*aVector(-1-j-i);
      else aResult(i) += aFilter(j)*aVector(j+i);
  }
  // Middle
  for (int i = x1; i < x2; i++) {
    aResult[i] = 0;
    for (int j = aFilter.A(); j < aFilter.B(); j++)
      aResult(i) += aFilter(j)*aVector(j+i);
  }
  // Right rim
  for (int i = x2; i < aResult.size(); i++) {
    aResult[i] = 0;
    for (int j = aFilter.A(); j < aFilter.B(); j++)
      if (j+i >= aVector.size()) aResult(i) += aFilter(j)*aVector(a2Size-j-i);
      else aResult(i) += aFilter(j)*aVector(j+i);
  }
}

// boxfilter
template <class T>
inline void boxFilter(CVector<T>& aVector, int aWidth) {
  CVector<T> aTemp(aVector);
  boxFilter(aTemp,aVector,aWidth);
}

template <class T>
void boxFilter(const CVector<T>& aVector, CVector<T>& aResult, int aWidth) {
  if (aWidth % 2 == 0) aWidth += 1;
  T* invWidth = new T[aWidth+1];
  invWidth[0] = 1.0f;
  for (int i = 1; i <= aWidth; i++)
    invWidth[i] = 1.0/i;
  int halfWidth = (aWidth >> 1);
  int aRight = halfWidth;
  if (aRight >= aVector.size()) aRight = aVector.size()-1;
  // Initialize
  T aSum = 0.0f;
  for (int x = 0; x <= aRight; x++)
    aSum += aVector(x);
  int aNum = aRight+1;
  // Shift
  for (int x = 0; x < aVector.size(); x++) {
    aResult(x) = aSum*invWidth[aNum];
    if (x-halfWidth >= 0) {
      aSum -= aVector(x-halfWidth); aNum--;
    }
    if (x+halfWidth+1 < aVector.size()) {
      aSum += aVector(x+halfWidth+1); aNum++;
    }
  }
  delete[] invWidth;
}

// 2D linear filtering ---------------------------------------------------------

template <class T>
inline void filter(CMatrix<T>& aMatrix, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filter(aMatrix,tempMatrix,aFilterX,1);
  filter(tempMatrix,aMatrix,1,aFilterY);
}

template <class T>
inline void filter(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filter(aMatrix,tempMatrix,aFilterX,1);
  filter(tempMatrix,aResult,1,aFilterY);
}

template <class T>
inline void filter(CMatrix<T>& aMatrix, const CFilter<T>& aFilter, const int aDummy) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filter(aMatrix,tempMatrix,aFilter,1);
  aMatrix = tempMatrix;
}

template <class T>
void filter(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const CFilter<T>& aFilter, const int aDummy) {
  if (aResult.xSize() != aMatrix.xSize() || aResult.ySize() != aMatrix.ySize())
    throw EFilterIncompatibleSize(aMatrix.xSize()*aMatrix.ySize(),aResult.xSize()*aResult.ySize());
  int x1 = -aFilter.A();
  int x2 = aMatrix.xSize()-aFilter.B();
  int a2Size = 2*aMatrix.xSize()-1;
  aResult = 0;
  for (int y = 0; y < aMatrix.ySize(); y++) {
    int aOffset = y*aMatrix.xSize();
    // Left rim
    for (int x = 0; x < x1; x++)
      for (int i = aFilter.A(); i < aFilter.B(); i++) {
        if (x+i < 0) aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[aOffset-1-x-i];
        else if (x+i >= aMatrix.xSize()) aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[aOffset+a2Size-x-i];
        else aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[aOffset+x+i];
      }
    // Center
    for (int x = x1; x < x2; x++)
      for (int i = aFilter.A(); i < aFilter.B(); i++)
        aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[aOffset+x+i];
    // Right rim
    for (int x = x2; x < aMatrix.xSize(); x++)
      for (int i = aFilter.A(); i < aFilter.B(); i++) {
        if (x+i < 0) aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[aOffset-1-x-i];
        else if (x+i >= aMatrix.xSize()) aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[aOffset+a2Size-x-i];
        else aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[aOffset+x+i];
      }
  }
}

template <class T>
inline void filter(CMatrix<T>& aMatrix, const int aDummy, const CFilter<T>& aFilter) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filter(aMatrix,tempMatrix,1,aFilter);
  aMatrix = tempMatrix;
}

template <class T>
void filter(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const int aDummy, const CFilter<T>& aFilter) {
  if (aResult.xSize() != aMatrix.xSize() || aResult.ySize() != aMatrix.ySize())
    throw EFilterIncompatibleSize(aMatrix.xSize()*aMatrix.ySize(),aResult.xSize()*aResult.ySize());
  int y1 = -aFilter.A();
  int y2 = aMatrix.ySize()-aFilter.B();
  int a2Size = 2*aMatrix.ySize()-1;
  // Upper rim
  for (int y = 0; y < y1; y++)
    for (int x = 0; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.A(); j < aFilter.B(); j++) {
        if (y+j < 0) aResult(x,y) += aFilter[j]*aMatrix(x,-1-y-j);
        else if (y+j >= aMatrix.ySize()) aResult(x,y) += aFilter[j]*aMatrix(x,a2Size-y-j);
        else aResult(x,y) += aFilter[j]*aMatrix(x,y+j);
      }
    }
  // Lower rim
  for (int y = y2; y < aMatrix.ySize(); y++)
    for (int x = 0; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.A(); j < aFilter.B(); j++) {
        if (y+j < 0) aResult(x,y) += aFilter[j]*aMatrix(x,-1-y-j);
        else if (y+j >= aMatrix.ySize()) aResult(x,y) += aFilter[j]*aMatrix(x,a2Size-y-j);
        else aResult(x,y) += aFilter[j]*aMatrix(x,y+j);
      }
    }
  // Center
  for (int y = y1; y < y2; y++)
    for (int x = 0; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.A(); j < aFilter.B(); j++)
        aResult(x,y) += aFilter[j]*aMatrix(x,y+j);
    }
}

template <class T>
inline void filter(CMatrix<T>& aMatrix, const CFilter2D<T>& aFilter) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filter(aMatrix,tempMatrix,aFilter);
  aMatrix = tempMatrix;
}

template <class T>
void filter(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const CFilter2D<T>& aFilter) {
  if (aResult.xSize() != aMatrix.xSize() || aResult.ySize() != aMatrix.ySize())
    throw EFilterIncompatibleSize(aMatrix.xSize()*aMatrix.ySize(),aResult.xSize()*aResult.ySize());
  int x1 = -aFilter.AX();
  int y1 = -aFilter.AY();
  int x2 = aMatrix.xSize()-aFilter.BX();
  int y2 = aMatrix.ySize()-aFilter.BY();
  int a2XSize = 2*aMatrix.xSize()-1;
  int a2YSize = 2*aMatrix.ySize()-1;
  // Upper rim
  for (int y = 0; y < y1; y++)
    for (int x = 0; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.AY(); j < aFilter.BY(); j++) {
        int tempY;
        if (y+j < 0) tempY = -1-y-j;
        else if (y+j >= aMatrix.ySize()) tempY = a2YSize-y-j;
        else tempY = y+j;
        for (int i = aFilter.AX(); i < aFilter.BX(); i++) {
          if (x+i < 0) aResult(x,y) += aFilter(i,j)*aMatrix(-1-x-i,tempY);
          else if (x+i >= aMatrix.xSize()) aResult(x,y) += aFilter(i,j)*aMatrix(a2XSize-x-i,tempY);
          else aResult(x,y) += aFilter(i,j)*aMatrix(x+i,tempY);
        }
      }
    }
  // Lower rim
  for (int y = y2; y < aMatrix.ySize(); y++)
    for (int x = 0; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.AY(); j < aFilter.BY(); j++) {
        int tempY;
        if (y+j < 0) tempY = -1-y-j;
        else if (y+j >= aMatrix.ySize()) tempY = a2YSize-y-j;
        else tempY = y+j;
        for (int i = aFilter.AX(); i < aFilter.BX(); i++) {
          if (x+i < 0) aResult(x,y) += aFilter(i,j)*aMatrix(-1-x-i,tempY);
          else if (x+i >= aMatrix.xSize()) aResult(x,y) += aFilter(i,j)*aMatrix(a2XSize-x-i,tempY);
          else aResult(x,y) += aFilter(i,j)*aMatrix(x+i,tempY);
        }
      }
    }
  for (int y = y1; y < y2; y++) {
    // Left rim
    for (int x = 0; x < x1; x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.AY(); j < aFilter.BY(); j++) {
        for (int i = aFilter.AX(); i < aFilter.BX(); i++) {
          if (x+i < 0) aResult(x,y) += aFilter(i,j)*aMatrix(-1-x-i,y+j);
          else if (x+i >= aMatrix.xSize()) aResult(x,y) += aFilter(i,j)*aMatrix(a2XSize-x-i,y+j);
          else aResult(x,y) += aFilter(i,j)*aMatrix(x+i,y+j);
        }
      }
    }
    // Right rim
    for (int x = x2; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.AY(); j < aFilter.BY(); j++) {
        for (int i = aFilter.AX(); i < aFilter.BX(); i++) {
          if (x+i < 0) aResult(x,y) += aFilter(i,j)*aMatrix(-1-x-i,y+j);
          else if (x+i >= aMatrix.xSize()) aResult(x,y) += aFilter(i,j)*aMatrix(a2XSize-x-i,y+j);
          else aResult(x,y) += aFilter(i,j)*aMatrix(x+i,y+j);
        }
      }
    }
  }
  // Center
  for (int y = y1; y < y2; y++)
    for (int x = x1; x < x2; x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.AY(); j < aFilter.BY(); j++)
        for (int i = aFilter.AX(); i < aFilter.BX(); i++)
          aResult(x,y) += aFilter(i,j)*aMatrix(x+i,y+j);
    }
}



template <class T>
inline void filterMin(CMatrix<T>& aMatrix, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filterMin(aMatrix,aMatrix,tempMatrix,aFilterX,1);
  CMatrix<T> tmp(aMatrix);
  filterMin(tempMatrix,tmp,aMatrix,1,aFilterY);
}

template <class T>
inline void filterMin(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filterMin(aMatrix,aMatrix,tempMatrix,aFilterX,1);
  filterMin(tempMatrix,aMatrix,aResult,1,aFilterY);
}

template <class T>
inline void filterMin(CMatrix<T>& aMatrix, const CFilter<T>& aFilter, const int aDummy) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filterMin(aMatrix,aMatrix,tempMatrix,aFilter,1);
  aMatrix = tempMatrix;
}

template <class T>
void filterMin(const CMatrix<T>& aMatrix, const CMatrix<T>& aOrig, CMatrix<T>& aResult, const CFilter<T>& aFilter, const int aDummy) {
  if (aResult.xSize() != aMatrix.xSize() || aResult.ySize() != aMatrix.ySize())
    throw EFilterIncompatibleSize(aMatrix.xSize()*aMatrix.ySize(),aResult.xSize()*aResult.ySize());
  int x1 = -aFilter.A();
  int x2 = aMatrix.xSize()-aFilter.B();
  int a2Size = 2*aMatrix.xSize()-1;
  aResult = 0;
  for (int y = 0; y < aMatrix.ySize(); y++) {
    int aOffset = y*aMatrix.xSize();
    // Left rim
    for (int x = 0; x < x1; x++)
      for (int i = aFilter.A(); i < aFilter.B(); i++) {
        int matrixIdx;
        if (x+i < 0) matrixIdx = aOffset-1-x-i;
        else if (x+i >= aMatrix.xSize()) matrixIdx = aOffset+a2Size-x-i;
        else matrixIdx = aOffset+x+i;
        if (matrixIdx == aOffset+x || aOrig.data()[matrixIdx] - 1e-5 <= aOrig.data()[aOffset+x])
          aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[matrixIdx];
      }
    // Center
    for (int x = x1; x < x2; x++)
      for (int i = aFilter.A(); i < aFilter.B(); i++)
        if (i == 0 || aOrig.data()[aOffset+x+i] - 1e-5 <= aOrig.data()[aOffset+x])
          aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[aOffset+x+i];
    // Right rim
    for (int x = x2; x < aMatrix.xSize(); x++)
      for (int i = aFilter.A(); i < aFilter.B(); i++) {
        int matrixIdx;
        if (x+i < 0) matrixIdx = aOffset-1-x-i;
        else if (x+i >= aMatrix.xSize()) matrixIdx = aOffset+a2Size-x-i;
        else matrixIdx = aOffset+x+i;
        if (matrixIdx == aOffset+x || aOrig.data()[matrixIdx] - 1e-5 <= aOrig.data()[aOffset+x])
          aResult.data()[aOffset+x] += aFilter[i]*aMatrix.data()[matrixIdx];
      }
  }
}

template <class T>
inline void filterMin(CMatrix<T>& aMatrix, const int aDummy, const CFilter<T>& aFilter) {
  CMatrix<T> tempMatrix(aMatrix.xSize(),aMatrix.ySize());
  filterMin(aMatrix, aMatrix,tempMatrix,1,aFilter);
  aMatrix = tempMatrix;
}

template <class T>
void filterMin(const CMatrix<T>& aMatrix, const CMatrix<T>& aOrig, CMatrix<T>& aResult, const int aDummy, const CFilter<T>& aFilter) {
  if (aResult.xSize() != aMatrix.xSize() || aResult.ySize() != aMatrix.ySize())
    throw EFilterIncompatibleSize(aMatrix.xSize()*aMatrix.ySize(),aResult.xSize()*aResult.ySize());
  int y1 = -aFilter.A();
  int y2 = aMatrix.ySize()-aFilter.B();
  int a2Size = 2*aMatrix.ySize()-1;
  // Upper rim
  for (int y = 0; y < y1; y++)
    for (int x = 0; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.A(); j < aFilter.B(); j++) {
        int matrixIdx;
        if (y+j < 0) matrixIdx = -1-y-j;
        else if (y+j >= aMatrix.ySize()) matrixIdx = a2Size-y-j;
        else matrixIdx = y+j;
        if (matrixIdx == y || aOrig(x, matrixIdx) - 1e-5 <= aOrig(x, y))
          aResult(x,y) += aFilter[j]*aMatrix(x,matrixIdx);
      }
    }
  // Lower rim
  for (int y = y2; y < aMatrix.ySize(); y++)
    for (int x = 0; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.A(); j < aFilter.B(); j++) {
        int matrixIdx;
        if (y+j < 0) matrixIdx = -1-y-j;
        else if (y+j >= aMatrix.ySize()) matrixIdx = a2Size-y-j;
        else matrixIdx = y+j;
        if (matrixIdx == y || aOrig(x, matrixIdx) - 1e-5 <= aOrig(x, y))
          aResult(x,y) += aFilter[j]*aMatrix(x,matrixIdx);
      }
    }
  // Center
  for (int y = y1; y < y2; y++)
    for (int x = 0; x < aMatrix.xSize(); x++) {
      aResult(x,y) = 0;
      for (int j = aFilter.A(); j < aFilter.B(); j++)
        if (j == 0 || aOrig(x,y+j) - 1e-5 <= aOrig(x,y))
          aResult(x,y) += aFilter[j]*aMatrix(x,y+j);
    }
}



// boxfilterX
template <class T>
inline void boxFilterX(CMatrix<T>& aMatrix, int aWidth) {
  CMatrix<T> aTemp(aMatrix);
  boxFilterX(aTemp,aMatrix,aWidth);
}

template <class T>
void boxFilterX(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, int aWidth) {
  if (aWidth & 1 == 0) aWidth += 1;
  T invWidth = 1.0/aWidth;
  int halfWidth = (aWidth >> 1);
  int aRight = halfWidth;
  if (aRight >= aMatrix.xSize()) aRight = aMatrix.xSize()-1;
  for (int y = 0; y < aMatrix.ySize(); y++) {
    int aOffset = y*aMatrix.xSize();
    // Initialize
    T aSum = 0.0f;
    for (int x = aRight-aWidth+1; x <= aRight; x++)
      if (x < 0) aSum += aMatrix.data()[aOffset-x-1];
      else aSum += aMatrix.data()[aOffset+x];
    // Shift
    int xm = -halfWidth;
    int xp = halfWidth+1;
    for (int x = 0; x < aMatrix.xSize(); x++,xm++,xp++) {
      aResult.data()[aOffset+x] = aSum*invWidth;
      if (xm < 0) aSum -= aMatrix.data()[aOffset-xm-1];
      else aSum -= aMatrix.data()[aOffset+xm];
      if (xp >= aMatrix.xSize()) aSum += aMatrix.data()[aOffset+2*aMatrix.xSize()-1-xp];
      else aSum += aMatrix.data()[aOffset+xp];
    }
  }
}

// boxfilterY
template <class T>
inline void boxFilterY(CMatrix<T>& aMatrix, int aWidth) {
  CMatrix<T> aTemp(aMatrix);
  boxFilterY(aTemp,aMatrix,aWidth);
}

template <class T>
void boxFilterY(const CMatrix<T>& aMatrix, CMatrix<T>& aResult, int aWidth) {
  if (aWidth & 1 == 0) aWidth += 1;
  T invWidth = 1.0/aWidth;
  int halfWidth = (aWidth >> 1);
  int aBottom = halfWidth;
  if (aBottom >= aMatrix.ySize()) aBottom = aMatrix.xSize()-1;
  for (int x = 0; x < aMatrix.xSize(); x++) {
    // Initialize
    T aSum = 0.0f;
    for (int y = aBottom-aWidth+1; y <= aBottom; y++)
      if (y < 0) aSum += aMatrix(x,-1-y);
      else aSum += aMatrix(x,y);
    // Shift
    int ym = -halfWidth;
    int yp = halfWidth+1;
    for (int y = 0; y < aMatrix.ySize(); y++,ym++,yp++) {
      aResult(x,y) = aSum*invWidth;
      if (ym < 0) aSum -= aMatrix(x,-1-ym);
      else aSum -= aMatrix(x,ym);
      if (yp >= aMatrix.ySize()) aSum += aMatrix(x,2*aMatrix.ySize()-1-yp);
      else aSum += aMatrix(x,yp);
    }
  }
}

template <class T>
void recursiveSmoothX(CMatrix<T>& aMatrix, float aSigma) {
  CVector<T> aVals1(aMatrix.xSize());
  CVector<T> aVals2(aMatrix.xSize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int y = 0; y < aMatrix.ySize(); y++) {
    aVals1(0) = (0.5f-k*aPreMinus)*aMatrix(0,y);
    aVals1(1) = k*(aMatrix(1,y)+aPreMinus*aMatrix(0,y))+(a2Exp-aExpSqr)*aVals1(0);
    for (int x = 2; x < aMatrix.xSize(); x++)
      aVals1(x) = k*(aMatrix(x,y)+aPreMinus*aMatrix(x-1,y))+a2Exp*aVals1(x-1)-aExpSqr*aVals1(x-2);
    aVals2(aMatrix.xSize()-1) = (0.5f+k*aPreMinus)*aMatrix(aMatrix.xSize()-1,y);
    aVals2(aMatrix.xSize()-2) = k*((aPrePlus-aExpSqr)*aMatrix(aMatrix.xSize()-1,y))+(a2Exp-aExpSqr)*aVals2(aMatrix.xSize()-1);
    for (int x = aMatrix.xSize()-3; x >= 0; x--)
      aVals2(x) = k*(aPrePlus*aMatrix(x+1,y)-aExpSqr*aMatrix(x+2,y))+a2Exp*aVals2(x+1)-aExpSqr*aVals2(x+2);
    for (int x = 0; x < aMatrix.xSize(); x++)
      aMatrix(x,y) = aVals1(x)+aVals2(x);
  }
}

template <class T>
void recursiveSmoothY(CMatrix<T>& aMatrix, float aSigma) {
  CVector<T> aVals1(aMatrix.ySize());
  CVector<T> aVals2(aMatrix.ySize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int x = 0; x < aMatrix.xSize(); x++) {
    aVals1(0) = (0.5f-k*aPreMinus)*aMatrix(x,0);
    aVals1(1) = k*(aMatrix(x,1)+aPreMinus*aMatrix(x,0))+(a2Exp-aExpSqr)*aVals1(0);
    for (int y = 2; y < aMatrix.ySize(); y++)
      aVals1(y) = k*(aMatrix(x,y)+aPreMinus*aMatrix(x,y-1))+a2Exp*aVals1(y-1)-aExpSqr*aVals1(y-2);
    aVals2(aMatrix.ySize()-1) = (0.5f+k*aPreMinus)*aMatrix(x,aMatrix.ySize()-1);
    aVals2(aMatrix.ySize()-2) = k*((aPrePlus-aExpSqr)*aMatrix(x,aMatrix.ySize()-1))+(a2Exp-aExpSqr)*aVals2(aMatrix.ySize()-1);
    for (int y = aMatrix.ySize()-3; y >= 0; y--)
      aVals2(y) = k*(aPrePlus*aMatrix(x,y+1)-aExpSqr*aMatrix(x,y+2))+a2Exp*aVals2(y+1)-aExpSqr*aVals2(y+2);
    for (int y = 0; y < aMatrix.ySize(); y++)
      aMatrix(x,y) = aVals1(y)+aVals2(y);
  }
}

template <class T>
inline void recursiveSmooth(CMatrix<T>& aMatrix, float aSigma) {
  recursiveSmoothX(aMatrix,aSigma);
  recursiveSmoothY(aMatrix,aSigma);
}

// Linear 3D filtering ---------------------------------------------------------

template <class T>
inline void filter(CTensor<T>& aTensor, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY, const CFilter<T>& aFilterZ) {
  CTensor<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize());
  filter(aTensor,tempTensor,aFilterX,1,1);
  filter(tempTensor,aTensor,1,aFilterY,1);
  filter(aTensor,tempTensor,1,1,aFilterZ);
  aTensor = tempTensor;
}

template <class T>
inline void filter(const CTensor<T>& aTensor, CTensor<T>& aResult, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY, const CFilter<T>& aFilterZ) {
  CTensor<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize());
  filter(aTensor,aResult,aFilterX,1,1);
  filter(aResult,tempTensor,1,aFilterY,1);
  filter(tempTensor,aResult,1,1,aFilterZ);
}

template <class T>
inline void filter(CTensor<T>& aTensor, const CFilter<T>& aFilter, const int aDummy1, const int aDummy2) {
  CTensor<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize());
  filter(aTensor,tempTensor,aFilter,1,1);
  aTensor = tempTensor;
}

template <class T>
void filter(const CTensor<T>& aTensor, CTensor<T>& aResult, const CFilter<T>& aFilter, const int aDummy1, const int aDummy2) {
  if (aResult.xSize() != aTensor.xSize() || aResult.ySize() != aTensor.ySize() || aResult.zSize() != aTensor.zSize())
    throw EFilterIncompatibleSize(aTensor.xSize()*aTensor.ySize()*aTensor.zSize(),aResult.xSize()*aResult.ySize()*aResult.zSize());
  int x1 = -aFilter.A();
  int x2 = aTensor.xSize()-aFilter.B();
  int a2Size = 2*aTensor.xSize()-1;
  for (int z = 0; z < aTensor.zSize(); z++)
    for (int y = 0; y < aTensor.ySize(); y++) {
      // Left rim
      for (int x = 0; x < x1; x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++) {
          if (x+i < 0) aResult(x,y,z) += aFilter[i]*aTensor(-1-x-i,y,z);
          else if (x+i >= aTensor.xSize()) aResult(x,y,z) += aFilter[i]*aTensor(a2Size-x-i,y,z);
          else aResult(x,y,z) += aFilter[i]*aTensor(x+i,y,z);
        }
      }
      // Center
      for (int x = x1; x < x2; x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++)
          aResult(x,y,z) += aFilter[i]*aTensor(x+i,y,z);
      }
      // Right rim
      for (int x = x2; x < aTensor.xSize(); x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++) {
          if (x+i < 0) aResult(x,y,z) += aFilter[i]*aTensor(-1-x-i,y,z);
          else if (x+i >= aTensor.xSize()) aResult(x,y,z) += aFilter[i]*aTensor(a2Size-x-i,y,z);
          else aResult(x,y,z) += aFilter[i]*aTensor(x+i,y,z);
        }
      }
    }
}

template <class T>
inline void filter(CTensor<T>& aTensor, const int aDummy1, const CFilter<T>& aFilter, const int aDummy2) {
  CTensor<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize());
  filter(aTensor,tempTensor,1,aFilter,1);
  aTensor = tempTensor;
}

template <class T>
void filter(const CTensor<T>& aTensor, CTensor<T>& aResult, const int aDummy1, const CFilter<T>& aFilter, const int aDummy2) {
  if (aResult.xSize() != aTensor.xSize() || aResult.ySize() != aTensor.ySize() || aResult.zSize() != aTensor.zSize())
    throw EFilterIncompatibleSize(aTensor.xSize()*aTensor.ySize()*aTensor.zSize(),aResult.xSize()*aResult.ySize()*aResult.zSize());
  int y1 = -aFilter.A();
  int y2 = aTensor.ySize()-aFilter.B();
  int a2Size = 2*aTensor.ySize()-1;
  for (int z = 0; z < aTensor.zSize(); z++) {
    // Upper rim
    for (int y = 0; y < y1; y++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++) {
          if (y+i < 0) aResult(x,y,z) += aFilter[i]*aTensor(x,-1-y-i,z);
          else if (y+i >= aTensor.ySize()) aResult(x,y,z) += aFilter[i]*aTensor(x,a2Size-y-i,z);
          else aResult(x,y,z) += aFilter[i]*aTensor(x,y+i,z);
        }
      }
    // Lower rim
    for (int y = y2; y < aTensor.ySize(); y++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++) {
          if (y+i < 0) aResult(x,y,z) += aFilter[i]*aTensor(x,-1-y-i,z);
          else if (y+i >= aTensor.ySize()) aResult(x,y,z) += aFilter[i]*aTensor(x,a2Size-y-i,z);
          else aResult(x,y,z) += aFilter[i]*aTensor(x,y+i,z);
        }
      }
  }
  // Center
  for (int z = 0; z < aTensor.zSize(); z++)
    for (int y = y1; y < y2; y++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++)
          aResult(x,y,z) += aFilter[i]*aTensor(x,y+i,z);
      }
}

template <class T>
inline void filter(CTensor<T>& aTensor, const int aDummy1, const int aDummy2, const CFilter<T>& aFilter) {
  CTensor<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize());
  filter(aTensor,tempTensor,1,1,aFilter);
  aTensor = tempTensor;
}

template <class T>
void filter(const CTensor<T>& aTensor, CTensor<T>& aResult, const int aDummy1, const int aDummy2, const CFilter<T>& aFilter) {
  if (aResult.xSize() != aTensor.xSize() || aResult.ySize() != aTensor.ySize() || aResult.zSize() != aTensor.zSize())
    throw EFilterIncompatibleSize(aTensor.xSize()*aTensor.ySize()*aTensor.zSize(),aResult.xSize()*aResult.ySize()*aResult.zSize());
  int z1 = -aFilter.A();
  int z2 = aTensor.zSize()-aFilter.B();
  if (z2 < 0) z2 = 0;
  int a2Size = 2*aTensor.zSize()-1;
  // Front rim
  for (int z = 0; z < z1; z++)
    for (int y = 0; y < aTensor.ySize(); y++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++) {
          if (z+i < 0) aResult(x,y,z) += aFilter[i]*aTensor(x,y,-1-z-i);
          else if (z+i >= aTensor.zSize()) aResult(x,y,z) += aFilter[i]*aTensor(x,y,a2Size-z-i);
          else aResult(x,y,z) += aFilter[i]*aTensor(x,y,z+i);
        }
      }
  // Back rim
  for (int z = z2; z < aTensor.zSize(); z++)
    for (int y = 0; y < aTensor.ySize(); y++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++) {
          if (z+i < 0) aResult(x,y,z) += aFilter[i]*aTensor(x,y,-1-z-i);
          else if (z+i >= aTensor.zSize()) aResult(x,y,z) += aFilter[i]*aTensor(x,y,a2Size-z-i);
          else aResult(x,y,z) += aFilter[i]*aTensor(x,y,z+i);
        }
      }
  // Center
  for (int z = z1; z < z2; z++)
    for (int y = 0; y < aTensor.ySize(); y++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        aResult(x,y,z) = 0;
        for (int i = aFilter.A(); i < aFilter.B(); i++)
          aResult(x,y,z) += aFilter[i]*aTensor(x,y,z+i);
      }
}

// boxfilterX
template <class T>
inline void boxFilterX(CTensor<T>& aTensor, int aWidth) {
  CTensor<T> aTemp(aTensor);
  boxFilterX(aTemp,aTensor,aWidth);
}

template <class T>
void boxFilterX(const CTensor<T>& aTensor, CTensor<T>& aResult, int aWidth) {
  if (aWidth % 2 == 0) aWidth += 1;
  T* invWidth = new T[aWidth+1];
  invWidth[0] = 1.0f;
  for (int i = 1; i <= aWidth; i++)
    invWidth[i] = 1.0/i;
  int halfWidth = (aWidth >> 1);
  int aRight = halfWidth;
  if (aRight >= aTensor.xSize()) aRight = aTensor.xSize()-1;
  for (int z = 0; z < aTensor.zSize(); z++)
    for (int y = 0; y < aTensor.ySize(); y++) {
      int aOffset = (z*aTensor.ySize()+y)*aTensor.xSize();
      // Initialize
      int aNum = 0;
      T aSum = 0.0f;
      for (int x = 0; x <= aRight; x++) {
        aSum += aTensor.data()[aOffset+x]; aNum++;
      }
      // Shift
      for (int x = 0; x < aTensor.xSize(); x++) {
        aResult.data()[aOffset+x] = aSum*invWidth[aNum];
        if (x-halfWidth >= 0) {
          aSum -= aTensor.data()[aOffset+x-halfWidth]; aNum--;
        }
        if (x+halfWidth+1 < aTensor.xSize()) {
          aSum += aTensor.data()[aOffset+x+halfWidth+1]; aNum++;
        }
      }
    }
  delete[] invWidth;
}

// boxfilterY
template <class T>
inline void boxFilterY(CTensor<T>& aTensor, int aWidth) {
  CTensor<T> aTemp(aTensor);
  boxFilterY(aTemp,aTensor,aWidth);
}

template <class T>
void boxFilterY(const CTensor<T>& aTensor, CTensor<T>& aResult, int aWidth) {
  if (aWidth % 2 == 0) aWidth += 1;
  T* invWidth = new T[aWidth+1];
  invWidth[0] = 1.0f;
  for (int i = 1; i <= aWidth; i++)
    invWidth[i] = 1.0/i;
  int halfWidth = (aWidth >> 1);
  int aBottom = halfWidth;
  if (aBottom >= aTensor.ySize()) aBottom = aTensor.ySize()-1;
  for (int z = 0; z < aTensor.zSize(); z++)
    for (int x = 0; x < aTensor.xSize(); x++) {
      // Initialize
      int aNum = 0;
      T aSum = 0.0f;
      for (int y = 0; y <= aBottom; y++) {
        aSum += aTensor(x,y,z); aNum++;
      }
      // Shift
      for (int y = 0; y < aTensor.ySize(); y++) {
        aResult(x,y,z) = aSum*invWidth[aNum];
        if (y-halfWidth >= 0) {
          aSum -= aTensor(x,y-halfWidth,z); aNum--;
        }
        if (y+halfWidth+1 < aTensor.ySize()) {
          aSum += aTensor(x,y+halfWidth+1,z); aNum++;
        }
      }
    }
  delete[] invWidth;
}

// boxfilterZ
template <class T>
inline void boxFilterZ(CTensor<T>& aTensor, int aWidth) {
  CTensor<T> aTemp(aTensor);
  boxFilterZ(aTemp,aTensor,aWidth);
}

template <class T>
void boxFilterZ(const CTensor<T>& aTensor, CTensor<T>& aResult, int aWidth) {
  if (aWidth % 2 == 0) aWidth += 1;
  T* invWidth = new T[aWidth+1];
  invWidth[0] = 1.0f;
  for (int i = 1; i <= aWidth; i++)
    invWidth[i] = 1.0/i;
  int halfWidth = (aWidth >> 1);
  int aBottom = halfWidth;
  if (aBottom >= aTensor.zSize()) aBottom = aTensor.zSize()-1;
  for (int y = 0; y < aTensor.ySize(); y++)
    for (int x = 0; x < aTensor.xSize(); x++) {
      // Initialize
      int aNum = 0;
      T aSum = 0.0f;
      for (int z = 0; z <= aBottom; z++) {
        aSum += aTensor(x,y,z); aNum++;
      }
      // Shift
      for (int z = 0; z < aTensor.zSize(); z++) {
        aResult(x,y,z) = aSum*invWidth[aNum];
        if (z-halfWidth >= 0) {
          aSum -= aTensor(x,y,z-halfWidth); aNum--;
        }
        if (z+halfWidth+1 < aTensor.zSize()) {
          aSum += aTensor(x,y,z+halfWidth+1); aNum++;
        }
      }
    }
  delete[] invWidth;
}

template <class T>
void recursiveSmoothX(CTensor<T>& aTensor, float aSigma) {
  CVector<T> aVals1(aTensor.xSize());
  CVector<T> aVals2(aTensor.xSize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int z = 0; z < aTensor.zSize(); z++)
    for (int y = 0; y < aTensor.ySize(); y++) {
      int aOffset = (z*aTensor.ySize()+y)*aTensor.xSize();
      aVals1(0) = (0.5-k*aPreMinus)*aTensor.data()[aOffset];
      aVals1(1) = k*(aTensor.data()[aOffset+1]+aPreMinus*aTensor.data()[aOffset])+(2.0*aExp-aExpSqr)*aVals1(0);
      for (int x = 2; x < aTensor.xSize(); x++)
        aVals1(x) = k*(aTensor.data()[aOffset+x]+aPreMinus*aTensor.data()[aOffset+x-1])+a2Exp*aVals1(x-1)-aExpSqr*aVals1(x-2);
      aVals2(aTensor.xSize()-1) = (0.5+k*aPreMinus)*aTensor.data()[aOffset+aTensor.xSize()-1];
      aVals2(aTensor.xSize()-2) = k*((aPrePlus-aExpSqr)*aTensor.data()[aOffset+aTensor.xSize()-1])+(a2Exp-aExpSqr)*aVals2(aTensor.xSize()-1);
      for (int x = aTensor.xSize()-3; x >= 0; x--)
        aVals2(x) = k*(aPrePlus*aTensor.data()[aOffset+x+1]-aExpSqr*aTensor.data()[aOffset+x+2])+a2Exp*aVals2(x+1)-aExpSqr*aVals2(x+2);
      for (int x = 0; x < aTensor.xSize(); x++)
        aTensor.data()[aOffset+x] = aVals1(x)+aVals2(x);
    }
}

template <class T>
void recursiveSmoothY(CTensor<T>& aTensor, float aSigma) {
  CVector<T> aVals1(aTensor.ySize());
  CVector<T> aVals2(aTensor.ySize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int z = 0; z < aTensor.zSize(); z++)
    for (int x = 0; x < aTensor.xSize(); x++) {
      aVals1(0) = (0.5-k*aPreMinus)*aTensor(x,0,z);
      aVals1(1) = k*(aTensor(x,1,z)+aPreMinus*aTensor(x,0,z))+(2.0*aExp-aExpSqr)*aVals1(0);
      for (int y = 2; y < aTensor.ySize(); y++)
        aVals1(y) = k*(aTensor(x,y,z)+aPreMinus*aTensor(x,y-1,z))+a2Exp*aVals1(y-1)-aExpSqr*aVals1(y-2);
      aVals2(aTensor.ySize()-1) = (0.5+k*aPreMinus)*aTensor(x,aTensor.ySize()-1,z);
      aVals2(aTensor.ySize()-2) = k*((aPrePlus-aExpSqr)*aTensor(x,aTensor.ySize()-1,z))+(a2Exp-aExpSqr)*aVals2(aTensor.ySize()-1);
      for (int y = aTensor.ySize()-3; y >= 0; y--)
        aVals2(y) = k*(aPrePlus*aTensor(x,y+1,z)-aExpSqr*aTensor(x,y+2,z))+a2Exp*aVals2(y+1)-aExpSqr*aVals2(y+2);
      for (int y = 0; y < aTensor.ySize(); y++)
        aTensor(x,y,z) = aVals1(y)+aVals2(y);
    }
}

template <class T>
void recursiveSmoothZ(CTensor<T>& aTensor, float aSigma) {
  CVector<T> aVals1(aTensor.zSize());
  CVector<T> aVals2(aTensor.zSize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int y = 0; y < aTensor.ySize(); y++)
    for (int x = 0; x < aTensor.xSize(); x++) {
      aVals1(0) = (0.5-k*aPreMinus)*aTensor(x,y,0);
      aVals1(1) = k*(aTensor(x,y,1)+aPreMinus*aTensor(x,y,0))+(2.0*aExp-aExpSqr)*aVals1(0);
      for (int z = 2; z < aTensor.zSize(); z++)
        aVals1(z) = k*(aTensor(x,y,z)+aPreMinus*aTensor(x,y,z-1))+a2Exp*aVals1(z-1)-aExpSqr*aVals1(z-2);
      aVals2(aTensor.zSize()-1) = (0.5+k*aPreMinus)*aTensor(x,y,aTensor.zSize()-1);
      aVals2(aTensor.zSize()-2) = k*((aPrePlus-aExpSqr)*aTensor(x,y,aTensor.zSize()-1))+(a2Exp-aExpSqr)*aVals2(aTensor.zSize()-1);
      for (int z = aTensor.zSize()-3; z >= 0; z--)
        aVals2(z) = k*(aPrePlus*aTensor(x,y,z+1)-aExpSqr*aTensor(x,y,z+2))+a2Exp*aVals2(z+1)-aExpSqr*aVals2(z+2);
      for (int z = 0; z < aTensor.zSize(); z++)
        aTensor(x,y,z) = aVals1(z)+aVals2(z);
    }
}

// Linear 4D filtering ---------------------------------------------------------

template <class T>
inline void filter(CTensor4D<T>& aTensor, const CFilter<T>& aFilterX, const CFilter<T>& aFilterY, const CFilter<T>& aFilterZ, const CFilter<T>& aFilterA) {
  CTensor4D<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize());
  filter(aTensor,tempTensor,aFilterX,1,1,1);
  filter(tempTensor,aTensor,1,aFilterY,1,1);
  filter(aTensor,tempTensor,1,1,aFilterZ,1);
  filter(tempTensor,aTensor,1,1,1,aFilterA);
}

template <class T>
inline void filter(CTensor4D<T>& aTensor, const CFilter<T>& aFilter, const int aDummy1, const int aDummy2, const int aDummy3) {
  CTensor4D<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),aTensor.aSize());
  filter(aTensor,tempTensor,aFilter,1,1,1);
  aTensor = tempTensor;
}

template <class T>
void filter(const CTensor4D<T>& aTensor, CTensor4D<T>& aResult, const CFilter<T>& aFilter, const int aDummy1, const int aDummy2, const int aDummy3) {
  if (aResult.xSize() != aTensor.xSize() || aResult.ySize() != aTensor.ySize() || aResult.zSize() != aTensor.zSize() || aResult.aSize() != aTensor.aSize())
    throw EFilterIncompatibleSize(aTensor.xSize()*aTensor.ySize()*aTensor.zSize()*aTensor.aSize(),aResult.xSize()*aResult.ySize()*aResult.zSize()*aResult.aSize());
  int x1 = -aFilter.A();
  int x2 = aTensor.xSize()-aFilter.B();
  int a2Size = 2*aTensor.xSize()-1;
  aResult = 0;
  for (int a = 0; a < aTensor.aSize(); a++)
    for (int z = 0; z < aTensor.zSize(); z++)
      for (int y = 0; y < aTensor.ySize(); y++) {
        int aOffset = aTensor.xSize()*(y+aTensor.ySize()*(z+aTensor.zSize()*a));
        // Left rim
        for (int x = 0; x < x1; x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++) {
            if (x+i < 0) aResult.data()[aOffset+x] += aFilter[i]*aTensor.data()[aOffset-1-x-i];
            else if (x+i >= aTensor.xSize()) aResult.data()[aOffset+x] += aFilter[i]*aTensor.data()[aOffset+a2Size-x-i];
            else aResult.data()[aOffset+x] += aFilter[i]*aTensor.data()[x+i+aOffset];
          }
        // Center
        for (int x = x1; x < x2; x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++)
            aResult.data()[aOffset+x] += aFilter[i]*aTensor.data()[aOffset+x+i];
        // Right rim
        for (int x = x2; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++) {
            if (x+i < 0) aResult.data()[aOffset+x] += aFilter[i]*aTensor.data()[aOffset-1-x-i];
            else if (x+i >= aTensor.xSize()) aResult.data()[aOffset+x] += aFilter[i]*aTensor.data()[aOffset+a2Size-x-i];
            else aResult.data()[aOffset+x] += aFilter[i]*aTensor.data()[x+i+aOffset];
          }
      }
}

template <class T>
inline void filter(CTensor4D<T>& aTensor, const int aDummy1, const CFilter<T>& aFilter, const int aDummy2, const int aDummy3) {
  CTensor4D<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),aTensor.aSize());
  filter(aTensor,tempTensor,1,aFilter,1,1);
  aTensor = tempTensor;
}

template <class T>
void filter(const CTensor4D<T>& aTensor, CTensor4D<T>& aResult, const int aDummy1, const CFilter<T>& aFilter, const int aDummy2, const int aDummy3) {
  if (aResult.xSize() != aTensor.xSize() || aResult.ySize() != aTensor.ySize() || aResult.zSize() != aTensor.zSize() || aResult.aSize() != aTensor.aSize())
    throw EFilterIncompatibleSize(aTensor.xSize()*aTensor.ySize()*aTensor.zSize()*aTensor.aSize(),aResult.xSize()*aResult.ySize()*aResult.zSize()*aResult.aSize());
  int y1 = -aFilter.A();
  int y2 = aTensor.ySize()-aFilter.B();
  int a2Size = 2*aTensor.ySize()-1;
  aResult = 0;
  for (int a = 0; a < aTensor.aSize(); a++) {
    for (int z = 0; z < aTensor.zSize(); z++) {
      // Upper rim
      for (int y = 0; y < y1; y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++) {
            if (y+i < 0) aResult(x,y,z,a) += aFilter[i]*aTensor(x,-1-y-i,z,a);
            else if (y+i >= aTensor.ySize()) aResult(x,y,z,a) += aFilter[i]*aTensor(x,a2Size-y-i,z,a);
            else aResult(x,y,z,a) += aFilter[i]*aTensor(x,y+i,z,a);
          }
      // Lower rim
      for (int y = y2; y < aTensor.ySize(); y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++) {
            if (y+i < 0) aResult(x,y,z,a) += aFilter[i]*aTensor(x,-1-y-i,z,a);
            else if (y+i >= aTensor.ySize()) aResult(x,y,z,a) += aFilter[i]*aTensor(x,a2Size-y-i,z,a);
            else aResult(x,y,z,a) += aFilter[i]*aTensor(x,y+i,z,a);
          }
    }
    // Center
    for (int z = 0; z < aTensor.zSize(); z++)
      for (int y = y1; y < y2; y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++)
            aResult(x,y,z,a) += aFilter[i]*aTensor(x,y+i,z,a);
  }
}

template <class T>
inline void filter(CTensor4D<T>& aTensor, const int aDummy1, const int aDummy2, const CFilter<T>& aFilter, const int aDummy3) {
  CTensor4D<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),aTensor.aSize());
  filter(aTensor,tempTensor,1,1,aFilter,1);
  aTensor = tempTensor;
}

template <class T>
void filter(const CTensor4D<T>& aTensor, CTensor4D<T>& aResult, const int aDummy1, const int aDummy2, const CFilter<T>& aFilter, const int aDummy3) {
  if (aResult.xSize() != aTensor.xSize() || aResult.ySize() != aTensor.ySize() || aResult.zSize() != aTensor.zSize() || aResult.aSize() != aTensor.aSize())
    throw EFilterIncompatibleSize(aTensor.xSize()*aTensor.ySize()*aTensor.zSize()*aTensor.aSize(),aResult.xSize()*aResult.ySize()*aResult.zSize()*aResult.aSize());
  int z1 = -aFilter.A();
  int z2 = aTensor.zSize()-aFilter.B();
  int a2Size = 2*aTensor.zSize()-1;
  aResult = 0;
  for (int a = 0; a < aTensor.aSize(); a++) {
    // Front rim
    for (int z = 0; z < z1; z++)
      for (int y = 0; y < aTensor.ySize(); y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++) {
            if (z+i < 0) aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,-1-z-i,a);
            else if (z+i >= aTensor.zSize()) aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,a2Size-z-i,a);
            else aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z+i,a);
          }
    // Back rim
    for (int z = z2; z < aTensor.zSize(); z++)
      for (int y = 0; y < aTensor.ySize(); y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++) {
            if (z+i < 0) aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,-1-z-i,a);
            else if (z+i >= aTensor.zSize()) aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,a2Size-z-i,a);
            else aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z+i,a);
          }
    // Center
    for (int z = z1; z < z2; z++)
      for (int y = 0; y < aTensor.ySize(); y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++)
            aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z+i,a);
  }
}

template <class T>
inline void filter(CTensor4D<T>& aTensor, const int aDummy1, const int aDummy2, const int aDummy3, const CFilter<T>& aFilter) {
  CTensor4D<T> tempTensor(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),aTensor.aSize());
  filter(aTensor,tempTensor,1,1,1,aFilter);
  aTensor = tempTensor;
}

template <class T>
void filter(const CTensor4D<T>& aTensor, CTensor4D<T>& aResult, const int aDummy1, const int aDummy2, const int aDummy3, const CFilter<T>& aFilter) {
  if (aResult.xSize() != aTensor.xSize() || aResult.ySize() != aTensor.ySize() || aResult.zSize() != aTensor.zSize() || aResult.aSize() != aTensor.aSize())
    throw EFilterIncompatibleSize(aTensor.xSize()*aTensor.ySize()*aTensor.zSize()*aTensor.aSize(),aResult.xSize()*aResult.ySize()*aResult.zSize()*aResult.aSize());
  int a1 = -aFilter.A();
  int a2 = aTensor.aSize()-aFilter.B();
  int a2Size = 2*aTensor.aSize()-1;
  aResult = 0;
  // Front rim
  for (int a = 0; a < a1; a++)
    for (int z = 0; z < aTensor.zSize(); z++)
      for (int y = 0; y < aTensor.ySize(); y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++) {
            if (a+i < 0) aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z,-1-a-i);
            else if (a+i >= aTensor.aSize()) aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z,a2Size-a-i);
            else aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z,a+i);
          }
  // Back rim
  for (int a = a2; a < aTensor.aSize(); a++)
    for (int z = 0; z < aTensor.zSize(); z++)
      for (int y = 0; y < aTensor.ySize(); y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++) {
            if (a+i < 0) aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z,-1-a-i);
            else if (a+i >= aTensor.aSize()) aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z,a2Size-a-i);
            else aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z,a+i);
          }
  // Center
  for (int a = a1; a < a2; a++)
    for (int z = 0; z < aTensor.zSize(); z++)
      for (int y = 0; y < aTensor.ySize(); y++)
        for (int x = 0; x < aTensor.xSize(); x++)
          for (int i = aFilter.A(); i < aFilter.B(); i++)
            aResult(x,y,z,a) += aFilter[i]*aTensor(x,y,z,a+i);
}

template <class T>
void recursiveSmoothX(CTensor4D<T>& aTensor, float aSigma) {
  CVector<T> aVals1(aTensor.xSize());
  CVector<T> aVals2(aTensor.xSize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int a = 0; a < aTensor.aSize(); a++)
    for (int z = 0; z < aTensor.zSize(); z++)
      for (int y = 0; y < aTensor.ySize(); y++) {
        int aOffset = ((a*aTensor.zSize()+z)*aTensor.ySize()+y)*aTensor.xSize();
        aVals1(0) = (0.5-k*aPreMinus)*aTensor.data()[aOffset];
        aVals1(1) = k*(aTensor.data()[aOffset+1]+aPreMinus*aTensor.data()[aOffset])+(2.0*aExp-aExpSqr)*aVals1(0);
        for (int x = 2; x < aTensor.xSize(); x++)
          aVals1(x) = k*(aTensor.data()[aOffset+x]+aPreMinus*aTensor.data()[aOffset+x-1])+a2Exp*aVals1(x-1)-aExpSqr*aVals1(x-2);
        aVals2(aTensor.xSize()-1) = (0.5+k*aPreMinus)*aTensor.data()[aOffset+aTensor.xSize()-1];
        aVals2(aTensor.xSize()-2) = k*((aPrePlus-aExpSqr)*aTensor.data()[aOffset+aTensor.xSize()-1])+(a2Exp-aExpSqr)*aVals2(aTensor.xSize()-1);
        for (int x = aTensor.xSize()-3; x >= 0; x--)
          aVals2(x) = k*(aPrePlus*aTensor.data()[aOffset+x+1]-aExpSqr*aTensor.data()[aOffset+x+2])+a2Exp*aVals2(x+1)-aExpSqr*aVals2(x+2);
        for (int x = 0; x < aTensor.xSize(); x++)
          aTensor.data()[aOffset+x] = aVals1(x)+aVals2(x);
      }
}

template <class T>
void recursiveSmoothY(CTensor4D<T>& aTensor, float aSigma) {
  CVector<T> aVals1(aTensor.ySize());
  CVector<T> aVals2(aTensor.ySize());
  CVector<T> aVals3(aTensor.ySize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int a = 0; a < aTensor.aSize(); a++)
    for (int z = 0; z < aTensor.zSize(); z++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        for (int y = 0; y < aTensor.ySize(); y++)
          aVals3(y) = aTensor(x,y,z,a);
        aVals1(0) = (0.5-k*aPreMinus)*aVals3(0);
        aVals1(1) = k*(aVals3(1)+aPreMinus*aVals3(0))+(2.0*aExp-aExpSqr)*aVals1(0);
        for (int y = 2; y < aTensor.ySize(); y++)
          aVals1(y) = k*(aVals3(y)+aPreMinus*aVals3(y-1))+a2Exp*aVals1(y-1)-aExpSqr*aVals1(y-2);
        aVals2(aTensor.ySize()-1) = (0.5+k*aPreMinus)*aVals3(aTensor.ySize()-1);
        aVals2(aTensor.ySize()-2) = k*((aPrePlus-aExpSqr)*aVals3(aTensor.ySize()-1))+(a2Exp-aExpSqr)*aVals2(aTensor.ySize()-1);
        for (int y = aTensor.ySize()-3; y >= 0; y--)
          aVals2(y) = k*(aPrePlus*aVals3(y+1)-aExpSqr*aVals3(y+2))+a2Exp*aVals2(y+1)-aExpSqr*aVals2(y+2);
        for (int y = 0; y < aTensor.ySize(); y++)
          aTensor(x,y,z,a) = aVals1(y)+aVals2(y);
      }
}

template <class T>
void recursiveSmoothZ(CTensor4D<T>& aTensor, float aSigma) {
  CVector<T> aVals1(aTensor.zSize());
  CVector<T> aVals2(aTensor.zSize());
  CVector<T> aVals3(aTensor.zSize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int a = 0; a < aTensor.aSize(); a++)
    for (int y = 0; y < aTensor.ySize(); y++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        for (int z = 0; z < aTensor.zSize(); z++)
          aVals3(z) = aTensor(x,y,z,a);
        aVals1(0) = (0.5-k*aPreMinus)*aVals3(0);
        aVals1(1) = k*(aVals3(1)+aPreMinus*aVals3(0))+(2.0*aExp-aExpSqr)*aVals1(0);
        for (int z = 2; z < aTensor.zSize(); z++)
          aVals1(z) = k*(aVals3(z)+aPreMinus*aVals3(z-1))+a2Exp*aVals1(z-1)-aExpSqr*aVals1(z-2);
        aVals2(aTensor.zSize()-1) = (0.5+k*aPreMinus)*aVals3(aTensor.zSize()-1);
        aVals2(aTensor.zSize()-2) = k*((aPrePlus-aExpSqr)*aVals3(aTensor.zSize()-1))+(a2Exp-aExpSqr)*aVals2(aTensor.zSize()-1);
        for (int z = aTensor.zSize()-3; z >= 0; z--)
          aVals2(z) = k*(aPrePlus*aVals3(z+1)-aExpSqr*aVals3(z+2))+a2Exp*aVals2(z+1)-aExpSqr*aVals2(z+2);
        for (int z = 0; z < aTensor.zSize(); z++)
          aTensor(x,y,z,a) = aVals1(z)+aVals2(z);
      }
}

template <class T>
void recursiveSmoothA(CTensor4D<T>& aTensor, float aSigma) {
  CVector<T> aVals1(aTensor.aSize());
  CVector<T> aVals2(aTensor.aSize());
  CVector<T> aVals3(aTensor.aSize());
  float aAlpha = 2.5/(sqrt(NMath::Pi)*aSigma);
  float aExp = exp(-aAlpha);
  float aExpSqr = aExp*aExp;
  float a2Exp = 2.0*aExp;
  float k = (1.0-aExp)*(1.0-aExp)/(1.0+2.0*aAlpha*aExp-aExpSqr);
  float aPreMinus = aExp*(aAlpha-1.0);
  float aPrePlus = aExp*(aAlpha+1.0);
  for (int z = 0; z < aTensor.zSize(); z++)
    for (int y = 0; y < aTensor.ySize(); y++)
      for (int x = 0; x < aTensor.xSize(); x++) {
        for (int a = 0; a < aTensor.aSize(); a++)
          aVals3(a) = aTensor(x,y,z,a);
        aVals1(0) = (0.5-k*aPreMinus)*aVals3(0);
        aVals1(1) = k*(aVals3(1)+aPreMinus*aVals3(0))+(2.0*aExp-aExpSqr)*aVals1(0);
        for (int a = 2; a < aTensor.aSize(); a++)
          aVals1(a) = k*(aVals3(a)+aPreMinus*aVals3(a-1))+a2Exp*aVals1(a-1)-aExpSqr*aVals1(a-2);
        aVals2(aTensor.aSize()-1) = (0.5+k*aPreMinus)*aVals3(aTensor.aSize()-1);
        aVals2(aTensor.aSize()-2) = k*((aPrePlus-aExpSqr)*aVals3(aTensor.aSize()-1))+(a2Exp-aExpSqr)*aVals2(aTensor.aSize()-1);
        for (int a = aTensor.aSize()-3; a >= 0; a--)
          aVals2(a) = k*(aPrePlus*aVals3(a+1)-aExpSqr*aVals3(a+2))+a2Exp*aVals2(a+1)-aExpSqr*aVals2(a+2);
        for (int a = 0; a < aTensor.aSize(); a++)
          aTensor(x,y,z,a) = aVals1(a)+aVals2(a);
      }
}

// Nonlinear filters -----------------------------------------------------------

// osher (2D)
template <class T>
void osher(CMatrix<T>& aData, int aIterations) {
  CMatrix<T> aDiff(aData.xSize(),aData.ySize());
  for (int t = 0; t < aIterations; t++) {
    for (int y = 0; y < aData.ySize(); y++)
      for (int x = 0; x < aData.xSize(); x++) {
        T u00,u01,u02,u10,u11,u12,u20,u21,u22;
        if (x > 0) {
          if (y > 0) u00 = aData(x-1,y-1);
          else u00 = aData(x-1,0);
          u01 = aData(x-1,y);
          if (y < aData.ySize()-1) u02 = aData(x-1,y+1);
          else u02 = aData(x-1,y);
        }
        else {
          if (y > 0) u00 = aData(0,y-1);
          else u00 = aData(0,0);
          u01 = aData(0,y);
          if (y < aData.ySize()-1) u02 = aData(0,y+1);
          else u02 = aData(0,y);
        }
        if (y > 0) u10 = aData(x,y-1);
        else u10 = aData(x,y);
        u11 = aData(x,y);
        if (y < aData.ySize()-1) u12 = aData(x,y+1);
        else u12 = aData(x,y);
        if (x < aData.xSize()-1) {
          if (y > 0) u20 = aData(x+1,y-1);
          else u20 = aData(x+1,y);
          u21 = aData(x+1,y);
          if (y < aData.ySize()-1) u22 = aData(x+1,y+1);
          else u22 = aData(x+1,y);
        }
        else {
          if (y > 0) u20 = aData(x,y-1);
          else u20 = aData(x,y);
          u21 = aData(x,y);
          if (y < aData.ySize()-1) u22 = aData(x,y+1);
          else u22 = aData(x,y);
        }
        T ux = 0.5*(u21-u01);
        T uy = 0.5*(u12-u10);
        T uxuy = ux*uy;
        T uxx = u01-2.0*u11+u21;
        T uyy = u10-2.0*u11+u12;
        T uxy;
        if (uxuy < 0) uxy = 2.0*u11+u00+u22-u10-u12-u01-u21;
        else uxy = u10+u12+u01+u21-2.0*u11-u02-u20;
        T laPlace = uyy*uy*uy+uxy*uxuy+uxx*ux*ux;
        T uxLeft = u11-u01;
        T uxRight = u21-u11;
        T uyUp = u11-u10;
        T uyDown = u12-u11;
        if (laPlace < 0) {
          T aSum = 0;
          if (uxRight > 0) aSum += uxRight*uxRight;
          if (uxLeft < 0) aSum += uxLeft*uxLeft;
          if (uyDown > 0) aSum += uyDown*uyDown;
          if (uyUp < 0) aSum += uyUp*uyUp;
          aDiff(x,y) = sqrt(aSum);
        }
        else if (laPlace > 0) {
          T aSum = 0;
          if (uxRight < 0) aSum += uxRight*uxRight;
          if (uxLeft > 0) aSum += uxLeft*uxLeft;
          if (uyDown < 0) aSum += uyDown*uyDown;
          if (uyUp > 0) aSum += uyUp*uyUp;
          aDiff(x,y) = -sqrt(aSum);
        }
      }
    for (int i = 0; i < aData.size(); i++)
      aData.data()[i] += 0.25*aDiff.data()[i];
  }
}

template <class T>
inline void osher(const CMatrix<T>& aData, CMatrix<T>& aResult, int aIterations) {
  aResult = aData;
  osher(aResult,aIterations);
}

}

#endif

