// CMatrix
// A two-dimensional array including basic matrix operations
//
// Author: Thomas Brox
//-------------------------------------------------------------------------

#ifndef CMATRIX_H
#define CMATRIX_H

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <stack>
#ifdef GNU_COMPILER
  #include <strstream>
#else
  #include <sstream>
#endif
#include <CVector.h>

template <class T>
class CMatrix {
public:
  // standard constructor
  inline CMatrix();
  // constructor
  inline CMatrix(const int aXSize, const int aYSize);
  // copy constructor
  CMatrix(const CMatrix<T>& aCopyFrom);
  // constructor with implicit filling
  CMatrix(const int aXSize, const int aYSize, const T aFillValue);
  // destructor
  virtual ~CMatrix();

  // Changes the size of the matrix, data will be lost
  void setSize(int aXSize, int aYSize);
  // Downsamples the matrix
  void downsampleBool(int aNewXSize, int aNewYSize, float aThreshold = 0.5);
  void downsampleInt(int aNewXSize, int aNewYSize);
  void downsample(int aNewXSize, int aNewYSize);
  void downsample(int aNewXSize, int aNewYSize, CMatrix<float>& aConfidence);
  void downsampleBilinear(int aNewXSize, int aNewYSize);  
  // Upsamples the matrix
  void upsample(int aNewXSize, int aNewYSize);
  void upsampleBilinear(int aNewXSize, int aNewYSize);
//  void upsampleBicubic(int aNewXSize, int aNewYSize);
  // Scales the matrix (includes upsampling and downsampling)
  void rescale(int aNewXSize, int aNewYSize);
  // Creates an identity matrix
  void identity(int aSize);
  // Fills the matrix with the value aValue (see also operator =)
  void fill(const T aValue);
  // Fills a rectangular area with the value aValue
  void fillRect(const T aValue, int ax1, int ay1, int ax2, int ay2);
  // Copies a rectangular part from the matrix into aResult, the size of aResult will be adjusted
  void cut(CMatrix<T>& aResult,const int x1, const int y1, const int x2, const int y2);
  // Copies aCopyFrom at a certain position of the matrix
  void paste(CMatrix<T>& aCopyFrom, int ax, int ay);
  // Mirrors the boundaries, aFrom is the distance from the boundaries where the pixels are copied from,
  // aTo is the distance from the boundaries they are copied to
  void mirror(int aFrom, int aTo);
  // Transforms the values so that they are all between aMin and aMax
  // aInitialMin/Max are initializations for seeking the minimum and maximum, change if your
  // data is not in this range or the data type T cannot hold these values
  void normalize(T aMin, T aMax, T aInitialMin = -30000, T aInitialMax = 30000);
  // Clips values that exceed the given range
  void clip(T aMin, T aMax);

  // Applies a similarity transform (translation, rotation, scaling) to the image
  void applySimilarityTransform(CMatrix<T>& aWarped, CMatrix<bool>& aOutside, float tx, float ty, float cx, float cy, float phi, float scale);
  // Applies a homography (linear projective transformation) to the image
  void applyHomography(CMatrix<T>& aWarped, CMatrix<bool>& aOutside, const CMatrix<float>& H);

  // Draws a line into the image
  void drawLine(int dStartX, int dStartY, int dEndX, int dEndY, T aValue = 255);
  // Inverts a gray value image
  void invertImage();
  // Extracts the connected component starting from (x,y)
  // Component -> 255, Remaining area -> 0
  void connectedComponent(int x, int y);

  // Appends another matrix with the same column number
  void append(CMatrix<T>& aMatrix);
  // Inverts a square matrix with Gauss elimination
  void inv();
  // Transposes a square matrix
  void trans();
  // Multiplies with two vectors (from left and from right)
  float scalar(CVector<T>& aLeft, CVector<T>& aRight);
  
  // Reads a picture from a pgm-File
  void readFromPGM(const char* aFilename);
  // Saves the matrix as a picture in pgm-Format
  void writeToPGM(const char *aFilename);
  // Read matrix from text file
  void readFromTXT(const char* aFilename, bool aHeader = true, int aXSize = 0, int aYSize = 0);
  // Read matrix from Matlab ascii file
  void readFromMatlabTXT(const char* aFilename, bool aHeader = true, int aXSize = 0, int aYSize = 0);
  // Save matrix as text file
  void writeToTXT(const char* aFilename, bool aHeader = true);
  // Reads a projection matrix in a format used by Bodo Rosenhahn
  void readBodoProjectionMatrix(const char* aFilename);

  // Gives full access to matrix values
  inline T& operator()(const int ax, const int ay) const;
  // Fills the matrix with the value aValue (equivalent to fill())
  inline CMatrix<T>& operator=(const T aValue);
  // Copies the matrix aCopyFrom to this matrix (size of matrix might change)
  CMatrix<T>& operator=(const CMatrix<T>& aCopyFrom);
  // matrix sum
  CMatrix<T>& operator+=(const CMatrix<T>& aMatrix);
  // Adds a constant to the matrix
  CMatrix<T>& operator+=(const T aValue);
  // matrix difference
  CMatrix<T>& operator-=(const CMatrix<T>& aMatrix);
  // matrix product
  CMatrix<T>& operator*=(const CMatrix<T>& aMatrix);
  // Multiplication with a scalar
  CMatrix<T>& operator*=(const T aValue);

  // Comparison of two matrices
  bool operator==(const CMatrix<T>& aMatrix);

  // Returns the minimum value
  T min() const;
  // Returns the maximum value
  T max() const;
  // Returns the average value
  T avg() const;
  // Gives access to the matrix' size
  inline int xSize() const;
  inline int ySize() const;
  inline int size() const;
  // Returns one row from the matrix
  void getVector(CVector<T>& aVector, int ay);
  // Gives access to the internal data representation
  inline T* data() const;
protected:
  int mXSize,mYSize;
  T *mData;
};

// Returns a matrix where all negative elements are turned positive
template <class T> CMatrix<T> abs(const CMatrix<T>& aMatrix);
// Returns the tranposed matrix
template <class T> CMatrix<T> trans(const CMatrix<T>& aMatrix);
// matrix sum
template <class T> CMatrix<T> operator+(const CMatrix<T>& aM1, const CMatrix<T>& aM2);
// matrix difference
template <class T> CMatrix<T> operator-(const CMatrix<T>& aM1, const CMatrix<T>& aM2);
// matrix product
template <class T> CMatrix<T> operator*(const CMatrix<T>& aM1, const CMatrix<T>& aM2);
// Multiplication with a vector
template <class T> CVector<T> operator*(const CMatrix<T>& aMatrix, const CVector<T>& aVector);
// Multiplikation with a scalar
template <class T> CMatrix<T> operator*(const CMatrix<T>& aMatrix, const T aValue);
template <class T> inline CMatrix<T> operator*(const T aValue, const CMatrix<T>& aMatrix);
// Provides basic output functionality (only appropriate for small matrices)
template <class T> std::ostream& operator<<(std::ostream& aStream, const CMatrix<T>& aMatrix);

// Exceptions thrown by CMatrix-------------------------------------------


// Thrown when one tries to access an element of a matrix which is out of
// the matrix' bounds
struct EMatrixRangeOverflow {
  EMatrixRangeOverflow(const int ax, const int ay) {
    using namespace std;
    cerr << "Exception EMatrixRangeOverflow: x = " << ax << ", y = " << ay << endl;
  }
};

// Thrown when one tries to multiply two matrices where M1's column number
// is not equal to M2's row number or when one tries to add two matrices
// which have not the same size
struct EIncompatibleMatrices {
  EIncompatibleMatrices(const int x1, const int y1, const int x2, const int y2) {

    using namespace std;
    cerr << "Exception EIncompatibleMatrices: M1 = " << x1 << "x" << y1;
    cerr << "  M2 = " << x2 << "x" << y2 << endl;
  }
};

// Thrown when a nonquadratic matrix is tried to be inversed
struct ENonquadraticMatrix {
  ENonquadraticMatrix(const int x, const int y) {
    using namespace std;
    cerr << "Exception ENonquadarticMatrix: M = " << x << "x" << y << endl;
  }
};

// Thrown when a matrix is not positive definite
struct ENonPositiveDefinite {
  ENonPositiveDefinite() {
  using namespace std;
    cerr << "Exception ENonPositiveDefinite" << endl;
  }
};

// Thrown when reading a file which does not keep to the PGM specification
struct EInvalidFileFormat {
  EInvalidFileFormat(const char* s) {
    using namespace std;
    cerr << "Exception EInvalidFileFormat: File is not in " << s << " format" << endl;
  }
};

// I M P L E M E N T A T I O N --------------------------------------------
//
// You might wonder why there is implementation code in a header file.
// The reason is that not all C++ compilers yet manage separate compilation
// of templates. Inline functions cannot be compiled separately anyway.
// So in this case the whole implementation code is added to the header
// file.
// Users of CMatrix should ignore everything that's beyond this line :)
// ------------------------------------------------------------------------

// P U B L I C ------------------------------------------------------------

// standard constructor
template <class T>
inline CMatrix<T>::CMatrix() {
  mData = 0; mXSize = mYSize = 0;
}

// constructor
template <class T>
inline CMatrix<T>::CMatrix(const int aXSize, const int aYSize)
  : mXSize(aXSize), mYSize(aYSize) {
  mData = new T[aXSize*aYSize];
}

// copy constructor
template <class T>
CMatrix<T>::CMatrix(const CMatrix<T>& aCopyFrom)
  : mXSize(aCopyFrom.mXSize), mYSize(aCopyFrom.mYSize) {
  if (aCopyFrom.mData == 0) mData = 0;
  else {
    int wholeSize = mXSize*mYSize;
    mData = new T[wholeSize];
    for (register int i = 0; i < wholeSize; i++)
      mData[i] = aCopyFrom.mData[i];
  }
}

// constructor with implicit filling
template <class T>
CMatrix<T>::CMatrix(const int aXSize, const int aYSize, const T aFillValue)
  : mXSize(aXSize), mYSize(aYSize) {
  mData = new T[aXSize*aYSize];
  fill(aFillValue);
}

// destructor
template <class T>
CMatrix<T>::~CMatrix() {
  delete [] mData;
}

// setSize
template <class T>
void CMatrix<T>::setSize(int aXSize, int aYSize) {
  if (mData != 0) delete[] mData;
  mData = new T[aXSize*aYSize];
  mXSize = aXSize;
  mYSize = aYSize;
}

// downsampleBool
template <class T>
void CMatrix<T>::downsampleBool(int aNewXSize, int aNewYSize, float aThreshold) {
  CMatrix<float> aTemp(mXSize,mYSize);
  int aSize = size();
  for (int i = 0; i < aSize; i++)
    aTemp.data()[i] = mData[i];
  aTemp.downsample(aNewXSize,aNewYSize);
  setSize(aNewXSize,aNewYSize);
  aSize = size();
  for (int i = 0; i < aSize; i++)
    mData[i] = (aTemp.data()[i] >= aThreshold);
}

// downsampleInt
template <class T>
void CMatrix<T>::downsampleInt(int aNewXSize, int aNewYSize) {
  T* newData = new int[aNewXSize*aNewYSize];
  float factorX = ((float)mXSize)/aNewXSize;
  float factorY = ((float)mYSize)/aNewYSize;
  float ay = 0.0;
  for (int y = 0; y < aNewYSize; y++) {
    float ax = 0.0;
    for (int x = 0; x < aNewXSize; x++) {
      CVector<float> aHistogram(256,0.0);
      for (float by = 0.0; by < factorY;) {
        float restY = floor(by+1.0)-by;
        if (restY+by >= factorY) restY = factorY-by;
        for (float bx = 0.0; bx < factorX;) {
          float restX = floor(bx+1.0)-bx;
          if (restX+bx >= factorX) restX = factorX-bx;
          aHistogram(operator()((int)(ax+bx),(int)(ay+by))) += restX*restY;
          bx += restX;
        }
        by += restY;
      }
      float aMax = 0; int aMaxVal;
      for (int i = 0; i < aHistogram.size(); i++)
        if (aHistogram(i) > aMax) {
          aMax = aHistogram(i);
          aMaxVal = i;
        }
      newData[x+aNewXSize*y] = aMaxVal;
      ax += factorX;
    }
    ay += factorY;
  }
  delete[] mData;
  mData = newData;
  mXSize = aNewXSize; mYSize = aNewYSize;
}

template <class T>
void CMatrix<T>::downsample(int aNewXSize, int aNewYSize) {
  // Downsample in x-direction
  int aIntermedSize = aNewXSize*mYSize;
  T* aIntermedData = new T[aIntermedSize];
  if (aNewXSize < mXSize) {
    for (int i = 0; i < aIntermedSize; i++)
      aIntermedData[i] = 0.0;
    T factor = ((float)mXSize)/aNewXSize;
    for (int y = 0; y < mYSize; y++) {
      int aFineOffset = y*mXSize;
      int aCoarseOffset = y*aNewXSize;
      int i = aFineOffset;
      int j = aCoarseOffset;
      int aLastI = aFineOffset+mXSize;
      int aLastJ = aCoarseOffset+aNewXSize;
      T rest = factor;
      T part = 1.0;
      do {
        if (rest > 1.0) {
          aIntermedData[j] += part*mData[i];
          rest -= part;
          part = 1.0;
          i++;
          if (rest <= 0.0) {
            rest = factor;
            j++;
          }
        }
        else {
          aIntermedData[j] += rest*mData[i];
          part = 1.0-rest;
          rest = factor;
          j++;
        }
      }
      while (i < aLastI && j < aLastJ);
    }
  }
  else {
    T* aTemp = aIntermedData;
    aIntermedData = mData;
    mData = aTemp;
  }
  // Downsample in y-direction
  delete[] mData;
  int aDataSize = aNewXSize*aNewYSize;
  mData = new T[aDataSize];
  if (aNewYSize < mYSize) {
    for (int i = 0; i < aDataSize; i++)
      mData[i] = 0.0;
    float factor = ((float)mYSize)/aNewYSize;
    for (int x = 0; x < aNewXSize; x++) {
      int i = x;
      int j = x;
      int aLastI = mYSize*aNewXSize+x;
      int aLastJ = aNewYSize*aNewXSize+x;
      float rest = factor;
      float part = 1.0;
      do {
        if (rest > 1.0) {
          mData[j] += part*aIntermedData[i];
          rest -= part;
          part = 1.0;
          i += aNewXSize;
          if (rest <= 0.0) {
            rest = factor;
            j += aNewXSize;
          }
        }
        else {
          mData[j] += rest*aIntermedData[i];
          part = 1.0-rest;
          rest = factor;
          j += aNewXSize;
        }
      }
      while (i < aLastI && j < aLastJ);
    }
  }
  else {
    T* aTemp = mData;
    mData = aIntermedData;
    aIntermedData = aTemp;
  }
  // Normalize
  float aNormalization = ((float)aDataSize)/size();
  for (int i = 0; i < aDataSize; i++)
    mData[i] *= aNormalization;
  // Adapt size of matrix
  mXSize = aNewXSize;
  mYSize = aNewYSize;
  delete[] aIntermedData;
}

template <class T>
void CMatrix<T>::downsample(int aNewXSize, int aNewYSize, CMatrix<float>& aConfidence) {
  int aNewSize = aNewXSize*aNewYSize;
  T* newData = new T[aNewSize];
  float* aCounter = new float[aNewSize];
  for (int i = 0; i < aNewSize; i++) {
    newData[i] = 0;
    aCounter[i] = 0;
  }
  float factorX = ((float)aNewXSize)/mXSize;
  float factorY = ((float)aNewYSize)/mYSize;
  for (int y = 0; y < mYSize; y++)
    for (int x = 0; x < mXSize; x++)
      if (aConfidence(x,y) > 0) {
        float ax = x*factorX;
        float ay = y*factorY;
        int x1 = (int)ax;
        int y1 = (int)ay;
        int x2 = x1+1;
        int y2 = y1+1;
        float alphax = ax-x1;
        float betax = 1.0-alphax;
        float alphay = ay-y1;
        float betay = 1.0-alphay;
        float conf = aConfidence(x,y);
        T val = conf*operator()(x,y);
        int i = x1+aNewXSize*y1;
        newData[i] += betax*betay*val;
        aCounter[i] += betax*betay*conf;
        if (x2 < aNewXSize) {
          i = x2+aNewXSize*y1;
          newData[i] += alphax*betay*val;
          aCounter[i] += alphax*betay*conf;
        }
        if (y2 < aNewYSize) {
          i = x1+aNewXSize*y2;
          newData[i] += betax*alphay*val;
          aCounter[i] += betax*alphay*conf;
        }
        if (x2 < aNewXSize && y2 < aNewYSize) {
          i = x2+aNewXSize*y2;
          newData[i] += alphax*alphay*val;
          aCounter[i] += alphax*alphay*conf;
        }
      }
  for (int i = 0; i < aNewSize; i++)
    if (aCounter[i] > 0) newData[i] /= aCounter[i];
  // Adapt size of matrix
  mXSize = aNewXSize;
  mYSize = aNewYSize;
  delete[] mData;
  delete[] aCounter;
  mData = newData;
}

// downsampleBilinear
template <class T>
void CMatrix<T>::downsampleBilinear(int aNewXSize, int aNewYSize) {
  int aNewSize = aNewXSize*aNewYSize;
  T* aNewData = new T[aNewSize];
  float factorX = ((float)mXSize)/aNewXSize;
  float factorY = ((float)mYSize)/aNewYSize;
  for (int y = 0; y < aNewYSize; y++)
    for (int x = 0; x < aNewXSize; x++) {
      float ax = (x+0.5)*factorX-0.5;
      float ay = (y+0.5)*factorY-0.5;
      if (ax < 0) ax = 0.0;
      if (ay < 0) ay = 0.0;
      int x1 = (int)ax;
      int y1 = (int)ay;
      int x2 = x1+1;
      int y2 = y1+1;
      float alphaX = ax-x1;
      float alphaY = ay-y1;
      if (x1 < 0) x1 = 0;
      if (y1 < 0) y1 = 0;
      if (x2 >= mXSize) x2 = mXSize-1;
      if (y2 >= mYSize) y2 = mYSize-1;
      float a = (1.0-alphaX)*mData[x1+y1*mXSize]+alphaX*mData[x2+y1*mXSize];
      float b = (1.0-alphaX)*mData[x1+y2*mXSize]+alphaX*mData[x2+y2*mXSize];
      aNewData[x+y*aNewXSize] = (1.0-alphaY)*a+alphaY*b;
    }
  delete[] mData;
  mData = aNewData;
  mXSize = aNewXSize;
  mYSize = aNewYSize;
}

template <class T>
void CMatrix<T>::upsample(int aNewXSize, int aNewYSize) {
  // Upsample in x-direction
  int aIntermedSize = aNewXSize*mYSize;
  T* aIntermedData = new T[aIntermedSize];
  if (aNewXSize > mXSize) {
    for (int i = 0; i < aIntermedSize; i++)
      aIntermedData[i] = 0.0;
    T factor = ((float)aNewXSize)/mXSize;
    for (int y = 0; y < mYSize; y++) {
      int aFineOffset = y*aNewXSize;
      int aCoarseOffset = y*mXSize;
      int i = aCoarseOffset;
      int j = aFineOffset;
      int aLastI = aCoarseOffset+mXSize;
      int aLastJ = aFineOffset+aNewXSize;
      T rest = factor;
      T part = 1.0;
      do {
        if (rest > 1.0) {
          aIntermedData[j] += part*mData[i];
          rest -= part;
          part = 1.0;
          j++;
          if (rest <= 0.0) {
            rest = factor;
            i++;
          }
        }
        else {
          aIntermedData[j] += rest*mData[i];
          part = 1.0-rest;
          rest = factor;
          i++;
        }
      }
      while (i < aLastI && j < aLastJ);
    }
  }
  else {
    T* aTemp = aIntermedData;
    aIntermedData = mData;
    mData = aTemp;
  }
  // Upsample in y-direction
  delete[] mData;
  int aDataSize = aNewXSize*aNewYSize;
  mData = new T[aDataSize];
  if (aNewYSize > mYSize) {
    for (int i = 0; i < aDataSize; i++)
      mData[i] = 0.0;
    float factor = ((float)aNewYSize)/mYSize;
    for (int x = 0; x < aNewXSize; x++) {
      int i = x;
      int j = x;
      int aLastI = mYSize*aNewXSize;
      int aLastJ = aNewYSize*aNewXSize;
      float rest = factor;
      float part = 1.0;
      do {
        if (rest > 1.0) {
          mData[j] += part*aIntermedData[i];
          rest -= part;
          part = 1.0;
          j += aNewXSize;
          if (rest <= 0.0) {
            rest = factor;
            i += aNewXSize;
          }
        }
        else {
          mData[j] += rest*aIntermedData[i];
          part = 1.0-rest;
          rest = factor;
          i += aNewXSize;
        }
      }
      while (i < aLastI && j < aLastJ);
    }
  }
  else {
    T* aTemp = mData;
    mData = aIntermedData;
    aIntermedData = aTemp;
  }
  // Adapt size of matrix
  mXSize = aNewXSize;
  mYSize = aNewYSize;
  delete[] aIntermedData;
}

// upsampleBilinear
template <class T>
void CMatrix<T>::upsampleBilinear(int aNewXSize, int aNewYSize) {
  int aNewSize = aNewXSize*aNewYSize;
  T* aNewData = new T[aNewSize];
  float factorX = (float)(mXSize)/(aNewXSize);
  float factorY = (float)(mYSize)/(aNewYSize);
  for (int y = 0; y < aNewYSize; y++)
    for (int x = 0; x < aNewXSize; x++) {
      float ax = (x+0.5)*factorX-0.5;
      float ay = (y+0.5)*factorY-0.5;
      if (ax < 0) ax = 0.0;
      if (ay < 0) ay = 0.0;
      int x1 = (int)ax;
      int y1 = (int)ay;
      int x2 = x1+1;
      int y2 = y1+1;
      float alphaX = ax-x1;
      float alphaY = ay-y1;
      if (x1 < 0) x1 = 0;
      if (y1 < 0) y1 = 0;
      if (x2 >= mXSize) x2 = mXSize-1;
      if (y2 >= mYSize) y2 = mYSize-1;
      float a = (1.0-alphaX)*mData[x1+y1*mXSize]+alphaX*mData[x2+y1*mXSize];
      float b = (1.0-alphaX)*mData[x1+y2*mXSize]+alphaX*mData[x2+y2*mXSize];
      aNewData[x+y*aNewXSize] = (1.0-alphaY)*a+alphaY*b;
    }
  delete[] mData;
  mData = aNewData;
  mXSize = aNewXSize;
  mYSize = aNewYSize;
}

template <class T>
void CMatrix<T>::rescale(int aNewXSize, int aNewYSize) {
  if (mXSize >= aNewXSize) {
    if (mYSize >= aNewYSize) downsample(aNewXSize,aNewYSize);
    else {
      downsample(aNewXSize,mYSize);
      upsample(aNewXSize,aNewYSize);
    }
  }
  else {
    if (mYSize >= aNewYSize) {
      downsample(mXSize,aNewYSize);
      upsample(aNewXSize,aNewYSize);
    }
    else upsample(aNewXSize,aNewYSize);
  }
}

// identity
template <class T>
void CMatrix<T>::identity(int aSize) {
  if (aSize != mXSize || aSize != mYSize) {
    delete[] mData;
    mData = new T[aSize*aSize];
    mXSize = aSize;
    mYSize = aSize;
  }
  fill(0);
  for (int i = 0; i < aSize; i++)
    operator()(i,i) = 1;
}

// fill
template <class T>
void CMatrix<T>::fill(const T aValue) {
  int wholeSize = mXSize*mYSize;
  for (register int i = 0; i < wholeSize; i++)
    mData[i] = aValue;
}

// fillRect
template <class T>
void CMatrix<T>::fillRect(const T aValue, int ax1, int ay1, int ax2, int ay2) {
  for (int y = ay1; y <= ay2; y++)
    for (register int x = ax1; x <= ax2; x++)
      operator()(x,y) = aValue;
}

// cut
template <class T>
void CMatrix<T>::cut(CMatrix<T>& aResult,const int x1, const int y1, const int x2, const int y2) {
  aResult.mXSize = x2-x1+1;
  aResult.mYSize = y2-y1+1;
  delete[] aResult.mData;
  aResult.mData = new T[aResult.mXSize*aResult.mYSize];
  for (int y = y1; y <= y2; y++)
    for (int x = x1; x <= x2; x++)
      aResult(x-x1,y-y1) = operator()(x,y);
}

// paste
template <class T>
void CMatrix<T>::paste(CMatrix<T>& aCopyFrom, int ax, int ay) {
  for (int y = 0; y < aCopyFrom.ySize(); y++)
    for (int x = 0; x < aCopyFrom.xSize(); x++)
      operator()(ax+x,ay+y) = aCopyFrom(x,y);
}

// mirror
template <class T>
void CMatrix<T>::mirror(int aFrom, int aTo) {
  int aToXIndex = mXSize-aTo-1;
  int aToYIndex = mYSize-aTo-1;
  int aFromXIndex = mXSize-aFrom-1;
  int aFromYIndex = mYSize-aFrom-1;
  for (int y = aFrom; y <= aFromYIndex; y++) {
    operator()(aTo,y) = operator()(aFrom,y);
    operator()(aToXIndex,y) = operator()(aFromXIndex,y);
  }
  for (int x = aTo; x <= aToXIndex; x++) {
    operator()(x,aTo) = operator()(x,aFrom);
    operator()(x,aToYIndex) = operator()(x,aFromYIndex);
  }
}

// normalize
template <class T>
void CMatrix<T>::normalize(T aMin, T aMax, T aInitialMin, T aInitialMax) {
  int aSize = mXSize*mYSize;
  T aCurrentMin = aInitialMax;
  T aCurrentMax = aInitialMin;
  for (int i = 0; i < aSize; i++)
    if (mData[i] > aCurrentMax) aCurrentMax = mData[i];
    else if (mData[i] < aCurrentMin) aCurrentMin = mData[i];
  T aTemp = (aCurrentMax-aCurrentMin);
  if (aTemp == 0) aTemp = 1;
  else aTemp = (aMax-aMin)/aTemp;
  for (int i = 0; i < aSize; i++) {
    mData[i] -= aCurrentMin;
    mData[i] *= aTemp;
    mData[i] += aMin;
  }
}

// clip
template <class T>
void CMatrix<T>::clip(T aMin, T aMax) {
  int aSize = size();
  for (int i = 0; i < aSize; i++)
    if (mData[i] < aMin) mData[i] = aMin;
    else if (mData[i] > aMax) mData[i] = aMax;
}

// applySimilarityTransform
template <class T>
void CMatrix<T>::applySimilarityTransform(CMatrix<T>& aWarped, CMatrix<bool>& aOutside, float tx, float ty, float cx, float cy, float phi, float scale) {
  float cosphi = scale*cos(phi);
  float sinphi = scale*sin(phi);
  float ctx = cx+tx-cx*cosphi+cy*sinphi;
  float cty = cy+ty-cy*cosphi-cx*sinphi;
  aOutside = false;
  int i = 0;
  for (int y = 0; y < aWarped.ySize(); y++)
    for (int x = 0; x < aWarped.xSize(); x++,i++) {
      float xf = x; float yf = y;
      float ax = xf*cosphi-yf*sinphi+ctx;
      float ay = yf*cosphi+xf*sinphi+cty;
      int x1 = (int)ax; int y1 = (int)ay;
      float alphaX = ax-x1; float alphaY = ay-y1;
      float betaX = 1.0-alphaX; float betaY = 1.0-alphaY;
      if (x1 < 0 || y1 < 0 || x1+1 >= mXSize || y1+1 >= mYSize) aOutside.data()[i] = true;
      else {
        int j = y1*mXSize+x1;
        float a = betaX*mData[j]       +alphaX*mData[j+1];
        float b = betaX*mData[j+mXSize]+alphaX*mData[j+1+mXSize];
        aWarped.data()[i] = betaY*a+alphaY*b;
      }
    }
}

// applyHomography
template <class T>
void CMatrix<T>::applyHomography(CMatrix<T>& aWarped, CMatrix<bool>& aOutside, const CMatrix<float>& H) {
  int aSize = size();
  aOutside = false;
  int i = 0;
  for (int y = 0; y < aWarped.ySize(); y++)
    for (int x = 0; x < aWarped.xSize(); x++,i++) {
      float xf = x; float yf = y;
      float ax = H.data()[0]*xf+H.data()[1]*yf+H.data()[2];
      float ay = H.data()[3]*xf+H.data()[4]*yf+H.data()[5];
      float az = H.data()[6]*xf+H.data()[7]*yf+H.data()[8];
      float invaz = 1.0/az;
      ax *= invaz; ay *= invaz;
      int x1 = (int)ax; int y1 = (int)ay;
      float alphaX = ax-x1; float alphaY = ay-y1;
      float betaX = 1.0-alphaX; float betaY = 1.0-alphaY;
      if (x1 < 0 || y1 < 0 || x1+1 >= mXSize || y1+1 >= mYSize) aOutside.data()[i] = true;
      else {
        int j = y1*mXSize+x1;
        float a = betaX*mData[j]       +alphaX*mData[j+1];
        float b = betaX*mData[j+mXSize]+alphaX*mData[j+1+mXSize];
        aWarped.data()[i] = betaY*a+alphaY*b;
      }
    }
}

// drawLine
template <class T>
void CMatrix<T>::drawLine(int dStartX, int dStartY, int dEndX, int dEndY, T aValue) {
    // vertical line
    if (dStartX == dEndX) {
    if (dStartX < 0 || dStartX >= mXSize)   return;
        int x = dStartX;
        if (dStartY < dEndY) {
            for (int y = dStartY; y <= dEndY; y++)
                if (y >= 0 && y < mYSize) mData[x+y*mXSize] = aValue;
    }
        else {
            for (int y = dStartY; y >= dEndY; y--)
                if (y >= 0 && y < mYSize) mData[x+y*mXSize] = aValue;
    }
    return;
  }
    // horizontal line
    if (dStartY == dEndY) {
    if (dStartY < 0 || dStartY >= mYSize) return;
        int y = dStartY;
        if (dStartX < dEndX) {
            for (int x = dStartX; x <= dEndX; x++)
                if (x >= 0 && x < mXSize) mData[x+y*mXSize] = aValue;
    }
        else {
            for (int x = dStartX; x >= dEndX; x--)
                if (x >= 0 && x < mXSize) mData[x+y*mXSize] = aValue;
    }
    return;
  }
  float m = float(dStartY - dEndY) / float(dStartX - dEndX);
  float invm = 1.0/m;
  if (fabs(m) > 1.0) {
    if (dEndY > dStartY) {
      for (int y = dStartY; y <= dEndY; y++) {
        int x = (int)(0.5+dStartX+(y-dStartY)*invm);
        if (x >= 0 && x < mXSize && y >= 0 && y < mYSize)
          mData[x+y*mXSize] = aValue;
      }
    }
    else {
      for (int y = dStartY; y >= dEndY; y--) {
        int x = (int)(0.5+dStartX+(y-dStartY)*invm);
        if (x >= 0 && x < mXSize && y >= 0 && y < mYSize)
          mData[x+y*mXSize] = aValue;
      }
    }
  }
  else {
    if (dEndX > dStartX) {
      for (int x = dStartX; x <= dEndX; x++) {
        int y = (int)(0.5+dStartY+(x-dStartX)*m);
        if (x >= 0 && x < mXSize && y >= 0 && y < mYSize)
          mData[x+y*mXSize] = aValue;
      }
    }
    else {
      for (int x = dStartX; x >= dEndX; x--) {
        int y = (int)(0.5+dStartY+(x-dStartX)*m);
        if (x >= 0 && x < mXSize && y >= 0 && y < mYSize)
          mData[x+y*mXSize] = aValue;
      }
    }
  }
}

// invertImage
template <class T>
void CMatrix<T>::invertImage() {
  int aSize = mXSize*mYSize;
  for (int i = 0; i < aSize; i++)
    mData[i] = 255-mData[i];
}

// connectedComponent
typedef struct {short y, xl, xr, dy;} CSegment;

template <class T>
void CMatrix<T>::connectedComponent (int x, int y) {
  std::stack<CSegment> aStack;
  #define PUSH(Y,XL,XR,DY) if (Y+(DY)>=0 && Y+(DY)<mYSize)\
   {CSegment S; S.y = Y; S.xl = XL; S.xr = XR;S.dy = DY;aStack.push(S);}
  #define POP(Y,XL,XR,DY) {CSegment& S = aStack.top(); Y = S.y+(DY = S.dy);XL = S.xl; XR = S.xr; aStack.pop();}
  T aCompValue = operator()(x,y);
  CMatrix<bool> aConnected(mXSize,mYSize,false);
  int l,x1,x2,dy;
  PUSH(y,x,x,1);
  PUSH(y+1,x,x,-1);
  while (!aStack.empty()) {
  	POP(y,x1,x2,dy);
  	for (x=x1; x >= 0 && operator()(x,y) == aCompValue && !aConnected(x,y);x--)
	    aConnected(x,y) = true;
  	if (x >= x1) goto skip2;
	  l = x+1;
	  if (l < x1) PUSH(y,l,x1-1,-dy);
	  x = x1+1;
	  do {
	    for (; x < mXSize && operator()(x,y) == aCompValue && !aConnected(x,y); x++)
    		aConnected(x,y) = true;
	    PUSH(y,l,x-1,dy);
	    if (x>x2+1) PUSH(y,x2+1,x-1,-dy);
      skip2: for (x++;x <= x2 && (operator()(x,y) != aCompValue || aConnected(x,y)); x++);
	    l = x;
	  }
    while (x <= x2);
  }
  int aSize = size();
  for (int i = 0; i < aSize; i++)
	  if (aConnected.data()[i]) mData[i] = 255;
	  else mData[i] = 0;
  #undef PUSH
  #undef POP
}

// append
template <class T>
void CMatrix<T>::append(CMatrix<T>& aMatrix) {
  #ifdef _DEBUG
  if (aMatrix.xSize() != mXSize) throw EIncompatibleMatrices(mXSize,mYSize,aMatrix.xSize(),aMatrix.ySize());
  #endif
  T* aNew = new T[mXSize*(mYSize+aMatrix.ySize())];
  int aSize = mXSize*mYSize;
  for (int i = 0; i < aSize; i++)
    aNew[i] = mData[i];
  int aSize2 = mXSize*aMatrix.ySize();
  for (int i = 0; i < aSize2; i++)
    aNew[i+aSize] = aMatrix.data()[i];
  delete[] mData;
  mData = aNew;
  mYSize += aMatrix.ySize();
}

// inv
template <class T>
void CMatrix<T>::inv() {
  if (mXSize != mYSize) throw ENonquadraticMatrix(mXSize,mYSize);
  int* p = new int[mXSize];
  T* hv = new T[mXSize];
    CMatrix<T>& I(*this);
    int n = mYSize;
    for (int j = 0; j < n; j++)
      p[j] = j;
  for (int j = 0; j < n; j++) {
    T max = fabs(I(j,j));
    int r = j;
    for (int i = j+1; i < n; i++)
      if (fabs(I(j,i)) > max) {
        max = fabs(I(j,i));
        r = i;
      }
    // Matrix singular
    if (max <= 0) return;
    // Swap row j and r
    if (r > j) {
      for (int k = 0; k < n; k++) {
        T hr = I(k,j);
        I(k,j) = I(k,r);
        I(k,r) = hr;
      }
      int hi = p[j];
      p[j] = p[r];
      p[r] = hi;
    }
    T hr = 1/I(j,j);
    for (int i = 0; i < n; i++)
      I(j,i) *= hr;
    I(j,j) = hr;
    hr *= -1;
    for (int k = 0; k < n; k++)
      if (k != j) {
        for (int i = 0; i < n; i++)
          if (i != j) I(k,i) -= I(j,i)*I(k,j);
        I(k,j) *= hr;
      }
  }
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++)
      hv[p[k]] = I(k,i);
    for (int k = 0; k < n; k++)
      I(k,i) = hv[k];
  }
  delete[] p;
  delete[] hv;
}

template <class T>
void CMatrix<T>::trans() {
  for (int y = 0; y < mYSize; y++)
    for (int x = y; x < mXSize; x++) {
      float temp = operator()(x,y);
      operator()(x,y) = operator()(y,x);
      operator()(y,x) = temp;
    }
}

template <class T>
float CMatrix<T>::scalar(CVector<T>& aLeft, CVector<T>& aRight) {
  #ifdef _DEBUG
  if ((aLeft.size() != mYSize) || (aRight.size() != mXSize))
    throw EIncompatibleMatrices(mXSize,mYSize,aRight.size(),aLeft.size());
  #endif
  T* vec = new T[mYSize];
  T* dat = mData;
  for (int y = 0; y < mYSize; y++) {
    vec[y] = 0;
    for (int x = 0; x < mXSize; x++)
      vec[y] += *(dat++)*aRight(x);
  }
  T aResult = 0.0;
  for (int y = 0; y < mYSize; y++)
    aResult += vec[y]*aLeft(y);
  delete[] vec;
  return aResult;
}

// readFromPGM
template <class T>
void CMatrix<T>::readFromPGM(const char* aFilename) {
  FILE *aStream;
  aStream = fopen(aFilename,"rb");
  if (aStream == 0) std::cerr << "File not found: " << aFilename << std::endl;
  int dummy;
  // Find beginning of file (P5)
  while (getc(aStream) != 'P');
  if (getc(aStream) != '5') throw EInvalidFileFormat("PGM");
  do dummy = getc(aStream); while (dummy != '\n' && dummy != ' ');
  // Remove comments and empty lines
  dummy = getc(aStream);
  while (dummy == '#') {
    while (getc(aStream) != '\n');
    dummy = getc(aStream);
  }
  while (dummy == '\n')
    dummy = getc(aStream);
  // Read image size
  mXSize = dummy-48;
  while ((dummy = getc(aStream)) >= 48 && dummy < 58)
    mXSize = 10*mXSize+dummy-48;
  while ((dummy = getc(aStream)) < 48 || dummy >= 58);
  mYSize = dummy-48;
  while ((dummy = getc(aStream)) >= 48 && dummy < 58)
    mYSize = 10*mYSize+dummy-48;
  while (dummy != '\n' && dummy != ' ')
    dummy = getc(aStream);
  while ((dummy = getc(aStream)) >= 48 && dummy < 58);
  if (dummy != '\n') while (getc(aStream) != '\n');
  // Adjust size of data structure
  delete[] mData;
  mData = new T[mXSize*mYSize];
  // Read image data
  for (int i = 0; i < mXSize*mYSize; i++)
    mData[i] = getc(aStream);
  fclose(aStream);
}

// writeToPGM
template <class T>
void CMatrix<T>::writeToPGM(const char *aFilename) {
  FILE *aStream;
  aStream = fopen(aFilename,"wb");
  // write header
  char line[60];
  sprintf(line,"P5\n%d %d\n255\n",mXSize,mYSize);
  fwrite(line,strlen(line),1,aStream);
  // write data
  for (int i = 0; i < mXSize*mYSize; i++) {
    char dummy = (char)mData[i];
    fwrite(&dummy,1,1,aStream);
  }
  fclose(aStream);
}

// readFromTXT
template <class T>
void CMatrix<T>::readFromTXT(const char* aFilename, bool aHeader, int aXSize, int aYSize) {
  std::ifstream aStream(aFilename);
  // read header
  if (aHeader) aStream >> mXSize >> mYSize;
  else {
    mXSize = aXSize; 
    mYSize = aYSize;
  }
  // Adjust size of data structure
  delete[] mData;
  mData = new T[mXSize*mYSize];
  // read data
  for (int i = 0; i < mXSize*mYSize; i++)
    aStream >> mData[i];
}

// readFromMatlabTXT
template <class T>
void CMatrix<T>::readFromMatlabTXT(const char* aFilename, bool aHeader, int aXSize, int aYSize) {
  std::ifstream aStream(aFilename);
  // read header
  float nx,ny;
  if (aHeader) {
    aStream >> nx >> ny;
    mXSize = (int)nx; mYSize = (int)ny;
  }
  else {
    mXSize = aXSize; mYSize = aYSize;
  }
  // Adjust size of data structure
  delete[] mData;
  mData = new T[mXSize*mYSize];
  // read data
  for (int i = 0; i < mXSize*mYSize; i++)
    aStream >> mData[i];
}

//writeToTXT
template <class T>
void CMatrix<T>::writeToTXT(const char* aFilename, bool aHeader) {
  std::ofstream aStream(aFilename);
  // write header
  if (aHeader) aStream << mXSize << " " << mYSize << std::endl;
  // write data
  int i = 0;
  for (int y = 0; y < mYSize; y++) {
    for (int x = 0; x < mXSize; x++, i++)
      aStream << mData[i] << " ";
    aStream << std::endl;
  }
}

// readBodoProjectionMatrix
template <class T>
void CMatrix<T>::readBodoProjectionMatrix(const char* aFilename) {
  readFromTXT(aFilename,false,4,3);
}

// operator ()
template <class T>
inline T& CMatrix<T>::operator()(const int ax, const int ay) const {
  #ifdef _DEBUG
    if (ax >= mXSize || ay >= mYSize || ax < 0 || ay < 0)
      throw EMatrixRangeOverflow(ax,ay);
  #endif
  return mData[mXSize*ay+ax];
}

// operator =
template <class T>
inline CMatrix<T>& CMatrix<T>::operator=(const T aValue) {
  fill(aValue);
  return *this;
}

template <class T>
CMatrix<T>& CMatrix<T>::operator=(const CMatrix<T>& aCopyFrom) {
  if (this != &aCopyFrom) {
    if (mData != 0) delete[] mData;
    mXSize = aCopyFrom.mXSize;
    mYSize = aCopyFrom.mYSize;
    if (aCopyFrom.mData == 0) mData = 0;
    else {
      int wholeSize = mXSize*mYSize;
      mData = new T[wholeSize];
      for (register int i = 0; i < wholeSize; i++)
        mData[i] = aCopyFrom.mData[i];
    }
  }
  return *this;
}

// operator +=
template <class T>
CMatrix<T>& CMatrix<T>::operator+=(const CMatrix<T>& aMatrix) {
  if ((mXSize != aMatrix.mXSize) || (mYSize != aMatrix.mYSize))
    throw EIncompatibleMatrices(mXSize,mYSize,aMatrix.mXSize,aMatrix.mYSize);
  int wholeSize = mXSize*mYSize;
  for (int i = 0; i < wholeSize; i++)
    mData[i] += aMatrix.mData[i];
  return *this;
}

template <class T>
CMatrix<T>& CMatrix<T>::operator+=(const T aValue) {
  int wholeSize = mXSize*mYSize;
  for (int i = 0; i < wholeSize; i++)
    mData[i] += aValue;
  return *this;
}

// operator -=
template <class T>
CMatrix<T>& CMatrix<T>::operator-=(const CMatrix<T>& aMatrix) {
  if ((mXSize != aMatrix.mXSize) || (mYSize != aMatrix.mYSize))
    throw EIncompatibleMatrices(mXSize,mYSize,aMatrix.mXSize,aMatrix.mYSize);
  int wholeSize = mXSize*mYSize;
  for (int i = 0; i < wholeSize; i++)
    mData[i] -= aMatrix.mData[i];
  return *this;
}

// operator *=
template <class T>
CMatrix<T>& CMatrix<T>::operator*=(const CMatrix<T>& aMatrix) {
  if (mXSize != aMatrix.mYSize)
    throw EIncompatibleMatrices(mXSize,mYSize,aMatrix.mXSize,aMatrix.mYSize);
  T* oldData = mData;
  mData = new T[mYSize*aMatrix.mXSize];
  for (int y = 0; y < mYSize; y++)
    for (int x = 0; x < aMatrix.mXSize; x++) {
      mData[aMatrix.mXSize*y+x] = 0;
      for (int i = 0; i < mXSize; i++)
        mData[aMatrix.mXSize*y+x] += oldData[mXSize*y+i]*aMatrix(x,i);
    }
  delete[] oldData;
  mXSize = aMatrix.mXSize;
  return *this;
}

template <class T>
CMatrix<T>& CMatrix<T>::operator*=(const T aValue) {
  int wholeSize = mXSize*mYSize;
  for (int i = 0; i < wholeSize; i++)
    mData[i] *= aValue;
  return *this;
}

// min
template <class T>
T CMatrix<T>::min() const {
  T aMin = mData[0];
  int aSize = mXSize*mYSize;
  for (int i = 1; i < aSize; i++)
    if (mData[i] < aMin) aMin = mData[i];
  return aMin;
}

// max
template <class T>
T CMatrix<T>::max() const {
  T aMax = mData[0];
  int aSize = mXSize*mYSize;
  for (int i = 1; i < aSize; i++)
    if (mData[i] > aMax) aMax = mData[i];
  return aMax;
}

// avg
template <class T>
T CMatrix<T>::avg() const {
  T aAvg = 0;
  int aSize = mXSize*mYSize;
  for (int i = 0; i < aSize; i++)
    aAvg += mData[i];
  return aAvg/aSize;
}

// xSize
template <class T>
inline int CMatrix<T>::xSize() const {
  return mXSize;
}

// ySize
template <class T>
inline int CMatrix<T>::ySize() const {
  return mYSize;
}

// size
template <class T>
inline int CMatrix<T>::size() const {
  return mXSize*mYSize;
}

// getVector
template <class T>
void CMatrix<T>::getVector(CVector<T>& aVector, int ay) {
  int aOffset = mXSize*ay; 
  for (int x = 0; x < mXSize; x++)
    aVector(x) = mData[x+aOffset];
}

// data()
template <class T>
inline T* CMatrix<T>::data() const {
  return mData;
}

// N O N - M E M B E R  F U N C T I O N S --------------------------------------

// abs
template <class T>
CMatrix<T> abs(const CMatrix<T>& aMatrix) {
  CMatrix<T> result(aMatrix.xSize(),aMatrix.ySize());
  int wholeSize = aMatrix.size();
  for (register int i = 0; i < wholeSize; i++) {
    if (aMatrix.data()[i] < 0) result.data()[i] = -aMatrix.data()[i];
    else result.data()[i] = aMatrix.data()[i];
  }
  return result;
}

// trans
template <class T>
CMatrix<T> trans(const CMatrix<T>& aMatrix) {
  CMatrix<T> result(aMatrix.ySize(),aMatrix.xSize());
  for (int y = 0; y < aMatrix.ySize(); y++)
    for (int x = 0; x < aMatrix.xSize(); x++)
      result(y,x) = aMatrix(x,y);
  return result;
}

// operator +
template <class T>
CMatrix<T> operator+(const CMatrix<T>& aM1, const CMatrix<T>& aM2) {
  if ((aM1.xSize() != aM2.xSize()) || (aM1.ySize() != aM2.ySize()))
    throw EIncompatibleMatrices(aM1.xSize(),aM1.ySize(),aM2.xSize(),aM2.ySize());
  CMatrix<T> result(aM1.xSize(),aM1.ySize());
  int wholeSize = aM1.xSize()*aM1.ySize();
  for (int i = 0; i < wholeSize; i++)
    result.data()[i] = aM1.data()[i] + aM2.data()[i];
  return result;
}

// operator -
template <class T>
CMatrix<T> operator-(const CMatrix<T>& aM1, const CMatrix<T>& aM2) {
  if ((aM1.xSize() != aM2.xSize()) || (aM1.ySize() != aM2.ySize()))
    throw EIncompatibleMatrices(aM1.xSize(),aM1.ySize(),aM2.xSize(),aM2.ySize());
  CMatrix<T> result(aM1.xSize(),aM1.ySize());
  int wholeSize = aM1.xSize()*aM1.ySize();
  for (int i = 0; i < wholeSize; i++)
    result.data()[i] = aM1.data()[i] - aM2.data()[i];
  return result;
}

// operator *
template <class T>
CMatrix<T> operator*(const CMatrix<T>& aM1, const CMatrix<T>& aM2) {
  if (aM1.xSize() != aM2.ySize())
    throw EIncompatibleMatrices(aM1.xSize(),aM1.ySize(),aM2.xSize(),aM2.ySize());
  CMatrix<T> result(aM2.xSize(),aM1.ySize(),0);
  for (int y = 0; y < result.ySize(); y++)
    for (int x = 0; x < result.xSize(); x++)
      for (int i = 0; i < aM1.xSize(); i++)
        result(x,y) += aM1(i,y)*aM2(x,i);
  return result;
}

template <class T>
CVector<T> operator*(const CMatrix<T>& aMatrix, const CVector<T>& aVector) {
  if (aMatrix.xSize() != aVector.size())
    throw EIncompatibleMatrices(aMatrix.xSize(),aMatrix.ySize(),1,aVector.size());
  CVector<T> result(aMatrix.ySize(),0);
  for (int y = 0; y < aMatrix.ySize(); y++)
    for (int x = 0; x < aMatrix.xSize(); x++)
      result(y) += aMatrix(x,y)*aVector(x);
  return result;
}

template <class T>
CMatrix<T> operator*(const CMatrix<T>& aMatrix, const T aValue) {
  CMatrix<T> result(aMatrix.xSize(),aMatrix.ySize());
  int wholeSize = aMatrix.xSize()*aMatrix.ySize();
  for (int i = 0; i < wholeSize; i++)
    result.data()[i] = aMatrix.data()[i]*aValue;
  return result;
}

template <class T>
inline CMatrix<T> operator*(const T aValue, const CMatrix<T>& aMatrix) {
  return aMatrix*aValue;
}

// operator <<
template <class T>
std::ostream& operator<<(std::ostream& aStream, const CMatrix<T>& aMatrix) {
  for (int y = 0; y < aMatrix.ySize(); y++) {
    for (int x = 0; x < aMatrix.xSize(); x++)
      aStream << aMatrix(x,y) << ' ';
    aStream << std::endl;
  }
  return aStream;
}


// Comparison of two matrices
template <class T>  bool CMatrix<T>::operator==(const CMatrix<T>& aMatrix)
{
  if((*this).size()!=aMatrix.size())
    return false;

  for(int i=0; i<aMatrix.size();i++)
    if(mData[i] != aMatrix.mData[i])
      return false;
  return true;
}

#endif
