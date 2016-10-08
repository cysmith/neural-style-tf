// CVector
// A one-dimensional array including basic vector operations
//
// Author: Thomas Brox
// Last change: 23.05.2005
//-------------------------------------------------------------------------
#ifndef CVECTOR_H
#define CVECTOR_H

#include <iostream>
#include <fstream>

template <class T> class CMatrix;
template <class T> class CTensor;

template <class T>
class CVector {
public:
  // constructor
  inline CVector();
  // constructor
  inline CVector(const int aSize);
  // copy constructor
  CVector(const CVector<T>& aCopyFrom);
  // constructor (from array)
  CVector(const T* aPointer, const int aSize);
  // constructor with implicit filling
  CVector(const int aSize, const T aFillValue);
  // destructor
  virtual ~CVector();

  // Changes the size of the vector (data is lost)
  void setSize(int aSize);
  // Fills the vector with the specified value (see also operator=)
  void fill(const T aValue);
  // Appends the values of another vector
  void append(CVector<T>& aVector);
  // Normalizes the length of the vector to 1
  void normalize();
  // Normalizes the component sum to 1
  void normalizeSum();
  // Reads values from a text file
  void readFromTXT(const char* aFilename);
  // Writes values to a text file
  void writeToTXT(char* aFilename);
  // Returns the sum of all values
  T sum();
  // Returns the minimum value
  T min();
  // Returns the maximum value
  T max();
  // Returns the Euclidean norm
  T norm();

  // Converts vector to homogeneous coordinates, i.e., all components are divided by last component
  CVector<T>& homogen();
  // Remove the last component
  inline void homogen_nD();
  // Computes the cross product between this vector and aVector
  void cross(CVector<T>& aVector);

  // Gives full access to the vector's values
  inline T& operator()(const int aIndex) const;
  inline T& operator[](const int aIndex) const;
  // Fills the vector with the specified value (equivalent to fill)
  inline CVector<T>& operator=(const T aValue);
  // Copies a vector into this vector (size might change)
  CVector<T>& operator=(const CVector<T>& aCopyFrom);
  // Copies values from a matrix to the vector (size might change)
  CVector<T>& operator=(const CMatrix<T>& aCopyFrom);
  // Copies values from a tensor to the vector (size might change)
  CVector<T>& operator=(const CTensor<T>& aCopyFrom);
  // Adds another vector
  CVector<T>& operator+=(const CVector<T>& aVector);
  // Substracts another vector
  CVector<T>& operator-=(const CVector<T>& aVector);
  // Multiplies the vector with a scalar
  CVector<T>& operator*=(const T aValue);
  // Scalar product
  T operator*=(const CVector<T>& aVector);
  // Checks (non-)equivalence to another vector
  bool operator==(const CVector<T>& aVector);
  inline bool operator!=(const CVector<T>& aVector);

  // Gives access to the vector's size
  inline int size() const;
  // Gives access to the internal data representation
  inline T* data() const {return mData;}
protected:
  int mSize;
  T* mData;
};

// Adds two vectors
template <class T> CVector<T> operator+(const CVector<T>& vec1, const CVector<T>& vec2);
// Substracts two vectors
template <class T> CVector<T> operator-(const CVector<T>& vec1, const CVector<T>& vec2);
// Multiplies vector with a scalar
template <class T> CVector<T> operator*(const CVector<T>& aVector, const T aValue);
template <class T> CVector<T> operator*(const T aValue, const CVector<T>& aVector);
// Computes the scalar product of two vectors
template <class T> T operator*(const CVector<T>& vec1, const CVector<T>& vec2);
// Computes cross product of two vectors
template <class T> CVector<T> operator/(const CVector<T>& vec1, const CVector<T>& vec2);
// Sends the vector to an output stream
template <class T> std::ostream& operator<<(std::ostream& aStream, const CVector<T>& aVector);

// Exceptions thrown by CVector--------------------------------------------

// Thrown if one tries to access an element of a vector which is out of
// the vector's bounds
struct EVectorRangeOverflow {
  EVectorRangeOverflow(const int aIndex) {
    using namespace std;
    cerr << "Exception EVectorRangeOverflow: Index = " << aIndex << endl;
  }
};

struct EVectorIncompatibleSize {
  EVectorIncompatibleSize(int aSize1, int aSize2) {
    using namespace std;
    cerr << "Exception EVectorIncompatibleSize: " << aSize1 << " <> " << aSize2 << endl;
  }
};


// I M P L E M E N T A T I O N --------------------------------------------
//
// You might wonder why there is implementation code in a header file.
// The reason is that not all C++ compilers yet manage separate compilation
// of templates. Inline functions cannot be compiled separately anyway.
// So in this case the whole implementation code is added to the header
// file.
// Users of CVector should ignore everything that's beyond this line.
// ------------------------------------------------------------------------

// P U B L I C ------------------------------------------------------------
// constructor
template <class T>
inline CVector<T>::CVector() : mSize(0) {
  mData = new T[0];
}

// constructor
template <class T>
inline CVector<T>::CVector(const int aSize)
  : mSize(aSize) {
  mData = new T[aSize];
}

// copy constructor
template <class T>
CVector<T>::CVector(const CVector<T>& aCopyFrom)
  : mSize(aCopyFrom.mSize) {
  mData = new T[mSize];
  for (int i = 0; i < mSize; i++)
    mData[i] = aCopyFrom.mData[i];
}

// constructor (from array)
template <class T>
CVector<T>::CVector(const T* aPointer, const int aSize)
  : mSize(aSize) {
  mData = new T[mSize];
  for (int i = 0; i < mSize; i++)
    mData[i] = aPointer[i];
}

// constructor with implicit filling
template <class T>
CVector<T>::CVector(const int aSize, const T aFillValue)
  : mSize(aSize) {
  mData = new T[aSize];
  fill(aFillValue);
}

// destructor
template <class T>
CVector<T>::~CVector() {
  delete[] mData;
}

// setSize
template <class T>
void CVector<T>::setSize(int aSize) {
  if (mData != 0) delete[] mData;
  mData = new T[aSize];
  mSize = aSize;
}

// fill
template <class T>
void CVector<T>::fill(const T aValue) {
  for (register int i = 0; i < mSize; i++)
    mData[i] = aValue;
}

// append
template <class T>
void CVector<T>::append(CVector<T>& aVector) {
  T* aNewData = new T[mSize+aVector.size()];
  for (int i = 0; i < mSize; i++)
    aNewData[i] = mData[i];
  for (int i = 0; i < aVector.size(); i++)
    aNewData[i+mSize] = aVector(i);
  mSize += aVector.size();
  delete[] mData;
  mData = aNewData;
}

// normalize
template <class T>
void CVector<T>::normalize() {
  T aSum = 0;
  for (register int i = 0; i < mSize; i++)
    aSum += mData[i]*mData[i];
  if (aSum == 0) return;
  aSum = 1.0/sqrt(aSum);
  for (register int i = 0; i < mSize; i++)
    mData[i] *= aSum;
}

// normalizeSum
template <class T>
void CVector<T>::normalizeSum() {
  T aSum = 0;
  for (register int i = 0; i < mSize; i++)
    aSum += mData[i];
  if (aSum == 0) return;
  aSum = 1.0/aSum;
  for (register int i = 0; i < mSize; i++)
    mData[i] *= aSum;
}

// readFromTXT
template<class T>
void CVector<T>::readFromTXT(const char* aFilename) {
  std::ifstream aStream(aFilename);
  mSize = 0;
  float aDummy;
  while (!aStream.eof()) {
    aStream >> aDummy;
    mSize++;
  }
  aStream.close();
  std::ifstream aStream2(aFilename);
  delete mData;
  mData = new T[mSize];
  for (int i = 0; i < mSize; i++)
    aStream2 >> mData[i];
}

// writeToTXT
template<class T>
void CVector<T>::writeToTXT(char* aFilename) {
  std::ofstream aStream(aFilename);
  for (int i = 0; i < mSize; i++)
    aStream << mData[i] << std::endl;
}

// sum
template <class T>
T CVector<T>::sum() {
  T val = mData[0];
  for (int i = 1; i < mSize; i++)
    val += mData[i];
  return val;
}

// min
template <class T>
T CVector<T>::min() {
  T bestValue = mData[0];
  for (int i = 1; i < mSize; i++)
    if (mData[i] < bestValue) bestValue = mData[i];
  return bestValue;
}

// max
template <class T>
T CVector<T>::max() {
  T bestValue = mData[0];
  for (int i = 1; i < mSize; i++)
    if (mData[i] > bestValue) bestValue = mData[i];
  return bestValue;
}

// norm
template <class T>
T CVector<T>::norm() {
  T aSum = 0.0;
  for (int i = 0; i < mSize; i++)
    aSum += mData[i]*mData[i];
  return sqrt(aSum);
}

// homogen
template <class T>
CVector<T>& CVector<T>::homogen() {
  if (mSize > 1 && mData[mSize-1] != 0) {
    T invVal = 1.0/mData[mSize-1];
  	for (int i = 0; i < mSize; i++)
      mData[i] *= invVal;
  }
  return (*this);
}

// homogen_nD
template <class T>
inline void CVector<T>::homogen_nD() {
  mSize--;
}

// cross
template <class T>
void CVector<T>::cross(CVector<T>& aVector) {
  T aHelp0 = aVector(2)*mData[1] - aVector(1)*mData[2];
  T aHelp1 = aVector(0)*mData[2] - aVector(2)*mData[0];
  T aHelp2 = aVector(1)*mData[0] - aVector(0)*mData[1];
  mData[0] = aHelp0;
  mData[1] = aHelp1;
  mData[2] = aHelp2;
}

// operator()
template <class T>
inline T& CVector<T>::operator()(const int aIndex) const {
  #ifdef _DEBUG
    if (aIndex >= mSize || aIndex < 0)
      throw EVectorRangeOverflow(aIndex);
  #endif
  return mData[aIndex];
}

// operator[]
template <class T>
inline T& CVector<T>::operator[](const int aIndex) const {
  return operator()(aIndex);
}

// operator=
template <class T>
inline CVector<T>& CVector<T>::operator=(const T aValue) {
  fill(aValue);
  return *this;
}

template <class T>
CVector<T>& CVector<T>::operator=(const CVector<T>& aCopyFrom) {
  if (this != &aCopyFrom) {
    if (mSize != aCopyFrom.size()) {
      delete[] mData;
      mSize = aCopyFrom.size();
      mData = new T[mSize];
    }
    for (register int i = 0; i < mSize; i++)
      mData[i] = aCopyFrom.mData[i];
  }
  return *this;
}

template <class T>
CVector<T>& CVector<T>::operator=(const CMatrix<T>& aCopyFrom) {
  if (mSize != aCopyFrom.size()) {
    delete[] mData;
    mSize = aCopyFrom.size();
    mData = new T[mSize];
  }
  for (register int i = 0; i < mSize; i++)
    mData[i] = aCopyFrom.data()[i];
  return *this;
}

template <class T>
CVector<T>& CVector<T>::operator=(const CTensor<T>& aCopyFrom) {
  if (mSize != aCopyFrom.size()) {
    delete[] mData;
    mSize = aCopyFrom.size();
    mData = new T[mSize];
  }
  for (register int i = 0; i < mSize; i++)
    mData[i] = aCopyFrom.data()[i];
  return *this;
}

// operator +=
template <class T>
CVector<T>& CVector<T>::operator+=(const CVector<T>& aVector) {
  #ifdef _DEBUG
  if (mSize != aVector.size()) throw EVectorIncompatibleSize(mSize,aVector.size());
  #endif
  for (int i = 0; i < mSize; i++)
    mData[i] += aVector(i);
  return *this;
}

// operator -=
template <class T>
CVector<T>& CVector<T>::operator-=(const CVector<T>& aVector) {
  #ifdef _DEBUG
  if (mSize != aVector.size()) throw EVectorIncompatibleSize(mSize,aVector.size());
  #endif
  for (int i = 0; i < mSize; i++)
    mData[i] -= aVector(i);
  return *this;
}

// operator *=
template <class T>
CVector<T>& CVector<T>::operator*=(const T aValue) {
  for (int i = 0; i < mSize; i++)
    mData[i] *= aValue;
  return *this;
}

template <class T>
T CVector<T>::operator*=(const CVector<T>& aVector) {
  #ifdef _DEBUG
  if (mSize != aVector.size()) throw EVectorIncompatibleSize(mSize,aVector.size());
  #endif
  T aSum = 0.0;
  for (int i = 0; i < mSize; i++)
    aSum += mData[i]*aVector(i);
  return aSum;
}

// operator ==
template <class T>
bool CVector<T>::operator==(const CVector<T>& aVector) {
  if (mSize != aVector.size()) return false;
  int i = 0;
  while (i < mSize && aVector(i) == mData[i])
    i++;
  return (i == mSize);
}

// operator !=
template <class T>
inline bool CVector<T>::operator!=(const CVector<T>& aVector) {
  return !((*this)==aVector);
}

// size
template <class T>
inline int CVector<T>::size() const {
  return mSize;
}

// N O N - M E M B E R   F U N C T I O N S -------------------------------------

// operator +
template <class T>
CVector<T> operator+(const CVector<T>& vec1, const CVector<T>& vec2) {
  #ifdef _DEBUG
  if (vec1.size() != vec2.size()) throw EVectorIncompatibleSize(vec1.size(),vec2.size());
  #endif
  CVector<T> result(vec1.size());
  for (int i = 0; i < vec1.size(); i++)
    result(i) = vec1[i]+vec2[i];
  return result;
}

// operator -
template <class T>
CVector<T> operator-(const CVector<T>& vec1, const CVector<T>& vec2) {
  #ifdef _DEBUG
  if (vec1.size() != vec2.size()) throw EVectorIncompatibleSize(vec1.size(),vec2.size());
  #endif
  CVector<T> result(vec1.size());
  for (int i = 0; i < vec1.size(); i++)
    result(i) = vec1(i)-vec2(i);
  return result;
}

// operator *
template <class T>
CVector<T> operator*(const T aValue, const CVector<T>& aVector) {
  CVector<T> result(aVector.size());
  for (int i = 0; i < aVector.size(); i++)
    result(i) = aValue*aVector(i);
  return result;
}

template <class T>
CVector<T> operator*(const CVector<T>& aVector, const T aValue) {
  return operator*(aValue,aVector);
}

template <class T>
T operator*(const CVector<T>& vec1, const CVector<T>& vec2) {
  #ifdef _DEBUG
  if (vec1.size() != vec2.size()) throw EVectorIncompatibleSize(vec1.size(),vec2.size());
  #endif
  T aSum = 0.0;
  for (int i = 0; i < vec1.size(); i++)
    aSum += vec1(i)*vec2(i);
  return aSum;
}

// operator /
template <class T>
CVector<T> operator/(const CVector<T>& vec1, const CVector<T>& vec2) {
  CVector<T> result(3);
  result[0]=vec1[1]*vec2[2] - vec1[2]*vec2[1];
  result[1]=vec1[2]*vec2[0] - vec1[0]*vec2[2];
  result[2]=vec1[0]*vec2[1] - vec1[1]*vec2[0];
  return result;
}

// operator <<
template <class T>
std::ostream& operator<<(std::ostream& aStream, const CVector<T>& aVector) {
  for (int i = 0; i < aVector.size(); i++)
    aStream << aVector(i) << '|';
  aStream << std::endl;
  return aStream;
}

#endif
