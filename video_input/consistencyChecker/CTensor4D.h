// CTensor4D
// A four-dimensional array
//
// Author: Thomas Brox
// Last change: 05.11.2001
//-------------------------------------------------------------------------
// Note:
// There is a difference between the GNU Compiler's STL and the standard
// concerning the definition and usage of string streams as well as substrings.
// Thus if using a GNU Compiler you should write #define GNU_COMPILER at the
// beginning of your program.
//
// Another Note:
// Linker problems occured in connection with <vector> from the STL.
// In this case you should include this file in a namespace.
// Example:
// namespace NTensor4D {
//   #include <CTensor4D.h>
// }
// After including other packages you can then write:
// using namespace NTensor4D;

#ifndef CTENSOR4D_H
#define CTENSOR4D_H

#include <iostream>
#include <fstream>
#include <string>
#ifdef GNU_COMPILER
  #include <strstream>
#else
  #include <sstream>
#endif
#include "CTensor.h"

template <class T>
class CTensor4D {
public:
  // constructor
  inline CTensor4D();
  inline CTensor4D(const int aXSize, const int aYSize, const int aZSize, const int aASize);
  // copy constructor
  CTensor4D(const CTensor4D<T>& aCopyFrom);
  // constructor with implicit filling
  CTensor4D(const int aXSize, const int aYSize, const int aZSize, const int aASize, const T aFillValue);
  // destructor
  virtual ~CTensor4D();

  // Changes the size of the tensor, data will be lost
  void setSize(int aXSize, int aYSize, int aZSize, int aASize);
  // Downsamples the tensor
  void downsample(int aNewXSize, int aNewYSize);
  void downsample(int aNewXSize, int aNewYSize, int aNewZSize);
  // Upsamples the tensor
  void upsample(int aNewXSize, int aNewYSize);
  void upsampleBilinear(int aNewXSize, int aNewYSize);
  void upsampleTrilinear(int aNewXSize, int aNewYSize, int aNewZSize);
  // Fills the tensor with the value aValue (see also operator =)
  void fill(const T aValue);
  // Copies a box from the tensor into aResult, the size of aResult will be adjusted
  void cut(CTensor4D<T>& aResult, int x1, int y1, int z1, int a1, int x2, int y2, int z2, int a2);
  // Reads data from a list of PPM or PGM files given in a text file
  void readFromFile(char* aFilename);
  // Writes a set of colour images to a large PPM image
  void writeToPPM(const char* aFilename, int aCols = 0, int aRows = 0);

  // Gives full access to tensor's values
  inline T& operator()(const int ax, const int ay, const int az, const int aa) const;
  // Read access with bilinear interpolation
  CVector<T> operator()(const float ax, const float ay, const int aa) const;
  // Fills the tensor with the value aValue (equivalent to fill())
  inline CTensor4D<T>& operator=(const T aValue);
  // Copies the tensor aCopyFrom to this tensor (size of tensor might change)
  CTensor4D<T>& operator=(const CTensor4D<T>& aCopyFrom);
  // Multiplication with a scalar
  CTensor4D<T>& operator*=(const T aValue);
  // Component-wise addition
  CTensor4D<T>& operator+=(const CTensor4D<T>& aTensor);

  // Gives access to the tensor's size
  inline int xSize() const;
  inline int ySize() const;
  inline int zSize() const;
  inline int aSize() const;
  inline int size() const;
  // Returns the aath layer of the 4D-tensor as 3D-tensor
  CTensor<T> getTensor3D(const int aa) const;
  // Removes one dimension and returns the resulting 3D-tensor
  void getTensor3D(CTensor<T>& aTensor, int aIndex, int aDim = 3) const;
  // Copies the components of a 3D-tensor in the aDimth layer of the 4D-tensor
  void putTensor3D(CTensor<T>& aTensor, int aIndex, int aDim = 3);
    // Removes two dimensions and returns the resulting matrix
  void getMatrix(CMatrix<T>& aMatrix, int aZIndex, int aAIndex) const;
  // Copies the components of a 3D-tensor in the aDimth layer of the 4D-tensor
  void putMatrix(CMatrix<T>& aMatrix, int aZIndex, int aAIndex);
  // Gives access to the internal data representation (use sparingly)
  inline T* data() const;
protected:
  int mXSize,mYSize,mZSize,mASize;
  T *mData;
};

// Provides basic output functionality (only appropriate for very small tensors)
template <class T> std::ostream& operator<<(std::ostream& aStream, const CTensor4D<T>& aTensor);

// Exceptions thrown by CTensor-------------------------------------------------

// Thrown when one tries to access an element of a tensor which is out of
// the tensor's bounds
struct ETensor4DRangeOverflow {
  ETensor4DRangeOverflow(const int ax, const int ay, const int az, const int aa) {
    using namespace std;
    cerr << "Exception ETensor4DRangeOverflow: x = " << ax << ", y = " << ay << ", z = " << az << ", a = " << aa << endl;
  }
};

// Thrown from getTensor3D if the parameter's size does not match with the size
// of this tensor
struct ETensor4DIncompatibleSize {
  ETensor4DIncompatibleSize(int ax, int ay, int az, int ax2, int ay2, int az2) {
    using namespace std;
    cerr << "Exception ETensor4DIncompatibleSize: x = " << ax << ":" << ax2;
    cerr << ", y = " << ay << ":" << ay2;
    cerr << ", z = " << az << ":" << az2 << endl;
  }
};

// Thrown from readFromFile if the file format is unknown
struct ETensor4DInvalidFileFormat {
  ETensor4DInvalidFileFormat() {
    using namespace std;
    cerr << "Exception ETensor4DInvalidFileFormat" << endl;
  }
};

// I M P L E M E N T A T I O N --------------------------------------------
//
// You might wonder why there is implementation code in a header file.
// The reason is that not all C++ compilers yet manage separate compilation
// of templates. Inline functions cannot be compiled separately anyway.
// So in this case the whole implementation code is added to the header
// file.
// Users of CTensor4D should ignore everything that's beyond this line :)
// ------------------------------------------------------------------------

// P U B L I C ------------------------------------------------------------

// constructor
template <class T>
inline CTensor4D<T>::CTensor4D() {
  mData = 0; mXSize = 0; mYSize = 0; mZSize = 0; mASize = 0;
}

// constructor
template <class T>
inline CTensor4D<T>::CTensor4D(const int aXSize, const int aYSize, const int aZSize, const int aASize)
  : mXSize(aXSize), mYSize(aYSize), mZSize(aZSize), mASize(aASize) {
  mData = new T[aXSize*aYSize*aZSize*aASize];
}

// copy constructor
template <class T>
CTensor4D<T>::CTensor4D(const CTensor4D<T>& aCopyFrom)
  : mXSize(aCopyFrom.mXSize), mYSize(aCopyFrom.mYSize), mZSize(aCopyFrom.mZSize), mASize(aCopyFrom.mASize) {
  int wholeSize = mXSize*mYSize*mZSize*mASize;
  mData = new T[wholeSize];
  for (register int i = 0; i < wholeSize; i++)
    mData[i] = aCopyFrom.mData[i];
}

// constructor with implicit filling
template <class T>
CTensor4D<T>::CTensor4D(const int aXSize, const int aYSize, const int aZSize, const int aASize, const T aFillValue)
  : mXSize(aXSize), mYSize(aYSize), mZSize(aZSize), mASize(aASize) {
  mData = new T[aXSize*aYSize*aZSize*aASize];
  fill(aFillValue);
}

// destructor
template <class T>
CTensor4D<T>::~CTensor4D() {
  delete[] mData;
}

// setSize
template <class T>
void CTensor4D<T>::setSize(int aXSize, int aYSize, int aZSize, int aASize) {
  if (mData != 0) delete[] mData;
  mData = new T[aXSize*aYSize*aZSize*aASize];
  mXSize = aXSize;
  mYSize = aYSize;
  mZSize = aZSize;
  mASize = aASize;
}

//downsample
template <class T>
void CTensor4D<T>::downsample(int aNewXSize, int aNewYSize) {
  T* mData2 = new T[aNewXSize*aNewYSize*mZSize*mASize];
  int aSize = aNewXSize*aNewYSize;
  for (int a = 0; a < mASize; a++)
    for (int z = 0; z < mZSize; z++) {
      CMatrix<T> aTemp(mXSize,mYSize);
      getMatrix(aTemp,z,a);
      aTemp.downsample(aNewXSize,aNewYSize);
      for (int i = 0; i < aSize; i++)
        mData2[i+(a*mZSize+z)*aSize] = aTemp.data()[i];
    }
  delete[] mData;
  mData = mData2;
  mXSize = aNewXSize;
  mYSize = aNewYSize;
}

template <class T>
void CTensor4D<T>::downsample(int aNewXSize, int aNewYSize, int aNewZSize) {
  T* mData2 = new T[aNewXSize*aNewYSize*aNewZSize*mASize];
  int aSize = aNewXSize*aNewYSize*aNewZSize;
  for (int a = 0; a < mASize; a++) {
    CTensor<T> aTemp(mXSize,mYSize,mZSize);
    getTensor3D(aTemp,a);
    aTemp.downsample(aNewXSize,aNewYSize,aNewZSize);
    for (int i = 0; i < aSize; i++)
      mData2[i+a*aSize] = aTemp.data()[i];
  }
  delete[] mData;
  mData = mData2;
  mXSize = aNewXSize;
  mYSize = aNewYSize;
  mZSize = aNewZSize;
}

// upsample
template <class T>
void CTensor4D<T>::upsample(int aNewXSize, int aNewYSize) {
  T* mData2 = new T[aNewXSize*aNewYSize*mZSize*mASize];
  int aSize = aNewXSize*aNewYSize;
  for (int a = 0; a < mASize; a++)
    for (int z = 0; z < mZSize; z++) {
      CMatrix<T> aTemp(mXSize,mYSize);
      getMatrix(aTemp,z,a);
      aTemp.upsample(aNewXSize,aNewYSize);
      for (int i = 0; i < aSize; i++)
        mData2[i+(a*mZSize+z)*aSize] = aTemp.data()[i];
    }
  delete[] mData;
  mData = mData2;
  mXSize = aNewXSize;
  mYSize = aNewYSize;
}

// upsampleBilinear
template <class T>
void CTensor4D<T>::upsampleBilinear(int aNewXSize, int aNewYSize) {
  T* mData2 = new T[aNewXSize*aNewYSize*mZSize*mASize];
  int aSize = aNewXSize*aNewYSize;
  for (int a = 0; a < mASize; a++)
    for (int z = 0; z < mZSize; z++) {
      CMatrix<T> aTemp(mXSize,mYSize);
      getMatrix(aTemp,z,a);
      aTemp.upsampleBilinear(aNewXSize,aNewYSize);
      for (int i = 0; i < aSize; i++)
        mData2[i+(a*mZSize+z)*aSize] = aTemp.data()[i];
    }
  delete[] mData;
  mData = mData2;
  mXSize = aNewXSize;
  mYSize = aNewYSize;
}

// upsampleTrilinear
template <class T>
void CTensor4D<T>::upsampleTrilinear(int aNewXSize, int aNewYSize, int aNewZSize) {
  T* mData2 = new T[aNewXSize*aNewYSize*aNewZSize*mASize];
  int aSize = aNewXSize*aNewYSize*aNewZSize;
  for (int a = 0; a < mASize; a++) {
    CTensor<T> aTemp(mXSize,mYSize,mZSize);
    getTensor3D(aTemp,a);
    aTemp.upsampleTrilinear(aNewXSize,aNewYSize,aNewZSize);
    for (int i = 0; i < aSize; i++)
      mData2[i+a*aSize] = aTemp.data()[i];
  }
  delete[] mData;
  mData = mData2;
  mXSize = aNewXSize;
  mYSize = aNewYSize;
  mZSize = aNewZSize;
}

// fill
template <class T>
void CTensor4D<T>::fill(const T aValue) {
  int wholeSize = mXSize*mYSize*mZSize*mASize;
  for (register int i = 0; i < wholeSize; i++)
    mData[i] = aValue;
}

// cut
template <class T>
void CTensor4D<T>::cut(CTensor4D<T>& aResult, int x1, int y1, int z1, int a1, int x2, int y2, int z2, int a2) {
  aResult.mXSize = x2-x1+1;
  aResult.mYSize = y2-y1+1;
  aResult.mZSize = z2-z1+1;
  aResult.mASize = a2-a1+1;
  delete[] aResult.mData;
  aResult.mData = new T[aResult.mXSize*aResult.mYSize*aResult.mZSize*aResult.mASize];
  for (int a = a1; a <= a2; a++)
    for (int z = z1; z <= z2; z++)
      for (int y = y1; y <= y2; y++)
        for (int x = x1; x <= x2; x++)
          aResult(x-x1,y-y1,z-z1,a-a1) = operator()(x,y,z,a);
}

// readFromFile
template <class T>
void CTensor4D<T>::readFromFile(char* aFilename) {
  if (mData != 0) delete[] mData;
  std::string s;
  std::string aPath = aFilename;
  aPath.erase(aPath.find_last_of('\\')+1,100);
  mASize = 0;
  {
    std::ifstream aStream(aFilename);
    while (!aStream.eof()) {
      aStream >> s;
      if (s != "") {
        mASize++;
        if (mASize == 1) {
          s.erase(0,s.find_last_of('.'));
          if (s == ".ppm" || s == ".PPM") mZSize = 3;
          else if (s == ".pgm" || s == ".PGM") mZSize = 1;
          else throw ETensor4DInvalidFileFormat();
        }
      }
    }
  }
  std::ifstream aStream(aFilename);
  aStream >> s;
  s = aPath+s;
  // PGM
  if (mZSize == 1) {
    CMatrix<float> aTemp;
    aTemp.readFromPGM(s.c_str());
    mXSize = aTemp.xSize();
    mYSize = aTemp.ySize();
    int aSize = mXSize*mYSize;
    mData = new T[aSize*mASize];
    for (int i = 0; i < aSize; i++)
      mData[i] = aTemp.data()[i];
    for (int a = 1; a < mASize; a++) {
      aStream >> s;
      s = aPath+s;
      aTemp.readFromPGM(s.c_str());
      for (int i = 0; i < aSize; i++)
        mData[i+a*aSize] = aTemp.data()[i];
    }
  }
  // PPM
  else {
    CTensor<float> aTemp;
    aTemp.readFromPPM(s.c_str());
    mXSize = aTemp.xSize();
    mYSize = aTemp.ySize();
    int aSize = 3*mXSize*mYSize;
    mData = new T[aSize*mASize];
    for (int i = 0; i < aSize; i++)
      mData[i] = aTemp.data()[i];
    for (int a = 1; a < mASize; a++) {
      aStream >> s;
      s = aPath+s;
      aTemp.readFromPPM(s.c_str());
      for (int i = 0; i < aSize; i++)
        mData[i+a*aSize] = aTemp.data()[i];
    }
  }
}

// writeToPPM
template <class T>
void CTensor4D<T>::writeToPPM(const char* aFilename, int aCols, int aRows) {
  int rows = (int)floor(sqrt(mASize));
  if (aRows != 0) rows = aRows;
  int cols = (int)ceil(mASize*1.0/rows);
  if (aCols != 0) cols = aCols;
  FILE* outimage = fopen(aFilename, "wb");
  fprintf(outimage, "P6 \n");
  fprintf(outimage, "%ld %ld \n255\n", cols*mXSize,rows*mYSize);
  for (int r = 0; r < rows; r++)
    for (int y = 0; y < mYSize; y++)
      for (int c = 0; c < cols; c++)
        for (int x = 0; x < mXSize; x++) {
          unsigned char aHelp;
          if (r*cols+c >= mASize) aHelp = 0;
          else aHelp = (unsigned char)operator()(x,y,0,r*cols+c);
          fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
          if (r*cols+c >= mASize) aHelp = 0;
          else aHelp = (unsigned char)operator()(x,y,1,r*cols+c);
          fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
          if (r*cols+c >= mASize) aHelp = 0;
          else aHelp = (unsigned char)operator()(x,y,2,r*cols+c);
          fwrite (&aHelp, sizeof(unsigned char), 1, outimage);
        }
  fclose(outimage);
}

// operator ()
template <class T>
inline T& CTensor4D<T>::operator()(const int ax, const int ay, const int az, const int aa) const {
  #ifdef DEBUG
    if (ax >= mXSize || ay >= mYSize || az >= mZSize || aa >= mASize || ax < 0 || ay < 0 || az < 0 || aa < 0)
      throw ETensorRangeOverflow(ax,ay,az,aa);
  #endif
  return mData[mXSize*(mYSize*(mZSize*aa+az)+ay)+ax];
}

template <class T>
CVector<T> CTensor4D<T>::operator()(const float ax, const float ay, const int aa) const {
  CVector<T> aResult(mZSize);
  int x1 = (int)ax;
  int y1 = (int)ay;
  int x2 = x1+1;
  int y2 = y1+1;
  #ifdef _DEBUG
  if (x2 >= mXSize || y2 >= mYSize || x1 < 0 || y1 < 0) throw ETensorRangeOverflow(ax,ay,0);
  #endif
  float alphaX = ax-x1; float alphaXTrans = 1.0-alphaX;
  float alphaY = ay-y1; float alphaYTrans = 1.0-alphaY;
  for (int k = 0; k < mZSize; k++) {
    float a = alphaXTrans*operator()(x1,y1,k,aa)+alphaX*operator()(x2,y1,k,aa);
    float b = alphaXTrans*operator()(x1,y2,k,aa)+alphaX*operator()(x2,y2,k,aa);
    aResult(k) = alphaYTrans*a+alphaY*b;
  }
  return aResult;
}

// operator =
template <class T>
inline CTensor4D<T>& CTensor4D<T>::operator=(const T aValue) {
  fill(aValue);
  return *this;
}

template <class T>
CTensor4D<T>& CTensor4D<T>::operator=(const CTensor4D<T>& aCopyFrom) {
  if (this != &aCopyFrom) {
    if (mData != 0) delete[] mData;
    mXSize = aCopyFrom.mXSize;
    mYSize = aCopyFrom.mYSize;
    mZSize = aCopyFrom.mZSize;
    mASize = aCopyFrom.mASize;
    int wholeSize = mXSize*mYSize*mZSize*mASize;
    mData = new T[wholeSize];
    for (register int i = 0; i < wholeSize; i++)
      mData[i] = aCopyFrom.mData[i];
  }
  return *this;
}

// operator *=
template <class T>
CTensor4D<T>& CTensor4D<T>::operator*=(const T aValue) {
  int wholeSize = mXSize*mYSize*mZSize*mASize;
  for (int i = 0; i < wholeSize; i++)
    mData[i] *= aValue;
  return *this;
}

// operator +=
template <class T>
CTensor4D<T>& CTensor4D<T>::operator+=(const CTensor4D<T>& aTensor) {
  #ifdef _DEBUG
  if (mXSize != aTensor.mXSize || mYSize != aTensor.mYSize || mZSize != aTensor.mZSize || mASize != aTensor.mASize)
    throw ETensorIncompatibleSize(mXSize,mYSize,mZSize);
  #endif
  int wholeSize = size();
  for (int i = 0; i < wholeSize; i++)
    mData[i] += aTensor.mData[i];
  return *this;
}

// xSize
template <class T>
inline int CTensor4D<T>::xSize() const {

  return mXSize;
}

// ySize
template <class T>
inline int CTensor4D<T>::ySize() const {
  return mYSize;
}

// zSize
template <class T>
inline int CTensor4D<T>::zSize() const {
  return mZSize;
}

// aSize
template <class T>
inline int CTensor4D<T>::aSize() const {
  return mASize;
}

// size
template <class T>
inline int CTensor4D<T>::size() const {
  return mXSize*mYSize*mZSize*mASize;
}

// getTensor3D
template <class T>
CTensor<T> CTensor4D<T>::getTensor3D(const int aa) const {
  CTensor<T> aTemp(mXSize,mYSize,mZSize);
  int aTensorSize = mXSize*mYSize*mZSize;
  int aOffset = aa*aTensorSize;
  for (int i = 0; i < aTensorSize; i++)
    aTemp.data()[i] = mData[i+aOffset];
  return aTemp;
}

// getTensor3D
template <class T>
void CTensor4D<T>::getTensor3D(CTensor<T>& aTensor, int aIndex, int aDim) const {
  int aSize;
  int aOffset;
  switch (aDim) {
  case 3:
    if (aTensor.xSize() != mXSize || aTensor.ySize() != mYSize || aTensor.zSize() != mZSize)
      throw ETensor4DIncompatibleSize(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),mXSize,mYSize,mZSize);
    aSize = mXSize*mYSize*mZSize;
    aOffset = aIndex*aSize;
    for (int i = 0; i < aSize; i++)
      aTensor.data()[i] = mData[i+aOffset];
    break;
  case 2:
    if (aTensor.xSize() != mXSize || aTensor.ySize() != mYSize || aTensor.zSize() != mASize)
      throw ETensor4DIncompatibleSize(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),mXSize,mYSize,mASize);
    aSize = mXSize*mYSize;
    aOffset = aIndex*aSize;
    for (int a = 0; a < mASize; a++) 
      for (int i = 0; i < aSize; i++)
        aTensor.data()[i+a*aSize] = mData[i+aOffset+a*aSize*mZSize];
    break;
  case 1:
    if (aTensor.xSize() != mXSize || aTensor.ySize() != mZSize || aTensor.zSize() != mASize)
      throw ETensor4DIncompatibleSize(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),mXSize,mZSize,mASize);
    for (int a = 0; a < mASize; a++)
      for (int z = 0; z < mZSize; z++)
        for (int x = 0; x < mXSize; x++)
          aTensor(x,z,a) = operator()(x,aIndex,z,a);
    break;
  case 0:
    if (aTensor.xSize() != mYSize || aTensor.ySize() != mZSize || aTensor.zSize() != mASize)
      throw ETensor4DIncompatibleSize(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),mYSize,mZSize,mASize);
    for (int a = 0; a < mASize; a++)
      for (int z = 0; z < mZSize; z++)
        for (int y = 0; y < mYSize; y++)
          aTensor(y,z,a) = operator()(aIndex,y,z,a);
    break;
  default: getTensor3D(aTensor,aIndex);
  }
}

// putTensor3D
template <class T>
void CTensor4D<T>::putTensor3D(CTensor<T>& aTensor, int aIndex, int aDim) {
  int aSize;
  int aOffset;
  switch (aDim) {
  case 3:
    if (aTensor.xSize() != mXSize || aTensor.ySize() != mYSize || aTensor.zSize() != mZSize)
      throw ETensor4DIncompatibleSize(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),mXSize,mYSize,mZSize);
    aSize = mXSize*mYSize*mZSize;
    aOffset = aIndex*aSize;
    for (int i = 0; i < aSize; i++)
      mData[i+aOffset] = aTensor.data()[i];
    break;
  case 2:
    if (aTensor.xSize() != mXSize || aTensor.ySize() != mYSize || aTensor.zSize() != mASize)
      throw ETensor4DIncompatibleSize(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),mXSize,mYSize,mASize);
    aSize = mXSize*mYSize;
    aOffset = aIndex*aSize;
    for (int a = 0; a < mASize; a++)
      for (int i = 0; i < aSize; i++)
        mData[i+aOffset+a*aSize*mZSize] = aTensor.data()[i+a*aSize];
    break;
  case 1:
    if (aTensor.xSize() != mXSize || aTensor.ySize() != mZSize || aTensor.zSize() != mASize)
      throw ETensor4DIncompatibleSize(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),mXSize,mZSize,mASize);
    for (int a = 0; a < mASize; a++)
      for (int z = 0; z < mZSize; z++)
        for (int x = 0; x < mXSize; x++)
          operator()(x,aIndex,z,a) = aTensor(x,z,a);
    break;
  case 0:
    if (aTensor.xSize() != mYSize || aTensor.ySize() != mZSize || aTensor.zSize() != mASize)
      throw ETensor4DIncompatibleSize(aTensor.xSize(),aTensor.ySize(),aTensor.zSize(),mYSize,mZSize,mASize);
    for (int a = 0; a < mASize; a++)
      for (int z = 0; z < mZSize; z++)
        for (int y = 0; y < mYSize; y++)
          operator()(aIndex,y,z,a) = aTensor(y,z,a);
    break;
  default: putTensor3D(aTensor,aIndex);
  }
}

// getMatrix
template <class T>
void CTensor4D<T>::getMatrix(CMatrix<T>& aMatrix, int aZIndex, int aAIndex) const {
  if (aMatrix.xSize() != mXSize || aMatrix.ySize() != mYSize)
    throw ETensor4DIncompatibleSize(aMatrix.xSize(),aMatrix.ySize(),1,mXSize,mYSize,1);
  int aSize = mXSize*mYSize;
  int aOffset = aSize*(aAIndex*mZSize+aZIndex);
  for (int i = 0; i < aSize; i++)
    aMatrix.data()[i] = mData[i+aOffset];
}

// putMatrix
template <class T>
void CTensor4D<T>::putMatrix(CMatrix<T>& aMatrix, int aZIndex, int aAIndex) {
  if (aMatrix.xSize() != mXSize || aMatrix.ySize() != mYSize)
    throw ETensor4DIncompatibleSize(aMatrix.xSize(),aMatrix.ySize(),1,mXSize,mYSize,1);
  int aSize = mXSize*mYSize;
  int aOffset = aSize*(aAIndex*mZSize+aZIndex);
  for (int i = 0; i < aSize; i++)
    mData[i+aOffset] = aMatrix.data()[i];
}

// data()
template <class T>
inline T* CTensor4D<T>::data() const {
  return mData;
}

// N O N - M E M B E R  F U N C T I O N S --------------------------------------

// operator <<
template <class T>
std::ostream& operator<<(std::ostream& aStream, const CTensor4D<T>& aTensor) {
  for (int a = 0; a < aTensor.aSize(); a++) {
    for (int z = 0; z < aTensor.zSize(); z++) {
      for (int y = 0; y < aTensor.ySize(); y++) {
        for (int x = 0; x < aTensor.xSize(); x++)
          aStream << aTensor(x,y,z) << ' ';
        aStream << std::endl;
      }
      aStream << std::endl;
    }
    aStream << std::endl;
  }
  return aStream;
}

#endif
