// consistencyChecker
// Check consistency of forward flow via backward flow.
//
// (c) Manuel Ruder, Alexey Dosovitskiy, Thomas Brox 2016

#include <algorithm>
#include <assert.h>
#include "CTensor.h"
#include "CFilter.h"

// Which certainty value motion boundaries should get. Value between 0 (uncertain) and 255 (certain).
#define MOTION_BOUNDARIE_VALUE 0

// The amount of gaussian smoothing that sould be applied. Set 0 to disable smoothing.
#define SMOOTH_STRENGH 0.8

// readMiddlebury
bool readMiddlebury(const char* filename, CTensor<float>& flow) {
  FILE *stream = fopen(filename, "rb");
  if (stream == 0) {
    std::cout << "Could not open " << filename << std::endl;
    return false;
  }
  float help;
  int dummy;
  dummy = fread(&help,sizeof(float),1,stream);
  int aXSize,aYSize;
  dummy = fread(&aXSize,sizeof(int),1,stream);
  dummy = fread(&aYSize,sizeof(int),1,stream);
  flow.setSize(aXSize,aYSize,2);
  for (int y = 0; y < flow.ySize(); y++)
    for (int x = 0; x < flow.xSize(); x++) {
      dummy = fread(&flow(x,y,0),sizeof(float),1,stream);
      dummy = fread(&flow(x,y,1),sizeof(float),1,stream);
    }
  fclose(stream);
  return true;
}

void checkConsistency(const CTensor<float>& flow1, const CTensor<float>& flow2, CMatrix<float>& reliable, int argc, char** args) {
  int xSize = flow1.xSize(), ySize = flow1.ySize();
  int size = xSize * ySize;
  CTensor<float> dx(xSize,ySize,2);
  CTensor<float> dy(xSize,ySize,2);
  CDerivative<float> derivative(3);
  NFilter::filter(flow1,dx,derivative,1,1);
  NFilter::filter(flow1,dy,1,derivative,1);
  CMatrix<float> motionEdge(xSize,ySize,0);
  for (int i = 0; i < size; i++) {
    motionEdge.data()[i] += dx.data()[i]*dx.data()[i];
    motionEdge.data()[i] += dx.data()[size+i]*dx.data()[size+i];
    motionEdge.data()[i] += dy.data()[i]*dy.data()[i];
    motionEdge.data()[i] += dy.data()[size+i]*dy.data()[size+i];
  }

  for (int ay = 0; ay < flow1.ySize(); ay++)
    for (int ax = 0; ax < flow1.xSize(); ax++) {
      float bx = ax+flow1(ax, ay, 0);
      float by = ay+flow1(ax, ay, 1);
      int x1 = floor(bx);
      int y1 = floor(by);
      int x2 = x1 + 1;
      int y2 = y1 + 1;
      if (x1 < 0 || x2 >= xSize || y1 < 0 || y2 >= ySize)
      { reliable(ax, ay) = 0.0f; continue; }
      float alphaX = bx-x1; float alphaY = by-y1;
      float a = (1.0-alphaX) * flow2(x1, y1, 0) + alphaX * flow2(x2, y1, 0);
      float b = (1.0-alphaX) * flow2(x1, y2, 0) + alphaX * flow2(x2, y2, 0);
      float u = (1.0-alphaY)*a+alphaY*b;
      a = (1.0-alphaX) * flow2(x1, y1, 1) + alphaX * flow2(x2, y1, 1);
      b = (1.0-alphaX) * flow2(x1, y2, 1) + alphaX * flow2(x2, y2, 1);
      float v = (1.0-alphaY)*a+alphaY*b;
      float cx = bx+u;
      float cy = by+v;
      float u2 = flow1(ax,ay,0);
      float v2 = flow1(ax,ay,1);
      if (((cx-ax) * (cx-ax) + (cy-ay) * (cy-ay)) >= 0.01*(u2*u2 + v2*v2 + u*u + v*v) + 0.5f) {
        // Set to a negative value so that when smoothing is applied the smoothing goes "to the outside".
        // Afterwards, we clip values below 0.
        reliable(ax, ay) = -255.0f;
        continue;
      }
      if (motionEdge(ax, ay) > 0.01 * (u2*u2+v2*v2) + 0.002f) {
        reliable(ax, ay) = MOTION_BOUNDARIE_VALUE;
        continue;
      }
    }
}

int main(int argc, char** args) {
  assert(argc >= 4);

  CTensor<float> flow1,flow2;
  readMiddlebury(args[1], flow1);
  readMiddlebury(args[2], flow2);
  
  assert(flow1.xSize() == flow2.xSize());
  assert(flow1.ySize() == flow2.ySize());
  
  int xSize = flow1.xSize(), ySize = flow1.ySize();
  
  // Check consistency of forward flow via backward flow and exclude motion boundaries
  CMatrix<float> reliable(xSize, ySize, 255.0f);
  checkConsistency(flow1, flow2, reliable, argc, args);
  
  if (SMOOTH_STRENGH > 0) {
    CSmooth<float> smooth(SMOOTH_STRENGH, 2.0f);
    NFilter::filter(reliable, smooth, smooth);
  }
  reliable.clip(0.0f, 255.0f);

  reliable.writeToPGM(args[3]);
  reliable.writeToTXT(args[3], true);
}
