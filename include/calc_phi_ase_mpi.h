#ifndef calcPhiAseMPI_H
#define calcPhiAseMPI_H

#include <vector>
#include <mesh.h>

float calcPhiAseMPI ( unsigned &hRaysPerSample,
		      const unsigned maxRaysPerSample,
		      const Mesh& dMesh,
		      const Mesh& hMesh,
		      const std::vector<double>& hSigmaA,
		      const std::vector<double>& hSigmaE,
		      const std::vector<float>& mseThreshold,
		      const bool useReflections,
		      std::vector<float> &hPhiAse,
		      std::vector<double> &mse,
		      std::vector<unsigned> &totalRays,
		      unsigned gpu_i,
		      unsigned maxSample_i);

#endif /* calcPhiAseMPI */