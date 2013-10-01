#include <thread.h>
#include <vector>
#include <iostream>
#include <pthread.h>

#include <mesh.h>
#include <calc_phi_ase.h>

struct calcDndtAseArgs 
{
  calcDndtAseArgs(unsigned &phostRaysPerSample,
		  const unsigned pmaxRaysPerSample,
		  const Mesh& pmesh,
		  const Mesh& phostMesh,
		  const std::vector<double>& psigmaA,
		  const std::vector<double>& psigmaE,
		  const float pexpectationThreshold,
		  const bool puseReflections,
		  std::vector<float> &pphiAse,
		  std::vector<double> &pexpectation,
      std::vector<unsigned> &pTotalRays,
		  unsigned pgpu_i,
		  unsigned pminSample_i,
		  unsigned pmaxSample_i,
		  float &pruntime): hostRaysPerSample(phostRaysPerSample),
				    maxRaysPerSample(pmaxRaysPerSample),
				    mesh(pmesh),
				    hostMesh(phostMesh),
				    sigmaA(psigmaA),
				    sigmaE(psigmaE),
				    expectationThreshold(pexpectationThreshold),
				    useReflections(puseReflections),
				    phiAse(pphiAse),
				    expectation(pexpectation),
            totalRays(pTotalRays),
				    gpu_i(pgpu_i),
				    minSample_i(pminSample_i),
				    maxSample_i(pmaxSample_i),
				    runtime(pruntime){

  }
  unsigned &hostRaysPerSample;
  const unsigned maxRaysPerSample;
  const Mesh& mesh;
  const Mesh& hostMesh;
  const std::vector<double>& sigmaA;
  const std::vector<double>& sigmaE;
  const float expectationThreshold;
  const bool useReflections;
  std::vector<float> &phiAse;
  std::vector<double> &expectation;
  std::vector<unsigned> &totalRays;
  unsigned gpu_i;
  unsigned minSample_i;
  unsigned maxSample_i;
  float &runtime;
};

void *entryPoint(void* arg){
  calcDndtAseArgs *a = (calcDndtAseArgs*) arg;
  calcPhiAse( a->hostRaysPerSample,
   	      a->maxRaysPerSample,
   	      a->mesh,
   	      a->hostMesh,
   	      a->sigmaA,
   	      a->sigmaE,
   	      a->expectationThreshold,
   	      a->useReflections,
   	      a->phiAse,
   	      a->expectation,
          a->totalRays,
   	      a->gpu_i,
   	      a->minSample_i,
   	      a->maxSample_i,
	      a->runtime);

  return arg;
}

pthread_t calcPhiAseThreaded( unsigned &hostRaysPerSample,
			      const unsigned maxRaysPerSample,
			      const Mesh& mesh,
			      const Mesh& hostMesh,
			      const std::vector<double>& sigmaA,
			      const std::vector<double>& sigmaE,
			      const float expectationThreshold,
			      const bool useReflections,
			      std::vector<float> &phiAse,
			      std::vector<double> &expectation,
			      std::vector<unsigned> &totalRays,
			      unsigned gpu_i,
			      unsigned minSample_i,
			      unsigned maxSample_i,
			      float &runtime){
  calcDndtAseArgs *args = new calcDndtAseArgs(hostRaysPerSample,
					      maxRaysPerSample,
					      mesh,
					      hostMesh,
					      sigmaA,
					      sigmaE,
					      expectationThreshold,
					      useReflections,
					      phiAse,
					      expectation,
                totalRays,
					      gpu_i,
					      minSample_i,
					      maxSample_i,
					      runtime);

  pthread_t threadId;
  pthread_create( &threadId, NULL, entryPoint, (void*) args);
  return threadId;

}

void joinAll(std::vector<pthread_t> threadIds){
  for(unsigned i = 0; i < threadIds.size(); ++i){
    pthread_join(threadIds[i], NULL);
  }

}
