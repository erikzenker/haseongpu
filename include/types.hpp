#pragma once

enum DeviceMode { NO_DEVICE_MODE, GPU_DEVICE_MODE, CPU_DEVICE_MODE};
enum ParallelMode { NO_PARALLEL_MODE, THREADED_PARALLEL_MODE, MPI_PARALLEL_MODE, GRAYBAT_PARALLEL_MODE};

struct ComputeParameters {

    ComputeParameters() {}
    
    ComputeParameters(  unsigned maxRepetitions,
		        unsigned gpu_i,
			DeviceMode deviceMode,
			ParallelMode parallelMode) :
	maxRepetitions(maxRepetitions),
	gpu_i(gpu_i),
	deviceMode(deviceMode),
	parallelMode(parallelMode){ }

    unsigned maxRepetitions;
    unsigned gpu_i;
    DeviceMode deviceMode;
    ParallelMode parallelMode;
    
};

struct Result {

    Result(){}
    
    Result( std::vector<float> hPhiAse,
	    std::vector<double> mse,
	    std::vector<unsigned> totalRays) :
	hPhiAse(hPhiAse),
	mse(mse),
	totalRays(totalRays) {}
  

    std::vector<float> hPhiAse;
    std::vector<double> mse;
    std::vector<unsigned> totalRays;



};

struct ExperimentParameters {

    ExperimentParameters() {}
    
    ExperimentParameters(  unsigned minRaysPerSample,
		 unsigned maxRaysPerSample,
		 std::vector<double> hSigmaA,
		 std::vector<double> hSigmaE,
		 double mseThreshold,
		 bool useReflections) :
	minRaysPerSample(minRaysPerSample),
	maxRaysPerSample(maxRaysPerSample),
	hSigmaA(hSigmaA),
	hSigmaE(hSigmaE),
	mseThreshold(mseThreshold),
	useReflections(useReflections) { }

     unsigned minRaysPerSample;
     unsigned maxRaysPerSample;
     std::vector<double> hSigmaA;
     std::vector<double> hSigmaE;
     double mseThreshold;
     bool useReflections;


};

    
