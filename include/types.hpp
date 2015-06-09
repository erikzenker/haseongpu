#pragma once

struct ComputeParameters {

    ComputeParameters( const unsigned maxRepetitions,
		       const unsigned gpu_i) :
	maxRepetitions(maxRepetitions),
	gpu_i(gpu_i) { }

    const unsigned maxRepetitions;
    const unsigned gpu_i;
    
};

struct Result {

    Result( std::vector<float> &hPhiAse,
	    std::vector<double> &mse,
	    std::vector<unsigned> &totalRays) :
	hPhiAse(hPhiAse),
	mse(mse),
	totalRays(totalRays) {}
  

    std::vector<float> &hPhiAse;
    std::vector<double> &mse;
    std::vector<unsigned> &totalRays;



};

struct ExperimentParameters {

    ExperimentParameters( const unsigned minRaysPerSample,
		const unsigned maxRaysPerSample,
		const std::vector<double>& hSigmaA,
		const std::vector<double>& hSigmaE,
		const double mseThreshold,
		const bool useReflections) :
	minRaysPerSample(minRaysPerSample),
	maxRaysPerSample(maxRaysPerSample),
	hSigmaA(hSigmaA),
	hSigmaE(hSigmaE),
	mseThreshold(mseThreshold),
	useReflections(useReflections) { }

    const unsigned minRaysPerSample;
    const unsigned maxRaysPerSample;
    const std::vector<double>& hSigmaA;
    const std::vector<double>& hSigmaE;
    const double mseThreshold;
    const bool useReflections;


};

    
