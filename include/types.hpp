#pragma once

struct Experiment {

    Experiment( const unsigned minRaysPerSample,
		const unsigned maxRaysPerSample,
		const unsigned maxRepetitions,
		const Mesh& mesh,
		const std::vector<double>& hSigmaA,
		const std::vector<double>& hSigmaE,
		const double mseThreshold,
		const bool useReflections,
		std::vector<float> &hPhiAse,
		std::vector<double> &mse,
		std::vector<unsigned> &totalRays,
		const unsigned gpu_i) :
	minRaysPerSample(minRaysPerSample),
	maxRaysPerSample(maxRaysPerSample),
	maxRepetitions(maxRepetitions),
	mesh(mesh),
	hSigmaA(hSigmaA),
	hSigmaE(hSigmaE),
	mseThreshold(mseThreshold),
	useReflections(useReflections),
	hPhiAse(hPhiAse),
	mse(mse),
        totalRays(totalRays),
	gpu_i(gpu_i) { }
	

    const unsigned minRaysPerSample;
    const unsigned maxRaysPerSample;
    const unsigned maxRepetitions;
    const Mesh& mesh;
    const std::vector<double>& hSigmaA;
    const std::vector<double>& hSigmaE;
    const double mseThreshold;
    const bool useReflections;
    std::vector<float> &hPhiAse;
    std::vector<double> &mse;
    std::vector<unsigned> &totalRays;
    const unsigned gpu_i;

};

    
