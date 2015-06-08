/**
 * Copyright 2013 Erik Zenker, Carlchristian Eckert, Marius Melzer
 *
 * This file is part of HASEonGPU
 *
 * HASEonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HASEonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HASEonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// STL
#include <vector>
#include <algorithm> /* std::iota */
#include <tuple>     /* std::make_tuple, std::tuple, std::get */


// HASEonGPU
#include <calc_phi_ase_mpi.hpp>
#include <calc_phi_ase.hpp>
#include <mesh.hpp>
#include <logging.hpp>
#include <progressbar.hpp>


// GrayBat
#include <graybat.hpp>
#include <pattern/FullyConnected.hpp>
#include <mapping/Roundrobin.hpp>


// Const messages
const int abortTag   = -1;
const float requestTag = -1.0;
const std::array<float, 4> requestMsg{{requestTag, 0, 0, 0}};
const std::array<int, 1> abortMsg{{abortTag}};

// Only for TESTING
typedef std::tuple<unsigned, unsigned, unsigned, Mesh&, std::vector<double>&, std::vector<double>&, double, bool, std::vector<float>&, std::vector<double>&, std::vector<unsigned>, unsigned> Experiment;

    

template <class Vertex, class Cage>
void masterFunction(Vertex master, const std::vector<unsigned> samples, Cage &cage, Experiment e){
    typedef typename Cage::Edge Edge;
    
    // Messages
    std::array<float, 4> resultMsg;
    std::array<int, 1>   sampleMsg; 
    
    for(auto sample = samples.begin(); sample != samples.end();){

	for(Edge inEdge : cage.getInEdges(master)){

	    // Receive request or results
	    cage.recv(inEdge, resultMsg);
		
	    if(resultMsg[0] == requestTag){
		sampleMsg = std::array<int, 1>{{ (int) *sample++ }};

		// Send next sample
		cage.send(inEdge.inverse(), sampleMsg);
			
	    }
	    else {
		// Process result
		unsigned sample_i      = (unsigned) (resultMsg[0]);
		std::get<8>(e).at(sample_i)   = resultMsg[1];
		std::get<9>(e).at(sample_i)       = resultMsg[2];
		std::get<10>(e).at(sample_i) = (unsigned) resultMsg[3];

		// Update progress bar
		fancyProgressBar(std::get<3>(e).numberOfSamples);

	    }
		
	}

    }
       
    // Send abort message to all slaves
    master.spread(abortMsg);
	    
}

template <class Vertex, class Cage>
void slaveFunction(const Vertex slave, const Vertex master, Cage &cage, Experiment e){
    typedef typename Cage::Edge Edge;
    
    // Messages
    std::array<float, 4> resultMsg;
    std::array<int, 1>   sampleMsg;

    float runtime = 0.0;

    bool abort = false;
    
    while(!abort){

	// Get edge to master
	Edge outEdge = cage.getEdge(slave, master);

	// Request new sampling point
	cage.send(outEdge, requestMsg);

	// Receive new sampling point or abort
	cage.recv(outEdge.inverse(), sampleMsg);
		    
	if(sampleMsg.at(0) == abortTag){
	    abort = true;
	}
	else {
	    calcPhiAse ( std::get<0>(e),
			 std::get<1>(e),
			 std::get<2>(e),
			 std::get<3>(e),
			 std::get<4>(e),
			 std::get<5>(e),
			 std::get<6>(e),
			 std::get<7>(e),
			 std::get<8>(e),
			 std::get<9>(e),
			 std::get<10>(e),
			 std::get<11>(e),
			 sampleMsg.at(0),
			 sampleMsg.at(0) + 1,
			 runtime);
			
	    unsigned sample_i = sampleMsg[0];
	    resultMsg = std::array<float, 4>{{ (float) sample_i,
					       (float) std::get<8>(e).at(sample_i),
					       (float) std::get<9>(e).at(sample_i),
					       (float) std::get<10>(e).at(sample_i) }};

	    // Send simulation results
	    cage.send(outEdge, resultMsg);
						
	}
	
    }
 
}


float calcPhiAseGrayBat ( const unsigned minRaysPerSample,
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
			  const unsigned gpu_i){

    // ONLY for TESTING
    // should be replaced by some struct soon !!
    Experiment e = std::make_tuple( minRaysPerSample,
				    maxRaysPerSample,
				    maxRepetitions,
				    mesh,
				    hSigmaA,
				    hSigmaE,
				    mseThreshold,
				    useReflections,
				    &hPhiAse,
				    &mse,
				    &totalRays,
				    gpu_i );
    
    /***************************************************************************
     * CAGE
     **************************************************************************/
    // Configuration
    typedef typename graybat::communicationPolicy::BMPI CP;
    typedef typename graybat::graphPolicy::BGL<>        GP;
    typedef typename graybat::Cage<CP, GP>              Cage;
    typedef typename Cage::Vertex                       Vertex;
    
    // Init
    Cage cage(graybat::pattern::FullyConnected(2));
    cage.distribute(graybat::mapping::Roundrobin());
    const Vertex master = cage.getVertex(0);
    
    /***************************************************************************
     * ASE SIMULATION
     **************************************************************************/
    // Create sample indices
    std::vector<unsigned> samples(mesh.numberOfSamples);
    std::iota(samples.begin(), samples.end(), 0);

    // Determine phi ase for each sample
    for(Vertex vertex : cage.hostedVertices) {

	/*******************************************************************
	 * MASTER
	 *******************************************************************/
	if(vertex == master){
	    masterFunction(vertex, samples, cage, e);

	}

	/*******************************************************************
	 * SLAVES
	 *******************************************************************/
	if(vertex != master){
	    slaveFunction(vertex, master, cage, e);

	}	

    }
    return 0;
    
}
