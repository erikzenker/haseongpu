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
#include <iostream>  /* std::cout     */
#include <algorithm> /* std::generate */

// haseongpu
#include <calc_phi_ase_mpi.hpp>
#include <calc_phi_ase.hpp>
#include <mesh.hpp>
#include <logging.hpp>
#include <progressbar.hpp>


// GrayBat
#include <graybat.hpp>
#include <pattern/FullyConnected.hpp>
#include <mapping/Roundrobin.hpp>

struct Increment {
    unsigned current;
    Increment(unsigned init) : current(init) {}
    
    unsigned operator()(){
	return current++;
    }
};

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

    /***************************************************************************
     * CAGE
     **************************************************************************/
    // Configure
    typedef typename graybat::communicationPolicy::BMPI CP;
    typedef typename graybat::graphPolicy::BGL<>        GP;
    typedef typename graybat::Cage<CP, GP>              Cage;
    typedef typename Cage::Vertex                       Vertex;
    typedef typename Cage::Edge                         Edge;
    
    // Init
    Cage cage(graybat::pattern::FullyConnected(2));
    cage.distribute(graybat::mapping::Roundrobin());
    const Vertex master = cage.getVertex(0);

    
    /***************************************************************************
     * ASE SIMULATION
     **************************************************************************/
    // Const messages
    const int abortTag   = -1;
    const float requestTag = -1.0;
    const std::array<float, 4> requestMsg{{requestTag, 0, 0, 0}};
    const std::array<int, 1> abortMsg{{abortTag}};
    
    // Messages
    std::array<float, 4> resultMsg;
    std::array<int, 1>   sampleMsg;    

    bool abort = false;
    float runtime = 0.0;

    // Create sample indices
    std::vector<unsigned> samples(mesh.numberOfSamples);
    std::generate(samples.begin(), samples.end(), Increment(0));
    auto sample = samples.begin();

    // Determine phi ase for each sample
    while(sample != samples.end() and !abort){
    
	for(Vertex v : cage.hostedVertices) {

	    /*******************************************************************
	     * SLAVES
	     *******************************************************************/
	    if(v != master){
	    
		for(Edge outEdge : cage.getOutEdges(v)){
		    // Request new sampling point
		    cage.send(outEdge, requestMsg);

		    // Receive new sampling point or abort
		    cage.recv(outEdge.inverse(), sampleMsg);
		    
		    if(sampleMsg.at(0) == abortTag){
			abort = true;
			break;
		    }
		    else {
			calcPhiAse ( minRaysPerSample,
				     maxRaysPerSample,
				     maxRepetitions,
				     mesh,
				     hSigmaA,
				     hSigmaE,
				     mseThreshold,
				     useReflections,
				     hPhiAse,
				     mse,
				     totalRays,
				     gpu_i,
				     sampleMsg.at(0),
				     sampleMsg.at(0) + 1,
				     runtime);
			
			unsigned sample_i = sampleMsg[0];
			resultMsg = std::array<float, 4>{{ (float) sample_i,
							   (float) hPhiAse.at(sample_i),
							   (float) mse.at(sample_i),
							   (float) totalRays.at(sample_i) }};
			// Send simulation results
			cage.send(outEdge, resultMsg);
						
		    }
		    
		}

	    }

	    /*******************************************************************
	     * MASTER
	     *******************************************************************/
	    if(v == master){

		for(Edge inEdge : cage.getInEdges(v)){
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
			hPhiAse.at(sample_i)   = resultMsg[1];
			mse.at(sample_i)       = resultMsg[2];
			totalRays.at(sample_i) = (unsigned) resultMsg[3];

			// Update progress bar
			fancyProgressBar(mesh.numberOfSamples);

		    }
		
		}
	    
	    }

	}
	
    }
    
    // Send abort message to all slaves
    for(Vertex v : cage.hostedVertices) {

    	if(v == master){
    	    v.spread(abortMsg);
    	}
	
    }

    return 0;
    
}
