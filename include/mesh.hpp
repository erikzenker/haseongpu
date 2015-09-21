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


/**
 * @author Erik Zenker
 * @author Carlchristian Eckert
 * @author Marius Melzer
 * @licence GPLv3
 *
 */

#pragma once

//CLIB
#include <cfloat> /* DBL_MAX*/

// STL
#include <vector>

// Alpaka
#include <alpaka/alpaka.hpp>

// HASEonGPU
#include <geometry.hpp>


#define REFLECTION_SMALL 1E-3
#define SMALL 1E-5
#define VERY_SMALL 0.0


/**
 * @brief Contains the structure of the crystal
 *
 * All the fixed values of how the crystal is meshed
 *
 * points The coordinates of the triangle vertices
 *        All x coordinates followed by all of the y coordinates of the triangle vertices
 *        structure: [x_1, x_2, ... x_n, y_1, y_2, ... y_n] (n == numberOfPoints)
 *
 *
 * betaVolume beta values for all prisms ordered accordingly to the prismIDs:
 *            prismID = triangleID + layer * numberOfTriangles;
 *            therefore, all betaVolume for a layer are grouped together
 *
 * normalVec the normal vectors for each triangle edge
 *           first half (size: 3*numberOfTriangles -> one for each side) contains
 *           the x components of each vector, second half contains the y components.
 *           the each half is ordered as follows:
 *           [ triangle1edge0, triangle2edge0, ... triangleNedge0, triangle1edge1, triangle2edge1, ... ]
 *           i.e. all first edges of each triangle, followed by all second edges of each triangle and so on.
 *
 * centers the coordinates of the center points for each triangle
 *         All x coordinates followed by all y coordinates of the triangle vertices
 *         similar to "points"
 *
 * triangleSurfaces the sizes of the surfaces of each triangle, ordered by the triangleID
 *
 * forbiddenEdge  describes the relation of edge indices of adjacent triangles
 *           -1 means, there is no adjacent triangle to that edge
 *           0,1,2 describes the index of the edge as seen from the ADJACENT triangle
 *
 *           order of data is similar to normalVec:
 *           [ triangle1edge0, triangle2edge0, ... triangleNedge0, triangle1edge1, triangle2edge1, ... ]
 *           i.e. all first edges of each triangle, followed by all second edges of each triangle and so on.
 *
 * trianglesPointIndices contains the indices to access the "points" datastructure 
 *           (each triangle has 3 points as vertices). Each entry is an
 *           index from 0 to numberOfPoints, corresponding to the positions 
 *           of a vertex in "points".
 *           structure is similar to "forbiddenEdge":
 *           [ triangle1A, triangle2A, ... triangleNA, triangle1B, triangle2B, ... triangleNB, triangle1C, ... ]
 *           i.e. for triangles with vertices A,B,C there are all the indices
 *           of the A-vertices, followed by all the B and C vertices.
 *
 * triangleNeighbors describes the relation of triangle indices to each other.
 *           Each entry corresponds to a triangleID (see "trianglePointIndices") which
 *           is adjacent to the current triangle.
 *           structure is similar to "forbiddenEdge":
 *           [ triangle1edge0, triangle2edge0, ... triangleNedge0, triangle1edge1, triangle2edge1, ... ]
 *
 * triangleNormalPoint contains indices to the x and y components of the positions where the
 *             normalVectors start (see normalVec). For each Triangle 3 points (3 sides)
 *             are stored in this list.
 *             Indices point to locations in "points" (i.e. normal vectors start at
 *             triangle vertices!)
 *             structure is VERY similar to trianglePointIndices: 
 *             [ triangle1p0, triangle2p0, ... triangleNp0, triangle1p1, triangle2p1, ... ]
 *
 * refractiveIndices [0]->bottomInside, [1]->bottomOutside, [2]->topInside, [3]->topOutside
 * 
 * reflectivities Contains the reflectivities for upper and lower surface of gain medium
 *                Structure is based on 2 layers of triangles:
 *                [refl_tri1_bott, refl_tri2_bott, ...,refl_triN_bott, refl_tri1_top, refl_tri2_top, ..., refl_triN_top]
 * 
 * totalReflectionAngles [0]-> bottomTotalReflectionAngle, [1]-> topTotalReflectionAngle
 */

template <typename T_Acc, typename T_Dev, typename T_Stream>
class Mesh {
 public:

    using Dim =  alpaka::dim::DimInt<1u>;
    using Size = std::size_t;

    // Constants
    double claddingAbsorption;
    float surfaceTotal;
    float thickness;
    float nTot;
    float crystalTFluo;
    unsigned numberOfTriangles;
    unsigned numberOfLevels;
    unsigned numberOfPrisms;
    unsigned numberOfPoints;
    unsigned numberOfSamples;
    unsigned claddingNumber;

    // Buffers
    alpaka::mem::buf::Buf<T_Dev, double,   Dim, Size> dPoints;
    alpaka::mem::buf::Buf<T_Dev, double,   Dim, Size> dNormalVec;    
    alpaka::mem::buf::Buf<T_Dev, double,   Dim, Size> dBetaVolume;
    alpaka::mem::buf::Buf<T_Dev, double,   Dim, Size> dCenters;
    alpaka::mem::buf::Buf<T_Dev, float,    Dim, Size> dTriangleSurfaces;
    alpaka::mem::buf::Buf<T_Dev, int,      Dim, Size> dForbiddenEdge;
    alpaka::mem::buf::Buf<T_Dev, double,   Dim, Size> dBetaCells;    
    alpaka::mem::buf::Buf<T_Dev, unsigned, Dim, Size> dCladdingCellTypes;            

    alpaka::mem::buf::Buf<T_Dev, float,    Dim, Size> dRefractiveIndices;
    alpaka::mem::buf::Buf<T_Dev, float,    Dim, Size> dReflectivities;  //based on triangleIndex, with offset from bottom/top
    alpaka::mem::buf::Buf<T_Dev, float,    Dim, Size> dTotalReflectionAngles;

    alpaka::mem::buf::Buf<T_Dev, unsigned, Dim, Size> dTrianglePointIndices;
    alpaka::mem::buf::Buf<T_Dev, int,      Dim, Size> dTriangleNeighbors;
    alpaka::mem::buf::Buf<T_Dev, unsigned, Dim, Size> dTriangleNormalPoint;

    // Pointers
    double*   points;
    double*   normalVec;    
    double*   betaVolume;
    double*   centers;
    float*    triangleSurfaces;
    int*      forbiddenEdge;
    double*   betaCells;    
    unsigned* claddingCellTypes;            

    float*    refractiveIndices;
    float*    reflectivities;  //based on triangleIndex, with offset from bottom/top
    float*    totalReflectionAngles;

    unsigned* trianglePointIndices;
    int*      triangleNeighbors;
    unsigned* triangleNormalPoint;

    Mesh(// Constants
	 double claddingAbsorption,
	 float surfaceTotal,
	 float thickness,
	 float nTot,
	 float crystalTFluo,
	 unsigned numberOfTriangles,
	 unsigned numberOfLevels,
	 unsigned numberOfPrisms,
	 unsigned numberOfPoints,
	 unsigned numberOfSamples,
	 unsigned claddingNumber,
	 // Buffers
	 std::vector<double> points,
	 std::vector<double> normalVec,
	 std::vector<double> betaVolume,
	 std::vector<double> centers,
	 std::vector<float> triangleSurfaces,
	 std::vector<int> forbiddenEdge,
	 std::vector<double> betaCells,
	 std::vector<unsigned> claddingCellTypes,
	 std::vector<float> refractiveIndices,
	 std::vector<float> reflectivities,
	 std::vector<float> totalReflectionAngles,
	 std::vector<unsigned> trianglePointIndices,
	 std::vector<int> triangleNeighbors,
	 std::vector<unsigned> triangleNormalPoint,
	 // Device
	 T_Dev &dev) :
	// Constants
	claddingAbsorption(claddingAbsorption),
	surfaceTotal(surfaceTotal),
	thickness(thickness),
	nTot(nTot),
	crystalTFluo(crystalTFluo),
	numberOfTriangles(numberOfTriangles),
	numberOfLevels(numberOfLevels),
	numberOfPrisms(numberOfPrisms),
	numberOfPoints(numberOfPoints),
	numberOfSamples(numberOfSamples),
	claddingNumber(claddingNumber),//
	// Vectors
	dPoints(alpaka::mem::buf::alloc                <double,   Size, Size, T_Dev>(dev, points.size())),
	dNormalVec(alpaka::mem::buf::alloc             <double,   Size, Size, T_Dev>(dev, normalVec.size())),
	dBetaVolume(alpaka::mem::buf::alloc            <double,   Size, Size, T_Dev>(dev, betaVolume.size())),
	dCenters(alpaka::mem::buf::alloc               <double,   Size, Size, T_Dev>(dev, centers.size())),
	dTriangleSurfaces(alpaka::mem::buf::alloc      <float,    Size, Size, T_Dev>(dev, triangleSurfaces.size())),
	dForbiddenEdge(alpaka::mem::buf::alloc         <int,      Size, Size, T_Dev>(dev, forbiddenEdge.size())),
	dBetaCells(alpaka::mem::buf::alloc             <double,   Size, Size, T_Dev>(dev, betaCells.size())),
	dCladdingCellTypes(alpaka::mem::buf::alloc     <unsigned, Size, Size, T_Dev>(dev, claddingCellTypes.size())),
	dRefractiveIndices(alpaka::mem::buf::alloc     <float,    Size, Size, T_Dev>(dev, refractiveIndices.size())),
	dReflectivities(alpaka::mem::buf::alloc        <float,    Size, Size, T_Dev>(dev, reflectivities.size())),
	dTotalReflectionAngles(alpaka::mem::buf::alloc <float,    Size, Size, T_Dev>(dev, totalReflectionAngles.size())),
	dTrianglePointIndices(alpaka::mem::buf::alloc  <unsigned, Size, Size, T_Dev>(dev, trianglePointIndices.size())),
	dTriangleNeighbors(alpaka::mem::buf::alloc     <int,      Size, Size, T_Dev>(dev, triangleNeighbors.size())),
	dTriangleNormalPoint(alpaka::mem::buf::alloc   <unsigned, Size, Size, T_Dev>(dev, triangleNormalPoint.size()))
    {

	// FIXIT: Is this the most general
	using DevHost = alpaka::dev::DevCpu;
	DevHost devHost (alpaka::dev::cpu::getDev());
	T_Stream  stream  (dev);
	
	using Dim = alpaka::dim::DimInt<1u>;
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, double,   Dim, Size> hPoints(points.data(), devHost, points.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, double,   Dim, Size> hNormalVec(normalVec.data(), devHost, normalVec.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, double,   Dim, Size> hBetaVolume(betaVolume.data(), devHost, betaVolume.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, double,   Dim, Size> hCenters(centers.data(), devHost, centers.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, float,    Dim, Size> hTriangleSurfaces(triangleSurfaces.data(), devHost, triangleSurfaces.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, int,      Dim, Size> hForbiddenEdge(forbiddenEdge.data(), devHost, forbiddenEdge.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, double,   Dim, Size> hBetaCells(betaCells.data(), devHost, betaCells.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, unsigned, Dim, Size> hCladdingCellTypes(claddingCellTypes.data(), devHost, claddingCellTypes.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, float,    Dim, Size> hRefractiveIndices(refractiveIndices.data(), devHost, refractiveIndices.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, float,    Dim, Size> hReflectivities(reflectivities.data(), devHost, reflectivities.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, float,    Dim, Size> hTotalReflectionsAngles(totalReflectionAngles.data(), devHost, totalReflectionAngles.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, unsigned, Dim, Size> hTrianglePointIndices(trianglePointIndices.data(), devHost, trianglePointIndices.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, int     , Dim, Size> hTriangleNeighbors(triangleNeighbors.data(), devHost, triangleNeighbors.size());
	alpaka::mem::buf::BufPlainPtrWrapper<DevHost, unsigned, Dim, Size> hTriangleNormalPoint(triangleNormalPoint.data(), devHost, triangleNormalPoint.size());		

	alpaka::mem::view::copy(stream, dPoints, hPoints, points.size());
	alpaka::mem::view::copy(stream, dNormalVec, hNormalVec, normalVec.size());
	alpaka::mem::view::copy(stream, dBetaVolume, hBetaVolume, betaVolume.size());
	alpaka::mem::view::copy(stream, dCenters, hCenters, centers.size());
	alpaka::mem::view::copy(stream, dTriangleSurfaces, hTriangleSurfaces, triangleSurfaces.size());
	alpaka::mem::view::copy(stream, dForbiddenEdge, hForbiddenEdge, forbiddenEdge.size());
	alpaka::mem::view::copy(stream, dBetaCells, hBetaCells, betaCells.size());
	alpaka::mem::view::copy(stream, dCladdingCellTypes, hCladdingCellTypes, claddingCellTypes.size());
	alpaka::mem::view::copy(stream, dRefractiveIndices, hRefractiveIndices, refractiveIndices.size());
	alpaka::mem::view::copy(stream, dReflectivities, hReflectivities, reflectivities.size());
	alpaka::mem::view::copy(stream, dTotalReflectionAngles, hTotalReflectionsAngles, totalReflectionAngles.size());
	alpaka::mem::view::copy(stream, dTrianglePointIndices, hTrianglePointIndices, trianglePointIndices.size());
	alpaka::mem::view::copy(stream, dTriangleNeighbors, hTriangleNeighbors, triangleNeighbors.size());
	alpaka::mem::view::copy(stream, dTriangleNormalPoint, hTriangleNormalPoint, triangleNormalPoint.size());

	// Pointers
	this->points = alpaka::mem::view::getPtrNative(dPoints);
	this->normalVec = alpaka::mem::view::getPtrNative(dNormalVec);    
	this->betaVolume = alpaka::mem::view::getPtrNative(dBetaVolume);
	this->centers = alpaka::mem::view::getPtrNative(dCenters);
	this->triangleSurfaces = alpaka::mem::view::getPtrNative(dTriangleSurfaces);
	this->forbiddenEdge = alpaka::mem::view::getPtrNative(dForbiddenEdge);
	this->betaCells = alpaka::mem::view::getPtrNative(dBetaCells);    
	this->claddingCellTypes = alpaka::mem::view::getPtrNative(dCladdingCellTypes);            

	this->refractiveIndices = alpaka::mem::view::getPtrNative(dRefractiveIndices);
	this->reflectivities = alpaka::mem::view::getPtrNative(dReflectivities);  //based on triangleIndex, with offset from bottom/top
	this->totalReflectionAngles = alpaka::mem::view::getPtrNative(dTotalReflectionAngles);

	this->trianglePointIndices = alpaka::mem::view::getPtrNative(dTrianglePointIndices);
	this->triangleNeighbors = alpaka::mem::view::getPtrNative(dTriangleNeighbors);
	this->triangleNormalPoint = alpaka::mem::view::getPtrNative(dTriangleNormalPoint);
    }

    template <class T, class B, class E>
    void assertRange(const std::vector<T> &v, const B minElement,const E maxElement, const bool equals){
	if(equals){
	    assert(*std::min_element(v.begin(),v.end()) == minElement);
	    assert(*std::max_element(v.begin(),v.end()) == maxElement);
	}else{
	    assert(*std::min_element(v.begin(),v.end()) >= minElement);
	    assert(*std::max_element(v.begin(),v.end()) <= maxElement);
	}
    }

    template <class T, class B>
    void assertMin(const std::vector<T>& v,const  B minElement,const bool equals){
	if(equals){
	    assert(*std::min_element(v.begin(),v.end()) == minElement);
	}else{
	    assert(*std::min_element(v.begin(),v.end()) >= minElement);
	}
    }


    /**
     * @brief fetch the id of an adjacent triangle
     *
     * @param triangle the index of the triangle of which you want the neighbor
     * @param edge the side of the triangle, for whih you want the neighbor
     *
     * @return the index of the neighbor triangle
     */
    ALPAKA_FN_ACC int getNeighbor(unsigned triangle, int edge) const{
	return triangleNeighbors[triangle + edge*numberOfTriangles];
    }

    /**
     * @brief generate a random point within a prism
     *
     * @param triangle the triangle to describe the desired prism
     * @param the level of the desired prism
     * @param *globalState a global state for the Mersenne Twister PRNG
     *
     * @return random 3D point inside the desired prism
     *
     * Uses a Mersenne Twister PRNG and Barycentric coordinates to generate a
     * random position inside a given triangle in a specific depth
     */
    


    //FIXIT: use random number generator of alpaka (picongpu: src/libPMACC/startposition/RandImpl)

    template <typename T_Rand>
    ALPAKA_FN_ACC Point genRndPoint(T_Rand &rndGenerator, unsigned triangle, unsigned level) const{
	// Random number generator
	Point startPoint = {0,0,0};
	double u = rndGenerator();//dist(gen); // curand_uniform_double(&globalState[blockIdx.x]);
	double v = rndGenerator();//dist(gen); // curand_uniform_double(&globalState[blockIdx.x]);

	if((u+v)>1) {
		u = 1-u;
		v = 1-v;
	}
	
	double w = 1-u-v;
	int t1 = trianglePointIndices[triangle];
	int t2 = trianglePointIndices[triangle + numberOfTriangles];
	int t3 = trianglePointIndices[triangle + 2 * numberOfTriangles];

	// convert the random startpoint into coordinates
	startPoint.z = (level + rndGenerator()) * thickness;
	startPoint.x =
	    (points[t1] * u) +
	    (points[t2] * v) +
	    (points[t3] * w);
	startPoint.y =
	    (points[t1+numberOfPoints] * u) +
	    (points[t2+numberOfPoints] * v) +
	    (points[t3+numberOfPoints] * w);

	return startPoint;
    }



    /**
     * @brief get a betaVolume for a specific triangle and level
     *
     * @param triangle the id of the desired triangle
     * @param level the level of the desired prism
     *
     * @return a beta value
     */
    ALPAKA_FN_ACC double getBetaVolume(unsigned triangle, unsigned level) const{
	return betaVolume[triangle + level * numberOfTriangles];
    }

    /**
     * @brief get a betaVolume for a specific prism
     *
     * @param the id of the desired prism
     *
     * @return a beta value
     */
    ALPAKA_FN_ACC double getBetaVolume(unsigned prism) const{
	return betaVolume[prism];
    }

    /**
     * @brief generates a normal vector for a given side of a triangle
     *
     * @param triangle the desired triangle
     * @param the edge (0,1,2) of the triangle
     *
     * @return a normal vector with length 1
     */
    ALPAKA_FN_ACC NormalRay getNormal(unsigned triangle, int edge) const{
	NormalRay ray = { {0,0},{0,0}};
	int offset =  edge*numberOfTriangles + triangle;
	ray.p.x = points[ triangleNormalPoint[offset] ];
	ray.p.y = points[ triangleNormalPoint[offset] + numberOfPoints ];

	ray.dir.x = normalVec[offset];
	ray.dir.y = normalVec[offset + 3*numberOfTriangles];

	return ray;
    }	

    /**
     * @brief genenerates a point with the coordinates of a given vertex
     *
     * @param sample the id of the desired samplepoint
     *
     * @return the Point with correct 3D coordinates
     */
    ALPAKA_FN_ACC Point getSamplePoint(unsigned sample_i) const{
	Point p = {0,0,0};
	unsigned level = sample_i/numberOfPoints;
	p.z = level*thickness;
	unsigned pos = sample_i - (numberOfPoints * level);
	p.x = points[pos];
	p.y = points[pos + numberOfPoints];
	return p;
    }

    /**
     * @brief get a Point in the center of a prism
     *
     * @param triangle the id of the desired triangle
     * @param level the level of the desired prism
     *
     * @return a point with the coordinates (3D) of the prism center
     */
    ALPAKA_FN_ACC Point getCenterPoint(unsigned triangle,unsigned level) const{
	Point p = {0,0,(level+0.5)*thickness};
	p.x = centers[triangle];
	p.y = centers[triangle + numberOfTriangles];
	return p;
    }

    /**
     * @brief gets the edge-id which will be forbidden
     *
     * @param trianle the index of the triangle you are currently in
     * @param edge the index of the edge through which you are leaving the triangle
     *
     * @return the id of the edge, which will be forbidden in the new triangle
     *
     * The forbidden edge corresponds to the edge through which you left the
     * previous triangle (has a different index in the new triangle)
     */
    ALPAKA_FN_ACC int getForbiddenEdge(unsigned triangle,int edge) const{
	return forbiddenEdge[edge * numberOfTriangles + triangle];
    }


    ALPAKA_FN_ACC unsigned getCellType(unsigned triangle) const{
	return claddingCellTypes[triangle];
    }


    ALPAKA_FN_HOST double distance2D(const TwoDimPoint p1, const TwoDimPoint p2) const{
	return abs(sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)));
    }

    ALPAKA_FN_HOST double getMaxDistance(std::vector<TwoDimPoint> points) const{
	double maxDistance = -1;

	for(unsigned p1=0 ; p1 < points.size() ; ++p1)
	    for(unsigned p2 = p1; p2 < points.size() ; ++p2)
		maxDistance = std::max(maxDistance,distance2D(points[p1],points[p2]));

	return maxDistance;
    }

    ALPAKA_FN_HOST double calculateMaxDiameter(const double* points, const unsigned offset) const{
	TwoDimPoint minX = {DBL_MAX,0};
	TwoDimPoint minY = {0,DBL_MAX};
	TwoDimPoint maxX = {DBL_MIN,0};
	TwoDimPoint maxY = {0,DBL_MIN};

	for(unsigned p=0; p<offset; ++p){
	    TwoDimPoint np = {points[p], points[p+offset]};
	    minX = (points[p] < minX.x) ? np : minX;
	    maxX = (points[p] > maxX.x) ? np : maxX;
	}
	for(unsigned p=offset;p<2*offset;++p){
	    TwoDimPoint np = {points[p-offset],points[p]};
	    minY = points[p] < minY.y ? np : minY;
	    maxY = points[p] > maxY.y ? np : maxY;
	}

	std::vector<TwoDimPoint> extrema;
	extrema.push_back(minX);
	extrema.push_back(minY);
	extrema.push_back(maxX);
	extrema.push_back(maxY);
	

	return getMaxDistance(extrema);
    }

    ALPAKA_FN_HOST unsigned getMaxReflections (ReflectionPlane reflectionPlane) const{
	double d    = calculateMaxDiameter(points, numberOfPoints);
	float alpha = getReflectionAngle(reflectionPlane) * M_PI / 180.;
	double h    = numberOfLevels * thickness; 
	double z    = d/tan(alpha);
	return ceil(z/h);
    }

    ALPAKA_FN_HOST unsigned getMaxReflections() const{
	unsigned top = getMaxReflections(TOP_REFLECTION);
	unsigned bottom = getMaxReflections(BOTTOM_REFLECTION);
	return std::max(top, bottom);
    }

    ALPAKA_FN_HOST_ACC float getReflectivity(ReflectionPlane reflectionPlane, unsigned triangle) const{
	switch(reflectionPlane){
	case BOTTOM_REFLECTION:
	    return reflectivities[triangle];
	case TOP_REFLECTION:
	    return reflectivities[triangle + numberOfTriangles];
	}
	return 0;
    }

    ALPAKA_FN_HOST_ACC float getReflectionAngle(ReflectionPlane reflectionPlane) const{
	switch(reflectionPlane){
	case BOTTOM_REFLECTION:
	    return totalReflectionAngles[0];
	    //return asin(refractiveIndices[1]/refractiveIndices[0]);
	case TOP_REFLECTION:
	    return totalReflectionAngles[1];
	    //return asin(refractiveIndices[3]/refractiveIndices[2]);
	}
	return  0;
    }

    ALPAKA_FN_HOST_ACC float getTriangleSurface(const unsigned triangle) const{
	return triangleSurfaces[triangle];
	    
    }

    ALPAKA_FN_HOST_ACC  void test() const{
	printf("Constants:\n");
	printf("claddingAbsorption: %f\n", claddingAbsorption);
	printf("surfaceTotal: %f\n", surfaceTotal);
	printf("thickness: %f\n", thickness);
	printf("nTot: %f\n", nTot);
	printf("crystalTFluo: %f\n", crystalTFluo);
	printf("numberOfTriangles: %u\n", numberOfTriangles);
	printf("numberOfLevels: %u\n", numberOfLevels);
	printf("numberOfPrisms: %u\n", numberOfPrisms);
	printf("numberOfPoints: %u\n", numberOfPoints);
	printf("numberOfSamples: %u\n", numberOfSamples);
	printf("claddingNumber: %u\n", claddingNumber);
  
    }

};

