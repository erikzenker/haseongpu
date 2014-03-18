#pragma once

#include <vector>
#include <iostream>

#include <cudachecks.h>
#include <logging.h>
#include <vector>

static const unsigned MIN_COMPUTE_CAPABILITY_MAJOR = 2;
static const unsigned MIN_COMPUTE_CAPABILITY_MINOR = 0;


/**
 * @brief Copy data from host to device
 *
 **/
template <class T>
T* copyToDevice(const std::vector<T> &v){
  T* deviceV;
  CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceV,  v.size()* sizeof(T)));
  CUDA_CHECK_RETURN(cudaMemcpy(deviceV, &(v[0]), v.size() * sizeof(T), cudaMemcpyHostToDevice));

  return deviceV;
}

template <class T>
void copyToDevice(const std::vector<T> &v,  T* deviceV){
  CUDA_CHECK_RETURN(cudaMemcpy(deviceV, &(v[0]), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
void copyToDevice(const std::vector<T> &v,  T* deviceV, unsigned size){
  assert(size <= v.size());
  CUDA_CHECK_RETURN(cudaMemcpy(deviceV, &(v[0]), size * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
T* copyToDevice(T a){
  T* deviceV;
  CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceV, sizeof(T)));
  CUDA_CHECK_RETURN(cudaMemcpy(deviceV, &a, sizeof(T), cudaMemcpyHostToDevice));

  return deviceV;
}

template <class T>
void copyToDevice(const T a, T* deviceV){
  CUDA_CHECK_RETURN(cudaMemcpy(deviceV, &a, sizeof(T), cudaMemcpyHostToDevice));
}


/**
 * @brief Copy data from device to host
 *
 **/
template <class T>
void copyFromDevice(std::vector<T> &v, const T* deviceV){
  CUDA_CHECK_RETURN(cudaMemcpy(&(v[0]), deviceV, v.size() * sizeof(T), cudaMemcpyDeviceToHost));

}

template <class T>
T copyFromDevice(const T* deviceV){
  T a;
  CUDA_CHECK_RETURN(cudaMemcpy(&a, deviceV, sizeof(T), cudaMemcpyDeviceToHost));
  
  return a;
}

/**
 * @brief Vector on host and array on device
 *        with transparent access. Access is
 *        triggered based on a compiler macro.
 *
 **/
template<class T>
class constHybridVector {
public:

  constHybridVector(std::vector<T> &srcV) :
    hostV(srcV),
    deviceV(copyToDevice(srcV)){
      
  }

  constHybridVector(){

  }

  __forceinline__ __host__ __device__ T at(int i) const{
#ifdef __CUDA_ARCH__
    return deviceV[i];
#else
    return hostV.at(i);
#endif
  }

  __forceinline__ __host__ __device__ const T operator[] (int i) const {
#ifdef __CUDA_ARCH__
    return deviceV[i];
#else
    return hostV[i];
#endif
  }

  // Assignment operator not needed because vector should be constant
  /* __host__ constHybridVector& operator= (const std::vector<T> &otherV){ */
  /*   deviceV = copyToDevice(otherV); */
  /*   hostV   = otherV; */
  /*   return *this; */
  /* } */

  __host__  operator T*(){
    return &(hostV.at(0));
  }

  __host__ T* toArray() const{
    std::vector<T> copyV = hostV;
    return &(copyV.at(0));
  }

  __host__ std::vector<T> toVector() const{
    return hostV;
  }
  
private:
  T *deviceV;
  std::vector<T> hostV;

};


/** 
 * @brief Queries for devices on the running mashine and collects
 *        them on the devices array. Set the first device in this 
 *        array as computation-device. On Errors the programm will
 *        be stoped by exit(). 
 * 
 * @param maxGpus max. devices which should be allocated
 * @return vector of possible devices
 */
std::vector<unsigned> getFreeDevices(unsigned maxGpus);

