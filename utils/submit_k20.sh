#!/bin/bash
#PBS -q k20
#PBS -l nodes=1:ppn=2
#PBS -l walltime=00:30:00
#PBS -N calcPhiASE

. /opt/modules-3.2.6/Modules/3.2.6/init/bash
export MODULES_NO_OUTPUT=1
module load ~/own.modules
export -n MODULES_NO_OUTPUT

uname -a

echo
cd ~/octrace

MAXGPUS="1"
#USE_REFLECTION="--reflection"
EXPECTATION="0.005"
RAYSPERSAMPLE="10000"
MAXRAYS="10000"

#WRITE_VTK="--write-vtk"
EXPERIMENT="testdata_2"
SILENT="--silent"
SAMPLE=$PBS_ARRAYID

MODE="ray_propagation_gpu"
#FIFO=$1
PIPE="/tmp/octrace_job_array_pipe"

FOLDER="$(pwd)"
echo "Executing..."
echo

echo 0 >> $PIPE

time ./bin/calcPhiASE --experiment="$FOLDER/utils/$EXPERIMENT" --mode=$MODE $SILENT --rays=$RAYSPERSAMPLE $WRITE_VTK $USE_REFLECTION --maxrays=$MAXRAYS --maxgpus=$MAXGPUS --sample_i=$SAMPLE

echo 1 >> $PIPE
