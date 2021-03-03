# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
#export train_cmd="queue.pl -l arch=*64"
#export decode_cmd="queue.pl -l arch=*64 --mem 2G"
#export mkgraph_cmd="queue.pl -l arch=*64 --mem 4G"
#export big_memory_cmd="queue.pl -l arch=*64 --mem 8G"
#export cuda_cmd="queue.pl -l gpu=1"

#b) run it locally...
export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl
export mkgraph_cmd=run.pl

#c) TU Delft cluster:  #BUT cluster:
#if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
#if [ "$(hostname -d)" == "hpc.tudelft.nl" ]; then
#export train_cmd=slurm.pl
#export decode_cmd=slurm.pl
#export cuda_cmd=slurm.pl
#export mkgraph_cmd=slurm.pl
#  queue="all.q@@blade,all.q@@speech"
#  gpu_queue="long.q@supergpu*,long.q@dellgpu*,long.q@pcspeech-gpu,long.q@pcgpu*"
#  storage="matylda5"
#  export train_cmd="queue.pl -q $queue -l ram_free=1500M,mem_free=1500M,${storage}=1"
#  export decode_cmd="queue.pl -q $queue -l ram_free=2500M,mem_free=2500M,${storage}=0.5"
#  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1" 
#fi

