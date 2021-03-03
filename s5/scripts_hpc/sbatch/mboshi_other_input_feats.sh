#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=00:30:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=16
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=30G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
##SBATCH --gres=gpu:2
#SBATCH --mail-type=END




# Discophone Multilingual ASR related (13-lang or GP 5-lang)
#srun bash scripts_hpc/run/apc_input/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector true --lang-name-suffix "" --stage 22 --stop-stage 24 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage 0 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 --post-model "28" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans"
srun bash scripts_hpc/run/apc_input/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector false --lang-name-suffix "" --stage 22 --stop-stage 24 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage 0 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 --post-model "28" --nj-clustering 16 --rand-state 4 --nclusters 50 --clustering-algo "kmeans"

