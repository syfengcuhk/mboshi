#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=00:50:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=16
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=30G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
##SBATCH --gres=gpu:3
#SBATCH --mail-type=END

# mainly baseline implementation
#srun bash scripts_hpc/run/baseline.sh --stage 20 --stop-stage 21 --nj-clustering 16 --rand-state 0 --nclusters 50 --cmvn-opt "_cmvn" #"" # or "_cmn" or "_cmvn"
#srun bash scripts_hpc/run/baseline.sh --stage 22 --stop-stage 24 --nj-clustering 16 --rand-state 0 --nclusters 50 --cmvn-opt "_cmn" --lang-name "French" --LM-TYPE "phn_ug_mono" # --ali-dir "exp/gmm_discophone_label/gp_Czech_train_phn_ug_mono/full/tri5_ali" #"" # or "_cmn" or "_cmvn", "_cmn" and "" better than "_cmvn" # change --lang-name and --LM-TYPE to change different segmentation reference
#srun bash scripts_hpc/run/baseline.sh --stage 22 --stop-stage 24 --nj-clustering 16 --rand-state 0 --nclusters 50 --cmvn-opt "_cmn" --lang-name "Spanish" --LM-TYPE "phn_ug_mono" # --ali-dir "exp/gmm_discophone_label/gp_Czech_train_phn_ug_mono/full/tri5_ali" #"" # or "_cmn" or "_cmvn", "_cmn" and "" better than "_cmvn" # change --lang-name and --LM-TYPE to change different segmentation reference
#srun bash scripts_hpc/run/baseline.sh --stage 22 --stop-stage 24 --nj-clustering 16 --rand-state 0 --nclusters 50 --cmvn-opt "_cmn" --lang-name "Mandarin" --LM-TYPE "phn_ug_mono" # --ali-dir "exp/gmm_discophone_label/gp_Czech_train_phn_ug_mono/full/tri5_ali" #"" # or "_cmn" or "_cmvn", "_cmn" and "" better than "_cmvn" # change --lang-name and --LM-TYPE to change different segmentation reference
#srun bash scripts_hpc/run/baseline.sh --stage 22 --stop-stage 24 --nj-clustering 16 --rand-state 0 --nclusters 50 --cmvn-opt "_cmn" --lang-name "Thai" --LM-TYPE "phn_ug_mono" # --ali-dir "exp/gmm_discophone_label/gp_Czech_train_phn_ug_mono/full/tri5_ali" #"" # or "_cmn" or "_cmvn", "_cmn" and "" better than "_cmvn" # change --lang-name and --LM-TYPE to change different segmentation reference
srun bash scripts_hpc/run/baseline.sh --stage 22 --stop-stage 24 --nj-clustering 16 --rand-state 0 --nclusters 50 --cmvn-opt "_cmn" --lang-name "universal" --lang-name-suffix "_GP" --LM-TYPE "phn_ug_multi" # --ali-dir "exp/gmm_discophone_label/gp_Czech_train_phn_ug_mono/full/tri5_ali" #"" # or "_cmn" or "_cmvn", "_cmn" and "" better than "_cmvn" # change --lang-name and --LM-TYPE to change different segmentation reference

