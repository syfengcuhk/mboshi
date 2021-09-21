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
#SBATCH --mem=28G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
##SBATCH --gres=gpu:2
#SBATCH --mail-type=END

# stage -1: mfcc; stage 0: hires mfcc feature; stage 1: ivector, 2: decoding, 3: get best path; 4: get data_plus_cgn_transcripts/ dirs and generate alignments; 5: generate phone-level alignments to either regular phone alignments (default) or ctm-style one; 6: position-dependent phone to postion-indpendent conversion 7: use Ondel's AUD evaluation code to evaluate NMI, F-score etc
#srun bash  scripts_hpc/run/cgn_labeling.sh --stage 5 --stop-stage 6 --nj 2 --ctm-format "false"


#######Aidatatang
#srun bash  scripts_hpc/run/aidatatang_labeling.sh --stage 2 --stop-stage 3 --decode-nj 2 --nj 2 --ctm-format "false" 


##### Discophone monolingual ASR
## stage 2: decoding; 3: best path; 4: data_plus* generation and alignment; 5: align to phone align from align_fmllr.sh output; 6: align to phone align from lattice-best-path output; 8: AUD evaluation  
gp_langs="Thai"
babel_langs=""
#decode_beam=60 #30.0 # default 15.0, change does not lead to big difference
#lattice_beam=32 #16.0 # default 8.0, change does not lead to big difference
acwt=10.0
#max_active=7000
eval_from_lattice=true
#srun bash scripts_hpc/run/disco_mono_labeling.sh --gpu-decode true --stage 5 --stop-stage 6 --nj 2 --decode-nj 2 --num-threads-decode 6 --align-fmllr-beam 100 --align-fmllr-retry-beam 640 --gp-langs "$gp_langs" --babel-langs "$babel_langs" --acwt $acwt --eval-from-lattice $eval_from_lattice # --max-active $max_active --decode-beam $decode_beam --lattice-beam $lattice_beam
##srun bash scripts_hpc/run/disco_mono_labeling.sh --stage 6 --stop-stage 9 --nj 2 --decode-nj 2 --num-threads-decode 6 --align-fmllr-beam 25 --align-fmllr-retry-beam 160 --gp-langs "$gp_langs" --babel-langs "$babel_langs" --acwt $acwt
#
#
##srun bash scripts_hpc/run/disco_mono_labeling.sh --stage 2 --stop-stage 4 --LMTYPE "phn_bg_mono" --decode-nj 2
##srun bash scripts_hpc/run/disco_mono_labeling.sh --stage 2 --stop-stage 4 --LMTYPE "words" --decode-nj 2 --num-threads-decode 4
######GMM HMM training of Mboshi based on discophone monolingual ASR transcript###
#srun bash scripts_hpc/run/disco_mono_gmmhmm.sh --stage 13 --stop-stage 14 --train-nj 2 --gp-langs "$gp_langs" --babel-langs "$babel_langs" --retry-beam 160
######TDNNF training of Mboshi based on discophone monolingual ASR transcript##and phone post generation#
#srun bash scripts_hpc/run/disco_mono_tdnnf_1g.sh --stage 18 --stop-stage 19  --gp-langs "Czech" --babel-langs "" --nj 2 --align-fmllr-lats-beam 50 --align-fmllr-lats-retry-beam 320 --train-stage 0 --train-exit-stage 10000
#srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --gp-langs "$gp_langs" --babel-langs "$babel_langs" --stage 20 --stop-stage 21 --nj 2 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --train-stage -10 --train-exit-stage 10000 --post-model "16" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans"
#  Tune learning rate see if Mboshi TDNNF w/ mono disco labels can improve
#srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --gp-langs "$gp_langs" --babel-langs "$babel_langs" --stage 20 --stop-stage 21 --nj 2 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --train-stage -10 --train-exit-stage 10000 --post-model "12" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" --common-egs-dir "exp/chain_discophone_lat_label/gp_${gp_langs}_train_phn_ug_mono/lat_gen_acwt10.0/tdnn1g_full_own_gmm_ep5/egs" --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 
#srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --gp-langs "$gp_langs" --babel-langs "$babel_langs" --stage 20 --stop-stage 21 --nj 2 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --train-stage -10 --train-exit-stage 10000 --post-model "24" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" --common-egs-dir "exp/chain_discophone_lat_label/gp_${gp_langs}_train_phn_ug_mono/lat_gen_acwt10.0/tdnn1g_full_own_gmm_ep5/egs" --initial-effective-lrate 0.0005 --final-effective-lrate 0.00005 --tdnn-affix "1g_lr5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 

###Try build a tree with no frame subsampling ##
#srun bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --stage 20 --stop-stage 21 --tree-affix "_no_subsampling" --tdnn-affix "1g_no_subsampling" --frame-subsampling-factor 1 --train-stage 0 --train-exit-stage 20 --post-model "final" --nj-clustering 16 --rand-state 0 --nclusters 50 

# Segment clustering
#srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --gp-langs "Czech" --babel-langs "$babel_langs" --stage 22 --stop-stage 23 --post-model "16" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20
#srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --gp-langs "French" --babel-langs "$babel_langs" --stage 22 --stop-stage 23 --post-model "12" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20
#srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --gp-langs "Spanish" --babel-langs "$babel_langs" --stage 22 --stop-stage 23 --post-model "12" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20
#srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --gp-langs "Mandarin" --babel-langs "$babel_langs" --stage 22 --stop-stage 23 --post-model "12" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20
#srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --gp-langs "Thai" --babel-langs "$babel_langs" --stage 22 --stop-stage 24 --post-model "12" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20

# TDNNF with no ivectors as input
gp_langs="Czech"
babel_langs=""
#for rstate in 1 2 3 4; do
#  srun  bash scripts_hpc/run/disco_mono_tdnnf_1g_own_gmm.sh --use-ivector false --gp-langs "${gp_langs}" --babel-langs "$babel_langs" --stage 22 --stop-stage 24 --nj 2 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --train-stage -10 --train-exit-stage 10000 --post-model "12" --nj-clustering 16 --rand-state $rstate --nclusters 50 --clustering-algo "kmeans" --common-egs-dir "exp/chain_discophone_lat_label/gp_${gp_langs}_train_phn_ug_mono/lat_gen_acwt10.0/tdnn1g_full_own_gmm_ep5/egs" --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2
#done
#### CE trained TDNN_1a model, not LF-MMI
#srun bash scripts_hpc/run/disco_mono_tdnn_1a_CE_own_gmm.sh --stage 20 --stop-stage 21 --train-stage 0 --train-exit-stage 10000 --num-jobs-initial 2 --num-jobs-final 2 --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans"

##### Discophone multilingual ASR, requiring a large beam and retry_beam
acwt=10.0
retry_beam=2048
beam=512
eval_from_lattice=true
#srun bash scripts_hpc/run/disco_multi_labeling.sh --gpu-decode true --stage 11 --stop-stage 12 --nj 2 --decode-nj 2 --num-threads-decode 8 --acwt $acwt --align-fmllr-beam $beam --align-fmllr-retry-beam  $retry_beam --eval-from-lattice $eval_from_lattice --lang-name-suffix "" --decode-fmllr-stage 2
#GMM HMM training of Mboshi based on discophone multilingual ASR transcript###
#srun bash scripts_hpc/run/disco_multi_gmmhmm.sh --stage 7  --stop-stage 14 --train-nj 2
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --stage 25 --stop-stage 26 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage 0 --train-exit-stage 1000 

# Below works on TDNNF training using 13-/5-lang multilingual disco ASR's labels.
#  --lang-name-suffix by default equals "_GP" which denotes 5-GP-lang multilingual discop ASR
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "" --stage 19 --stop-stage 21 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 --post-model "32" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "" --stage 20 --stop-stage 21 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --post-model "28" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "_GP" --stage 20 --stop-stage 21 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --post-model "28" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "_GP" --stage 20 --stop-stage 21 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.0005 --final-effective-lrate 0.00005 --tdnn-affix "1g_lr5e-4" --num-epochs 20 --post-model "12" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "" --stage 20 --stop-stage 21 --post-model "final" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans"

# Segment clustering
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "_GP" --stage 22 --stop-stage 24 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.0005 --final-effective-lrate 0.00005 --tdnn-affix "1g_lr5e-4" --num-epochs 20 --post-model "16" --nj-clustering 16 --rand-state 4 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "" --stage 22 --stop-stage 23 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --post-model "28" --nj-clustering 10 --rand-state 0 --nclusters 70 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "" --stage 22 --stop-stage 23 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --post-model "28" --nj-clustering 12 --rand-state 4 --nclusters 50 --clustering-algo "kmeans" --segment-subsampling-flag 4 #-3 #2 # 2 means first and 2nd half mean concatenated together, -3 means [mean,1st-half-mean,2nd-half-mean] 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "_GP" --stage 22 --stop-stage 24 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.0005 --final-effective-lrate 0.00005 --tdnn-affix "1g_lr5e-4" --num-epochs 20 --post-model "16" --nj-clustering 16 --rand-state 1 --nclusters 50 --clustering-algo "kmeans" --segment-subsampling-flag 3 

# below gives try to non-kmeans clustering 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector false --lang-name-suffix "" --stage 22 --stop-stage 24 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 --post-model "28" --nj-clustering 2 --rand-state 0 --nclusters 50 --clustering-algo "agglomerative" # memory > 160GB
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "" --stage 20 --stop-stage 21 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --post-model "28" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "hdbscan" 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --lang-name-suffix "" --stage 22 --stop-stage 23 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --post-model "28" --nj-clustering 1 --rand-state 0 --nclusters 50 --clustering-algo "hdbscan" # spectral has memory issue unsolved 

# Below gives a try if ivector is not used in TDNNF training, and adjustment in TDNNF config
# Frame clustering
#bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector false --lang-name-suffix "" --stage 20 --stop-stage 22 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 --post-model "28" --nj-clustering 16 --rand-state 4 --nclusters 50 --clustering-algo "kmeans"
# Segment level clustering
subsamp_flag=5
nc=30
#for rstate in 0 1 2 3 4; do
#  srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector false --lang-name-suffix "" --stage 22 --stop-stage 24 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 --post-model "28" --nj-clustering 12 --rand-state $rstate --nclusters $nc --clustering-algo "kmeans" #--segment-subsampling-flag $subsamp_flag 
#done
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector false --lang-name-suffix "_GP" --stage 19 --stop-stage 20 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.0005 --final-effective-lrate 0.00005 --tdnn-affix "1g_lr5e-4" --num-epochs 20 --post-model "16" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector false --lang-name-suffix "_GP" --stage 22 --stop-stage 24 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000   --num-jobs-initial 2 --num-jobs-final 2 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --post-model "24" --nj-clustering 16 --rand-state 4  --nclusters 50 --clustering-algo "kmeans" 

# Use golden phone segmentation boundary information to perform segment clustering
#for rstate in 0 1 2 3 4; do
#  srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector false --lang-name-suffix "" --stage 27 --stop-stage 29 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 --post-model "28" --nj-clustering 12 --rand-state $rstate --nclusters 50 --clustering-algo "kmeans" --segment-subsampling-flag 5
#done

# evaluated as phoneme NMI not the default, phone NMI
n_clusters=70
for rstate in 0 1 2 3 4; do
  srun bash scripts_hpc/run/disco_multi_tdnnf_1g_own_gmm.sh --use-ivector false --lang-name-suffix "" --stage 27 --stop-stage 29 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --nj 2 --train-stage -10 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 20 --num-jobs-initial 2 --num-jobs-final 2 --post-model "28" --nj-clustering 12 --rand-state $rstate --nclusters $n_clusters --clustering-algo "kmeans" --do-phoneme-discovery true #--segment-subsampling-flag 5
done


# Speed perturbed data to train TDNNF
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_sp_own_gmm.sh --use-ivector false --lang-name-suffix "" --stage 22 --stop-stage 24  --nj 6 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --train-stage 0 --train-exit-stage 10000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 10 --num-jobs-initial 2 --num-jobs-final 2 --post-model "28" --nj-clustering 12 --rand-state 4 --nclusters 50 --clustering-algo "kmeans"
#srun bash scripts_hpc/run/disco_multi_tdnnf_1g_sp_own_gmm.sh --use-ivector false --lang-name-suffix "_GP" --stage 22 --stop-stage 24 --nj 6 --align-fmllr-lats-beam 10 --align-fmllr-lats-retry-beam 40 --train-stage 0 --train-exit-stage 1000 --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4" --num-epochs 10 --num-jobs-initial 2 --num-jobs-final 2 --post-model "32" --nj-clustering 12 --rand-state 4 --nclusters 50 --clustering-algo "kmeans"

##### Discophone monolingual ASR but merging different versions of text transcripts of a single mboshi utterance into one dir. 
#srun bash scripts_hpc/run/disco_mono_merge_gmmhmm.sh --stage 8 --stop-stage 14 --train-nj 10
#srun bash scripts_hpc/run/disco_mono_merge_tdnnf_1g_own_gmm.sh --stage 19 --stop-stage 21 --nj 10 --train-stage 0 --train-exit-stage 10000 --num-jobs-initial 2 --num-jobs-final 2 --post-model "40" --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_mono_merge_tdnnf_1g_own_gmm.sh --stage 19 --stop-stage 20 --nj 10 --train-stage -10 --train-exit-stage 10000 --num-jobs-initial 3 --num-jobs-final 3 --post-model "80" --num-epochs 10 --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_mono_merge_tdnnf_1g_own_gmm.sh --stage 20 --stop-stage 21 --nj 10 --train-stage -10 --train-exit-stage 10000 --num-jobs-initial 3 --num-jobs-final 3 --post-model "final" --num-epochs 15 --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" 
#srun bash scripts_hpc/run/disco_mono_merge_tdnnf_1g_own_gmm.sh --stage 20 --stop-stage 21 --nj 10 --train-stage -10 --train-exit-stage 10000 --num-jobs-initial 3 --num-jobs-final 3 --post-model "32" --num-epochs 15 --nj-clustering 16 --rand-state 0 --nclusters 50 --clustering-algo "kmeans" --initial-effective-lrate 0.00025 --final-effective-lrate 0.000025 --tdnn-affix "1g_lr2.5e-4"
