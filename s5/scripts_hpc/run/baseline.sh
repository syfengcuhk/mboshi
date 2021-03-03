#!/bin/bash

# Baseline1: MFCC features (w or w/o cmvn), directly used for k-means
# Baseline2: MFCC features (w or w/o cmvn), used for segment-level k-means
max_vote=false
clustering_algo="kmeans"
train_set=full
cmvn_opt="" # or "_cmn" or "_cmvn"
stage=20
stop_stage=21
nj_clustering=1 
rand_state=0
nclusters=50
clustering_lfr=
lang_name=Czech
lang_name_suffix="" # or "_GP"
LM_TYPE=phn_ug_mono
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh
if [ ! "$lang_name" = "universal" ]; then # monolingual
  ali_dir=exp/gmm_discophone_label/gp_${lang_name}_train_${LM_TYPE}/full/tri5_ali #/${lang_name}_${LMTYPE}/$train_set/
else # multilingual (either 13 lang or 5 GP lang)
  ali_dir=exp/gmm_discophone_label_collapse_ali/${lang_name}${lang_name_suffix}_${LM_TYPE}/full/tri5_ali
fi
train_data_dir=data_hires_pitch_conf_discophone/${train_set}
if [ $stage -le 19 ] && [ $stop_stage -gt 19 ] ; then
  # getting feats.ark.txt
  copy-feats scp:$train_data_dir/feats.scp ark,t:$train_data_dir/feats.ark.txt
  ln -s $train_data_dir/feats.ark.txt $train_data_dir/feats_no_cmvn.ark.txt
  copy-feats scp:$train_data_dir/feats.scp ark:- | apply-cmvn --skip-dims=40:41:42 --utt2spk=ark:$train_data_dir/utt2spk scp:$train_data_dir/cmvn.scp ark:- ark,t:$train_data_dir/feats_cmn.ark.txt # do not apply cmvn on the last 3 dims which are pitch-related features
  copy-feats scp:$train_data_dir/feats.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --skip-dims=40:41:42 --utt2spk=ark:$train_data_dir/utt2spk scp:$train_data_dir/cmvn.scp ark:- ark,t:$train_data_dir/feats_cmvn.ark.txt # do not apply cmvn on the last 3 dims which are pitch-related features
fi

if [ -z $cmvn_opt ]; then
  cmvn_opt="_no_cmvn"
fi

if [ "$clustering_algo"   == "kmeans"  ]; then
  clustering_suffix="_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "spectral"  ]; then
  clustering_suffix="spec_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "dbscan"  ]; then
  clustering_suffix="dbscan_clusters${nclusters}_rand_${rand_state}"
else
    echo "non supported  clustering algo"
fi
if [ $stage -le 20 ]  && [ $stop_stage -gt 20 ] ; then
  # k-meams clustering on mfcc features
  if [ "$clustering_algo"   == "kmeans"  ]; then
    echo "K-means on mfcc features"
  elif [ "$clustering_algo"   == "spectral"  ]; then
    echo "Spectral clustering on mfcc features"
  elif [ "$clustering_algo"   == "dbscan" ] ; then
    echo "DBSCAN on mfcc"
  else
    echo "un-supported clustering algo"
    exit 0
  fi
  input_data=$train_data_dir/feats${cmvn_opt}.ark.txt #$dir/bnf_prefinal_l$post_suffix/$train_set/feats.ark.txt
  source activate beer
  if [ -z $clustering_lfr ]; then
    python scripts_hpc/run/clustering_kaldi_input.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo
  else
    python scripts_hpc/run/clustering_kaldi_input_lfr.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo $clustering_lfr
  fi
# Output of clustering will be output_feats${cmvn_opt}.ark.txt${clustering_suffix}
  source deactivate
fi

if [ $stage -le 21 ]  && [ $stop_stage -gt 21 ] ; then
  echo "evaluation AUD on k-means + mfcc"
  eval_set=full
  if [ ! $eval_set = $train_set ]; then
    echo "$0: only full set evaluation supported "
    exit 1;
  fi
  output_dir=$train_data_dir #$dir/bnf_prefinal_l$post_suffix/$train_set/ #$dir/phone_post/$train_set  #phone_post_1hot_sym.txt
  #output_dir=$expdir_root/tri5_phone_ali${suffix}
  hyp_trans=$output_dir/output_feats${cmvn_opt}.ark.txt${clustering_suffix} #$output_dir/phone_post_1hot_sym.txt
  if [ ! -z $clustering_lfr ]; then
    hyp_trans=${hyp_trans}_lfr${clustering_lfr}
    score_suffix="_lfr${clustering_lfr}"
  fi
  if [ ! -f $hyp_trans ]; then
    echo "$0: $hyp_trans not found";
    exit 1;
  fi
  eval_code_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/BEER/beer/recipes/hshmm/
  mboshi_ref_ali=$eval_code_root/data/mboshi/$eval_set/ali
  len_hyp=$(wc -l $hyp_trans | cut -d ' ' -f 1 )
  len_ref=$(wc -l $mboshi_ref_ali | cut -d ' ' -f 1)
  if [ ! $len_hyp = $len_ref  ]; then
    echo "exit because not all utterances in $eval_set have hyp transcripts"
    exit 1;
  fi
  source activate beer
  cwd=$(pwd)
  cd $eval_code_root
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud${cmvn_opt}${clustering_suffix}${score_suffix}
  source deactivate
  cd $cwd
fi



# now goes to segment level mfcc clustering, and segmentation from (1) mono discophone GP (2) multi (3) multi GP

# below considers segment clustering
if [ "$clustering_algo"   == "kmeans"  ]; then
  clustering_suffix="_seg_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "spectral"  ]; then
  clustering_suffix="_seg_spec_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "dbscan"  ]; then
  clustering_suffix="_seg_dbscan_clusters${nclusters}_rand_${rand_state}"
else
    echo "non supported  clustering algo"
fi
if [ $stage -le 22 ]  && [ $stop_stage -gt 22 ] ; then
  ref_segmentation=$ali_dir/../tri5_phone_ali_linked/phone_ali.ali
  echo "Using reference segmentation from $ref_segmentation"
  input_data=$train_data_dir/feats${cmvn_opt}.ark.txt
  source activate beer # hdbscan package installed under (beer)
  python scripts_hpc/run/clustering_kaldi_segment_input.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo $ref_segmentation
  source deactivate
fi

if [ $stage -le 23 ] && [ $stop_stage -gt 23 ]; then
  echo "$0: evaluate segment level k-means on mfcc"

  eval_set=full
  if [ ! $eval_set = $train_set ]; then
    echo "$0: only full set evaluation supported "
    exit 1;
  fi
  output_dir=$train_data_dir #$dir/bnf_prefinal_l$post_suffix/$train_set/ #$dir/phone_post/$train_set  #phone_post_1hot_sym.txt
  #output_dir=$expdir_root/tri5_phone_ali${suffix}
  hyp_trans=$output_dir/output_feats${cmvn_opt}.ark.txt${clustering_suffix} #$output_dir/phone_post_1hot_sym.txt
  if $max_vote; then
    hyp_trans=${hyp_trans}_maxvote${max_vote_threshold}
    max_vote_suffix="_maxvote${max_vote_threshold}"
  fi
  if [ ! -f $hyp_trans ]; then
    echo "$0: $hyp_trans not found";
    exit 1;
  fi
  eval_code_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/BEER/beer/recipes/hshmm/
  mboshi_ref_ali=$eval_code_root/data/mboshi/$eval_set/ali
  len_hyp=$(wc -l $hyp_trans | cut -d ' ' -f 1 )
  len_ref=$(wc -l $mboshi_ref_ali | cut -d ' ' -f 1)
  if [ ! $len_hyp = $len_ref  ]; then
    echo "exit because not all utterances in $eval_set have hyp transcripts"
    exit 1;
  fi
  seg_suffix="_${lang_name}${lang_name_suffix}_${LM_TYPE}"
  source activate beer
  cwd=$(pwd)
  cd $eval_code_root
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud${cmvn_opt}${clustering_suffix}${seg_suffix}${max_vote_suffix}
  source deactivate
  cd $cwd
fi

