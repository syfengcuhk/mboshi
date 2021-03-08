#!/bin/bash

# Siyuan Feng (2020): use a pretrained multilingual phone-level ASR model to generate label supervision for Mboshi
# *_own_gmm means gmm model used here comes from Mboshi GMM-HMM, rather than Discophone GMM-HMM
segment_subsampling_flag=1 # -1~1 means doing nothing, 2: if == 2, segment represented [1st-half-mean 2nd-half-mean], if == 3, [1st-third-mean, 2nd-third-mean, 3rd-third-mean], if == -3 [whole-mean, 1st-half-mean, 2nd-half-mean]; if > 3 then it's generalization to == 2 and ==3
max_vote=false
max_vote_threshold=0
post_model=final
nclusters=50
clustering_algo="kmeans"
rand_state=0
bin_output=false
bn_dim=40
LMTYPE=phn_ug_multi
model_name=final
discophone_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/discophone/v1_multilang_phones_lms #v1_phones_lms
align_fmllr_lats_retry_beam=10
align_fmllr_lats_beam=40
remove_egs=false
stop_stage=19
stage=17
align_fmllr_lats_stage=-10
nj=1
nj_clustering=2
num_threads_decode=1
decode_nj=4
train_set=full # or train
lat_generator_acwt=10.0
gmm=tri5
num_threads_ubm=32
nnet3_affix=
train_stage=-10
train_exit_stage=0
get_egs_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=1g
babel_langs="" #"307 103 101 402 107 206 404 203"
babel_recog="${babel_langs}"
gp_langs="Czech French Mandarin Spanish Thai"
gp_recog="${gp_langs}"
num_jobs_initial=1
num_jobs_final=1
num_epochs=5
frames_per_eg=150,120,90,75
initial_effective_lrate=0.001
final_effective_lrate=0.0001
minibatch_size=128
dropout_schedule='0,0@0.20,0.3@0.50,0'
max_param_change=2.0
gpu_extract_post=true
use_ivector=true
echo "$0 $@"  # Print the command line for logging
lang_name_suffix="_GP"
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
for l in  ${babel_langs}; do
  list_train_set="$l/data/train_${l} ${list_train_set}"
  list_dev_set="$l/data/dev_${l} ${dev_set}"
done
for l in ${gp_langs}; do
  list_train_set="GlobalPhone/gp_${l}_train ${list_train_set}"
  list_dev_set="GlobalPhone/gp_${l}_dev ${dev_set}"
done
list_train_set=${list_train_set%% }
list_dev_set=${dev_set%% }
function langname() {
  # Utility
  echo "$(basename "$1")"
}
lang_name=universal
if $use_ivector; then
  dir=exp/chain_discophone_lat_label/${lang_name}${lang_name_suffix}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tdnn${tdnn_affix}${data_aug_suffix}_${train_set}_own_gmm #_epochs$num_epochs
else
  dir=exp/chain_discophone_lat_label/${lang_name}${lang_name_suffix}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tdnn${tdnn_affix}_noivec${data_aug_suffix}_${train_set}_own_gmm
fi
train_ivector_dir=exp/chain_discophone_lat_label/nnet3/ivectors_${lang_name}${lang_name_suffix}_mboshi_hires/$train_set
gmm_dir=exp/gmm_discophone_label_collapse_ali/${lang_name}${lang_name_suffix}_${LMTYPE}/$train_set/$gmm #$discophone_root/exp/gmm/$lang_name/$gmm
ali_dir=${gmm_dir}_ali #exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/${train_set}_ali
tree_dir=exp/chain_discophone_lat_label/${lang_name}${lang_name_suffix}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tree_bi${tree_affix}_own_gmm
lat_dir=exp/chain_discophone_lat_label/${lang_name}${lang_name_suffix}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/${train_set}_lats_own_gmm
input_feat_affix=mfcc_hires_pitch
train_data_dir=data_hires_pitch_conf_discophone/${train_set}
lores_train_data_dir=data_plus_discophone_transcripts_collapse_ali/${lang_name}${lang_name_suffix}_${LMTYPE}/acwt${lat_generator_acwt}/${train_set}
common_egs_dir=exp/chain_discophone_lat_label/${lang_name}${lang_name_suffix}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tdnn1g_full_own_gmm/egs #exp/chain_discophone_lat_label/universal_phn_ug_multi/lat_gen_acwt10.0/tdnn1g_full_own_gmm
for f in  $train_data_dir/feats.scp $lores_train_data_dir/feats.scp ; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 15 ] && [  $stop_stage -gt 15 ]  ; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --stage $align_fmllr_lats_stage --nj $nj --cmd "$train_cmd" --beam $align_fmllr_lats_beam --retry-beam $align_fmllr_lats_retry_beam ${lores_train_data_dir}\
    $discophone_root/data/lang_combined_test $gmm_dir $lat_dir
  #rm $lat_dir/fsts.*.gz # save space
fi 

if [ $stage -le 16 ] && [ $stop_stage -gt 16 ] ; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} $discophone_root/data/lang_chain/lang_$lang_name $ali_dir $tree_dir
fi
xent_regularize=0.1
if [ $stage -le 17 ] && [ $stop_stage -gt 17 ]  ; then
  if $use_ivector; then
    input_line=",ReplaceIndex(ivector, t, 0)"
  else
    input_line=""
  fi
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  feat_dim=$(feat-to-dim scp:${train_data_dir}/feats.scp -)
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true" #"l2-regularize=0.002"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.0005" 
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=$feat_dim name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1${input_line}) affine-transform-file=$dir/configs/lda.mat
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=512
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=$bn_dim $linear_opts
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=512 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=512 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi


if [ $stage -le 18 ]  && [ $stop_stage -gt 18 ] ; then
  echo "start Chain LF-MMI AM training"

  steps/nnet3/chain/train.py --stage $train_stage --exit-stage $train_exit_stage \
    --cmd "run.pl" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 4 \
    --feat-dir ${train_data_dir} \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi
if [ !  "$post_model" = "final" ]; then
  post_suffix="_${post_model}"
else
  post_suffix=""
fi
if [ $stage -le 19 ]  && [ $stop_stage -gt 19 ] ; then
  #To do: generate BNF from prefinal-l
  bnf_name="prefinal-l"
  output_data=$dir/bnf_prefinal_l$post_suffix/$train_set/
  input_data=$train_data_dir
  ivector_dir=$train_ivector_dir
  steps/nnet3/make_bottleneck_features.sh --model-name $post_model --use-gpu $gpu_extract_post \
  --cmd run.pl --nj $nj --ivector-dir $ivector_dir $bnf_name $input_data $output_data $dir  || exit 1
  if ! $bin_output ; then
    copy-feats scp:$output_data/feats.scp ark,t:$output_data/feats.ark.txt
  fi
fi

if [ "$clustering_algo"   == "kmeans"  ]; then
  clustering_suffix="_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "spectral"  ]; then
  clustering_suffix="spec_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "agglomerative"  ]; then
  clustering_suffix="agglomerative_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "dbscan"  ]; then
  clustering_suffix="dbscan_clusters${nclusters}_rand_${rand_state}"
else
    echo "non supported  clustering algo"
fi
if [ $stage -le 20 ]  && [ $stop_stage -gt 20 ] ; then
  # k-meams clustering on bnf features
  if [ "$clustering_algo"   == "kmeans"  ]; then
    echo "K-means on BNF features"
  elif [ "$clustering_algo"   == "spectral"  ]; then
    echo "Spectral clustering on BNF features"
  elif [ "$clustering_algo"   == "dbscan" ] ; then
    echo "DBSCAN on BNF"
  elif [ "$clustering_algo"   == "hdbscan" ]; then
    echo "Hierarchical DBSCAN on bnf"
  elif [ "$clustering_algo" == "agglomerative" ]; then
    echo "Agglomerative Clustering"
  else
    echo "un-supported clustering algo"
    exit 0
  fi
  input_data=$dir/bnf_prefinal_l$post_suffix/$train_set/feats.ark.txt
  source activate beer # hdbscan package installed under (beer)
  python scripts_hpc/run/clustering_kaldi_input.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo
  source deactivate
  if $max_vote; then
    ref_phone_boundary=exp/gmm_discophone_label_collapse_ali/universal_phn_ug_multi/full/tri5_phone_ali_retry_beam1280/phone_ali.ali
    python scripts_hpc/run/max_vote.py $dir/bnf_prefinal_l${post_suffix}/$train_set/output_feats.ark.txt${clustering_suffix} $ref_phone_boundary $max_vote_threshold
  fi
fi

if [ $stage -le 21 ]  && [ $stop_stage -gt 21 ] ; then
  echo "evaluation AUD on k-means + bnf"
  eval_set=full
  if [ ! $eval_set = $train_set ]; then
    echo "$0: only full set evaluation supported "
    exit 1;
  fi
  output_dir=$dir/bnf_prefinal_l$post_suffix/$train_set/ #$dir/phone_post/$train_set  #phone_post_1hot_sym.txt
  #output_dir=$expdir_root/tri5_phone_ali${suffix}
  hyp_trans=$output_dir/output_feats.ark.txt${clustering_suffix} #$output_dir/phone_post_1hot_sym.txt
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
  source activate beer
  cwd=$(pwd)
  cd $eval_code_root
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud${clustering_suffix}${max_vote_suffix}
  source deactivate
  cd $cwd
fi


# below considers segment clustering
if [ "$clustering_algo"   == "kmeans"  ]; then
  clustering_suffix="_seg_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "spectral"  ]; then
  clustering_suffix="spectral_seg_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "dbscan"  ]; then
  clustering_suffix="dbscan_seg_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "agglomerative"  ]; then
  clustering_suffix="agglomerative_seg_clusters${nclusters}_rand_${rand_state}"
elif [ "$clustering_algo"   == "hdbscan"  ]; then
  clustering_suffix="hdbscan_seg_clusters${nclusters}_rand_${rand_state}"
else
    echo "non supported  clustering algo"
fi
if [ "$segment_subsampling_flag" = "2" ] || [ "$segment_subsampling_flag" = "3" ] || [ "$segment_subsampling_flag" = "-3" ] || [ $segment_subsampling_flag -gt 3 ] ; then
  clustering_suffix="_subsamp${segment_subsampling_flag}"${clustering_suffix} 
fi

if [ $stage -le 22 ] && [ $stop_stage -gt 22 ]; then
  echo "segment clustering"
  # if feats.ark.txt not found, generate it
  output_data=$dir/bnf_prefinal_l$post_suffix/$train_set/
  if [ ! -f $output_data/feats.ark.txt ] ; then
    copy-feats scp:$output_data/feats.scp ark,t:$output_data/feats.ark.txt
  fi
  input_data=$dir/bnf_prefinal_l$post_suffix/$train_set/feats.ark.txt
  ref_segmentation=$ali_dir/../tri5_phone_ali_linked/phone_ali.ali #exp/gmm_discophone_label_collapse_ali/universal_phn_ug_multi/full/tri5_phone_ali_retry_beam1280/phone_ali.ali
  source activate beer # hdbscan package installed under (beer)
  python scripts_hpc/run/clustering_kaldi_segment_input.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo $ref_segmentation $segment_subsampling_flag
  source deactivate
fi
if [ $stage -le 23 ]  && [ $stop_stage -gt 23 ] ; then
  echo "evaluation AUD on k-means + bnf, segment clustering"
  eval_set=full
  if [ ! $eval_set = $train_set ]; then
    echo "$0: only full set evaluation supported "
    exit 1;
  fi
  output_dir=$dir/bnf_prefinal_l$post_suffix/$train_set/ #$dir/phone_post/$train_set  #phone_post_1hot_sym.txt
  #output_dir=$expdir_root/tri5_phone_ali${suffix}
  hyp_trans=$output_dir/output_feats.ark.txt${clustering_suffix} #$output_dir/phone_post_1hot_sym.txt
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
  source activate beer
  cwd=$(pwd)
  cd $eval_code_root
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud${clustering_suffix}${max_vote_suffix}
  source deactivate
  cd $cwd
fi

if [ $stage -le 24 ]  && [ $stop_stage -gt 24 ] ; then
  echo "$0: create ${train_data_dir}_tmp/  that includes text from $lores_train_data_dir/ (from discophone ASR)"
#  mkdir -p ${train_data_dir}_tmp/.bkp || exit 1
#  mv ${train_data_dir}_tmp/* ${train_data_dir}_tmp/.bkp
  if [ -d ${train_data_dir}_tmp ]; then
    rm -rf ${train_data_dir}_tmp
  fi 
  utils/copy_data_dir.sh $train_data_dir ${train_data_dir}_tmp/
  cp $lores_train_data_dir/text ${train_data_dir}_tmp/text
#  if [ ! -f $train_data_dir/text ]; then
#    echo "no text found in $train_data_dir, so copying $lores_train_data_dir/text to it"
#    cp $lores_train_data_dir/text $train_data_dir/text 
#  fi

fi

if [ $stage -le 25 ]  && [ $stop_stage -gt 25 ] ; then
  # Create phone posterior
  node_name="output-xent.log-softmax"
  if [ !  "$post_model" = "final" ]; then
    post_suffix="_${post_model}"
  else
    post_suffix=""
  fi
  output_data=$dir/phone_post${post_suffix}/$train_set
  input_data=$train_data_dir
  ivector_dir=$train_ivector_dir
  tacc_file=${dir}_ali/$train_set/final.tacc
  if [ ! -f $tacc_file ]; then
    echo "Stage 18: no final.tacc found, so generate it first, to ${dir}_ali/$train_set/final.tacc"
#   (NOT appropriate as it's GMM-HMM final.mdl)    ali-to-post "ark:gunzip -c $ali_dir/ali.*.gz|" ark:- | post-to-tacc $ali_dir/final.mdl ark:- $ali_dir/final.tacc || exit 1;
    if [ ! -f ${dir}_ali/$train_set ]; then
      steps/nnet3/align.sh --nj $nj --use-gpu false \
        --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
        --online-ivector-dir $ivector_dir \
        ${input_data}_tmp $discophone_root/data/lang_chain/lang_$lang_name $dir ${dir}_ali/$train_set || exit 1
    fi
    ali-to-post "ark:gunzip -c ${dir}_ali/$train_set/ali.*.gz|" ark:- | post-to-tacc ${dir}_ali/$train_set/final.mdl ark:- ${dir}_ali/$train_set/final.tacc || exit 1  
    echo "final.tacc generated"
  fi
  steps/nnet3/make_bottleneck_features.sh --model-name $post_model --use-gpu $gpu_extract_post \
    --cmd run.pl --nj $nj --ivector-dir $ivector_dir $node_name $input_data $output_data $dir  || exit 1 
  logprob-to-post scp:$output_data/feats.scp ark:- | post-to-phone-post --transition-id-counts=${dir}_ali/$train_set/final.tacc $dir/final.mdl ark:- ark,t:$output_data/phone_post.txt
  python scripts_hpc/run/phone_post_to_1hot.py $output_data/phone_post.txt $output_data/phone_post_1hot.txt
  utils/int2sym.pl  -f 2- $dir/phones.txt $output_data/phone_post_1hot.txt > $output_data/phone_post_1hot_sym.txt
fi


if [ $stage -le 26 ] && [ $stop_stage -gt 26 ]; then
  echo "$0: Evaluating AUD using TDNNF AM phone posterior 1hot"
  eval_set=full
  if [ ! $eval_set = $train_set ]; then
    echo "$0: only full set evaluation supported "
    exit 1;
  fi
  output_dir=$dir/phone_post/$train_set  #phone_post_1hot_sym.txt
  #output_dir=$expdir_root/tri5_phone_ali${suffix}
  hyp_trans=$output_dir/phone_post_1hot_sym.txt
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
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud
  source deactivate
  cd $cwd
  
fi
 # below we show that using a golden phone segmentation boundary information, performance of our segment clustering

if [ $stage -le 27 ] && [ $stop_stage -gt 27 ]; then
  echo "segment clustering using golden phone segmentation boundary information"
  # if feats.ark.txt not found, generate it
  output_data=$dir/bnf_prefinal_l$post_suffix/$train_set/
  if [ ! -f $output_data/feats.ark.txt ] ; then
    copy-feats scp:$output_data/feats.scp ark,t:$output_data/feats.ark.txt
  fi
  current_path=$(pwd)
  cd $dir/bnf_prefinal_l$post_suffix/$train_set/
  echo "$pwd: creating symbolic link feats.ark.txt.gold_ali"
  ln -s feats.ark.txt feats.ark.txt.gold_ali
  cd $current_path 
  input_data=$dir/bnf_prefinal_l$post_suffix/$train_set/feats.ark.txt.gold_ali
  eval_code_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/BEER/beer/recipes/hshmm/
  eval_set=full
  mboshi_ref_ali_unsorted=$eval_code_root/data/mboshi/$eval_set/ali
  cp $mboshi_ref_ali_unsorted data/ali_${eval_set}_unsorted
  # Now we sort data/ali_${eval_set}_unsorted by using order data/$eval_set/utt2spk
  awk 'NR==FNR{o[FNR]=$1; next} {t[$1]=$0} END{for(x=1; x<=FNR; x++){y=o[x]; print t[y]}}' data/$eval_set/utt2spk data/ali_${eval_set}_unsorted > data/ali_${eval_set}
  mboshi_ref_ali=data/ali_${eval_set}
  ref_segmentation=$mboshi_ref_ali #$ali_dir/../tri5_phone_ali_linked/phone_ali.ali
  source activate beer # hdbscan package installed under (beer)
  python scripts_hpc/run/clustering_kaldi_segment_input.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo $ref_segmentation $segment_subsampling_flag
  source deactivate
fi
if [ $stage -le 28 ]  && [ $stop_stage -gt 28 ] ; then
  echo "evaluation AUD on k-means + bnf, segment clustering"
  eval_set=full
  if [ ! $eval_set = $train_set ]; then
    echo "$0: only full set evaluation supported "
    exit 1;
  fi
  output_dir=$dir/bnf_prefinal_l$post_suffix/$train_set/ #$dir/phone_post/$train_set  #phone_post_1hot_sym.txt
  #output_dir=$expdir_root/tri5_phone_ali${suffix}
  hyp_trans=$output_dir/output_feats.ark.txt.gold_ali${clustering_suffix} #$output_dir/phone_post_1hot_sym.txt
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
  source activate beer
  cwd=$(pwd)
  cd $eval_code_root
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud_gold_ali${clustering_suffix}${max_vote_suffix}
  source deactivate
  cd $cwd
fi

echo "succeeded"
