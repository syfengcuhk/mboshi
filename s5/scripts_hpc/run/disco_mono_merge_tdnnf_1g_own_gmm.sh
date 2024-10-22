#!/bin/bash
# Siyuan Feng (2020): use alignments generated from merge_transcripts to train Mboshi TDNNF. 5130 * 5 utterances, 1 waveform mapping to 5 versions of crosslingual phone labels.
frame_subsampling_factor=3
max_vote=true
max_vote_threshold=0
post_model=final
nclusters=50
clustering_algo="kmeans"
rand_state=0
bin_output=false
bn_dim=40
LMTYPE=phn_ug_mono
discophone_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/discophone/v1_phones_lms
align_fmllr_lats_retry_beam=40
align_fmllr_lats_beam=10
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
hires_conf_set=discophone
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=1g
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
common_egs_dir=exp/chain_discophone_lat_label/merge_transcript_GP_phn_ug_mono/lat_gen_acwt10.0/tdnn1g_full_own_gmm_ep5/egs
lang_name_suffix="_GP"
data_suffix="_collapse_ali"
echo "$0 $@"  # Print the command line for logging

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
lang_name=merge_transcript${lang_name_suffix}
dir=exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tdnn${tdnn_affix}${data_aug_suffix}_${train_set}_own_gmm_ep$num_epochs
train_ivector_dir=exp/chain_${hires_conf_set}_lat_label/nnet3/ivectors_${lang_name}${lang_name_suffix}_mboshi_hires/$train_set
gmm_dir=exp/gmm_discophone_label${data_suffix}/${lang_name}_${LMTYPE}/$train_set/$gmm
ali_dir=${gmm_dir}_ali
tree_dir=exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tree_bi${tree_affix}_own_gmm
lat_dir=exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/${train_set}_lats_own_gmm
input_feat_affix=mfcc_hires_pitch
train_data_dir=data_hires_pitch_conf_${hires_conf_set}/prefixed/${lang_name}_${LMTYPE}/acwt${lat_generator_acwt}/$train_set
lores_train_data_dir=data_plus_discophone_transcripts${data_suffix}/prefixed/${lang_name}_${LMTYPE}/acwt${lat_generator_acwt}/$train_set

# To do: Extract hires mfcc features to $train_data_dir
train_data_dir_no_pitch=data_hires_conf_${hires_conf_set}/prefixed/${lang_name}_${LMTYPE}/acwt${lat_generator_acwt}/$train_set/
if [ $stage -le 13 ] && [ $stop_stage -gt 13 ]; then
  utils/copy_data_dir.sh $lores_train_data_dir $train_data_dir || exit 1;
  steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires_from_${hires_conf_set}.conf --nj $nj $train_data_dir || exit 1;
  steps/compute_cmvn_stats.sh $train_data_dir
  if [ ! -f $train_data_dir_no_pitch/feats.scp ]; then
    utils/data/limit_feature_dim.sh 0:39 $train_data_dir $train_data_dir_no_pitch
    steps/compute_cmvn_stats.sh  $train_data_dir_no_pitch 
  fi
fi

for f in  $train_data_dir/feats.scp $lores_train_data_dir/feats.scp ; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done
# Extract ivectors 
if [ $stage -le 14 ] && [ $stop_stage -gt 14 ]; then
  input_data=$train_data_dir_no_pitch
  ivector_extractor=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/discophone/v1${lang_name_suffix}_multilang_phones_lms/exp/nnet3${nnet3_affix}/universal/extractor
  output_data=$train_ivector_dir 
  steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
    $input_data $ivector_extractor $output_data || exit 1;

fi

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
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} /tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/discophone/v1${lang_name_suffix}_multilang_phones_lms/data/lang_chain/lang_universal $ali_dir $tree_dir
fi
xent_regularize=0.1
if [ $stage -le 17 ] && [ $stop_stage -gt 17 ]  ; then
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
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
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
    --chain.frame-subsampling-factor $frame_subsampling_factor \
    --chain.alignment-subsampling-factor $frame_subsampling_factor \
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
  #To do: generate BNF from prefinal-l, by default 40 dimension
  bnf_name="prefinal-l"
  output_data=$dir/bnf_prefinal_l$post_suffix/$train_set/
  input_data=$train_data_dir
  ivector_dir=$train_ivector_dir
  steps/nnet3/make_bottleneck_features.sh --model-name $post_model --use-gpu $gpu_extract_post \
    --cmd run.pl --nj $nj --ivector-dir $ivector_dir $bnf_name $input_data $output_data $dir  || exit 1
  if ! $bin_output ; then
    grep -o 'Czech-.*' $output_data/feats.scp > $output_data/feats_uniq.scp
    copy-feats scp:$output_data/feats_uniq.scp ark,t:$output_data/feats.ark.txt
  fi
fi

if [ $stage -le 20 ]  && [ $stop_stage -gt 20 ] ; then
  # k-meams clustering on bnf features
  if [ "$clustering_algo"   == "kmeans"  ]; then
    echo "K-means on BNF features"
  elif [ "$clustering_algo"   == "spectral"  ]; then
    echo "Spectral clustering on BNF features"
  elif [ "$clustering_algo"   == "dbscan" ] ; then
    echo "DBSCAN on BNF"
  else
    echo "un-supported clustering algo"
    exit 0
  fi
  input_data=$dir/bnf_prefinal_l$post_suffix/$train_set/feats.ark.txt
  source activate beer
  if [ -z $clustering_lfr ]; then
    python scripts_hpc/run/clustering_kaldi_input.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo
  else
    python scripts_hpc/run/clustering_kaldi_input_lfr.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo $clustering_lfr
  fi
  source deactivate
fi

if [ $stage -le 21 ]  && [ $stop_stage -gt 21 ] ; then
  echo "evaluation AUD on k-means + bnf"
  eval_set=full
  if [ ! $eval_set = $train_set ]; then
    echo "$0: only full set evaluation supported "
    exit 1;
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
  output_dir=$dir/bnf_prefinal_l$post_suffix/$train_set/ #$dir/phone_post/$train_set  #phone_post_1hot_sym.txt
  #output_dir=$expdir_root/tri5_phone_ali${suffix}
  cut -d '-' -f 2- $output_dir/output_feats.ark.txt${clustering_suffix} > $output_dir/output_feats.ark.txt${clustering_suffix}.temp
  mv $output_dir/output_feats.ark.txt${clustering_suffix}.temp $output_dir/output_feats.ark.txt${clustering_suffix} 
  hyp_trans=$output_dir/output_feats.ark.txt${clustering_suffix}
  if [ ! -z $clustering_lfr ]; then
    hyp_trans=$hyp_trans_lfr${clustering_lfr}
    score_suffix="_lfr${clustering_lfr}"
  fi
  if [ ! -f $output_dir/output_feats.ark.txt${clustering_suffix} ]; then
    echo "$0: clustering output file not found";
    exit 1;
  fi
  eval_code_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/BEER/beer/recipes/hshmm/
  mboshi_ref_ali=$eval_code_root/data/mboshi/$eval_set/ali
  len_hyp=$(wc -l $output_dir/output_feats.ark.txt${clustering_suffix} | cut -d ' ' -f 1 )
  len_ref=$(wc -l $mboshi_ref_ali | cut -d ' ' -f 1)
  if [ ! $len_hyp = $len_ref  ]; then
    echo "exit because not all utterances in $eval_set have hyp transcripts"
    exit 1;
  fi
  source activate beer
  cwd=$(pwd)
  cd $eval_code_root
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud${clustering_suffix}${score_suffix}
  source deactivate
  cd $cwd
fi

echo "$0: Ended"
