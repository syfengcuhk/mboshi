#!/bin/bash

# Siyuan Feng (2020): use a pretrained monolingual phone-level ASR model to generate label supervision for Mboshi
bn_dim=40
frame_subsampling_factor=3
LMTYPE=phn_ug_mono
model_name=final
discophone_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/discophone/v1_phones_lms
align_fmllr_lats_retry_beam=10
align_fmllr_lats_beam=40
remove_egs=false
stop_stage=19
stage=17
align_fmllr_lats_stage=-10
nj=1
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
gp_langs="Czech" #"Czech French Mandarin Spanish Thai"
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
lang_name=$(langname $list_train_set)
dir=exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tdnn${tdnn_affix}${data_aug_suffix}_${train_set}
train_ivector_dir=exp/chain_discophone_lat_label/nnet3/ivectors_${lang_name}_mboshi_hires/$train_set
gmm_dir=$discophone_root/exp/gmm/$lang_name/$gmm
ali_dir=exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/${train_set}_ali
tree_dir=exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tree_bi${tree_affix}
lat_dir=exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/${train_set}_lats
input_feat_affix=mfcc_hires_pitch
lores_train_data_dir=data_plus_discophone_transcripts_collapse_ali/${lang_name}_${LMTYPE}/acwt${lat_generator_acwt}/${train_set}
train_data_dir=data_hires_pitch_conf_discophone/${train_set}
common_egs_dir=
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
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor $frame_subsampling_factor \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} $discophone_root/data/lang_chain/$lang_name/ $ali_dir $tree_dir
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
    --feat-dir ${train_data_dir} \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi
