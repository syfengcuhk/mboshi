#!/bin/bash
# based on wsj/s5/local/nnet3/tuning/run_tdnn_1a.sh
# similar to disco_mono_tdnnf_1g_own_gmm.sh, except nnet3, non-chain TDNN model, training objecttive is cross-entropy, no LF-MMI
# no frame subsampling rate
clustering_lfr= # by default no, if specified, (e.g. 3) when clustering BNF, 1/3 frame rate 
rand_state=0
bin_output=false
nclusters=50
clustering_algo="kmeans" # or "dbscan" or "spectral"
post_model=final
bn_dim=40
LMTYPE=phn_ug_mono
model_name=final
discophone_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/discophone/v1_phones_lms
stop_stage=19
stage=17
nj_clustering=2
nj=2
gmm=tri5
nnet3_affix=
tdnn_affix=1a
train_stage=-10
train_exit_stage=0
srand=0
get_egs_stage=-10
tree_affix=
tdnn_affix=1a
babel_langs="" #"307 103 101 402 107 206 404 203"
babel_recog="${babel_langs}"
gp_langs="Czech" #"Czech French Mandarin Spanish Thai"
gp_recog="${gp_langs}"
num_jobs_initial=1
train_set=full
lat_generator_acwt=10.0
num_jobs_final=1
num_epochs=5
initial_effective_lrate=0.001
final_effective_lrate=0.0001
minibatch_size=128
dropout_schedule='0,0@0.20,0.3@0.50,0'
max_param_change=2.0
gpu_extract_post=true
remove_egs=false
common_egs_dir=
. ./cmd.sh
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
dir=exp/chain_discophone_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${lat_generator_acwt}/tdnn${tdnn_affix}${data_aug_suffix}_${train_set}_own_gmm_ep$num_epochs
train_ivector_dir=exp/chain_discophone_lat_label/nnet3/ivectors_${lang_name}_mboshi_hires/$train_set
gmm_dir=exp/gmm_discophone_label/${lang_name}_${LMTYPE}/$train_set/$gmm
ali_dir=${gmm_dir}_ali
lang=$discophone_root/data/lang_combined_test
input_feat_affix=mfcc_hires_pitch
train_data_dir=data_hires_pitch_conf_discophone/${train_set}
lores_train_data_dir=data_plus_discophone_transcripts_collapse_ali/${lang_name}_${LMTYPE}/acwt${lat_generator_acwt}/${train_set}
for f in  $train_data_dir/feats.scp $lores_train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp ; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 17 ] && [ $stop_stage -gt 17 ]; then
  feat_dim=$(feat-to-dim scp:${train_data_dir}/feats.scp -)
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser"
  num_targets=$(tree-info $gmm_dir/tree |grep num-pdfs|awk '{print $2}')
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=$feat_dim name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=512
  relu-renorm-layer name=tdnn2 dim=512 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=512 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn4 dim=512 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn5 dim=512 input=Append(-6,-3,0)
  linear-component name=prefinal-l dim=$bn_dim $linear_opts
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 18 ] && [ $stop_stage -gt 18 ]; then
  steps/nnet3/train_dnn.py --stage=$train_stage --exit-stage $train_exit_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=$max_param_change \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=200000 \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.minibatch-size=$minibatch_size \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval 4 \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=$lang \
    --dir=$dir  || exit 1;
fi
if [ !  "$post_model" = "final" ]; then
  post_suffix="_${post_model}"
else
  post_suffix=""
fi
if [ $stage -le 19 ] && [ $stop_stage -gt 19 ] ; then
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
  if [ -z $clustering_lfr ]; then
    python scripts_hpc/run/clustering_kaldi_input.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo
  else
    python scripts_hpc/run/clustering_kaldi_input_lfr.py $input_data $nj_clustering $rand_state $nclusters $clustering_algo $clustering_lfr
  fi
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
  hyp_trans=$output_dir/output_feats.ark.txt${clustering_suffix}
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
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud${clustering_suffix}${score_suffix}
  source deactivate
  cd $cwd

fi
