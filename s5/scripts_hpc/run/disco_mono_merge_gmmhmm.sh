#!/bin/bash

# Similar to mono_gmmhmm.sh, but merge 5 versions of Discophone GP monolingual phone transcripts, and train a GMM-HMM. Do not include Babel, because their crosslingual transcripts are not reliable
LMTYPE=phn_ug_mono
beam=10
retry_beam=40
stage=0
stop_stage=500
train_nj=2
phone_tokens=false
disco_acwt=10.0
# Acoustic model parameters
numLeavesTri1=2000 #1000
numGaussTri1=25000 #10000
#numLeavesTri2=1000
#numGaussTri2=20000
#numLeavesTri3=6000
#numGaussTri3=75000
numLeavesMLLT=2000
numGaussMLLT=25000
numLeavesSAT=2000
numGaussSAT=25000
ctm_format=false
cmd="run.pl"
lang_name_suffix="_GP"
full_train_set=full
discophone_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/discophone/v1_multilang_phones_lms
data_suffix=_collapse_ali # by default, then data dir uses text from lattice-best-path ali, see last stage of disco_multi_labeling.sh 
. cmd.sh
. utils/parse_options.sh
. path.sh
lang_name=merge_transcript${lang_name_suffix}
expdir_root=exp/gmm_discophone_label${data_suffix}/${lang_name}_${LMTYPE}/$full_train_set
lang=$discophone_root/data/lang_combined_test
data_dir=data_plus_discophone_transcripts${data_suffix}/prefixed/${lang_name}_${LMTYPE}/acwt${disco_acwt}/$full_train_set

if [ $stage -le 6 ] && [ $stop_stage -gt 6 ]; then
# to do: merge 5/13 languages data/ dir and get into 5 times the amount of labeled speech
# 1st add language prefix
  data_root_add_prefix=data_plus_discophone_transcripts${data_suffix}/prefixed/
  mkdir -p $data_root_add_prefix || exit 1
  for gp_lang in Czech French Spanish Mandarin Thai; do
    utils/copy_data_dir.sh --spk-prefix "${gp_lang}-" --utt-prefix "${gp_lang}-" data_plus_discophone_transcripts${data_suffix}/gp_${gp_lang}_train_${LMTYPE}/acwt${disco_acwt}/$full_train_set $data_root_add_prefix/gp_${gp_lang}_train_${LMTYPE}/acwt${disco_acwt}/$full_train_set
  done
  utils/combine_data.sh $data_dir $data_root_add_prefix/gp_Czech_train_${LMTYPE}/acwt${disco_acwt}/$full_train_set $data_root_add_prefix/gp_French_train_${LMTYPE}/acwt${disco_acwt}/$full_train_set $data_root_add_prefix/gp_Spanish_train_${LMTYPE}/acwt${disco_acwt}/$full_train_set $data_root_add_prefix/gp_Mandarin_train_${LMTYPE}/acwt${disco_acwt}/$full_train_set $data_root_add_prefix/gp_Thai_train_${LMTYPE}/acwt${disco_acwt}/$full_train_set 
fi

if [ $stage -le 7 ] && [ $stop_stage -gt 7 ]; then
  steps/train_mono.sh \
    --nj $train_nj --cmd "$train_cmd" \
    $data_dir \
    $lang $expdir_root/mono || exit 1;
fi

if [ $stage -le 8 ] && [ $stop_stage -gt 8 ]; then
  # Tri1 training
  if [ ! -f $expdir_root/mono_ali/ali.1.gz ]; then
    steps/align_si.sh \
      --nj $train_nj --cmd "$train_cmd" \
      $data_dir \
      $lang \
      $expdir_root/mono \
      $expdir_root/mono_ali || exit 1;
  fi
  steps/train_deltas.sh \
    --cmd "$train_cmd" \
    $numLeavesTri1 \
    $numGaussTri1 \
    $data_dir \
    $lang \
    $expdir_root/mono_ali \
    $expdir_root/tri1 || exit 1;
fi

if (($stage <= 11)) && (($stop_stage > 11)); then
  # Tri4 training
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    $data_dir \
    $lang \
    $expdir_root/tri1 \
    $expdir_root/tri1_ali || exit 1;

  steps/train_lda_mllt.sh \
    --cmd "$train_cmd" \
    $numLeavesMLLT \
    $numGaussMLLT \
    $data_dir \
    $lang \
    $expdir_root/tri1_ali \
    $expdir_root/tri4 || exit 1;
fi
if (($stage <= 12)) && (($stop_stage > 12)); then
  # Tri5 training
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    $data_dir \
    $lang \
    $expdir_root/tri4 \
    $expdir_root/tri4_ali || exit 1;

  steps/train_sat.sh \
    --cmd "$train_cmd" \
    $numLeavesSAT \
    $numGaussSAT \
    $data_dir \
    $lang \
    $expdir_root/tri4_ali \
    $expdir_root/tri5 || exit 1;
fi
if [ ! $retry_beam = 40 ] ; then
  suffix="_retry_beam${retry_beam}"
else
  suffix=""
fi

if (($stage <= 13)) && (($stop_stage > 13)); then
  # Tri5 alignments
  steps/align_fmllr.sh --beam $beam --retry-beam $retry_beam \
    --nj $train_nj --cmd "$train_cmd" \
    $data_dir \
    $lang \
    $expdir_root/tri5 \
    $expdir_root/tri5_ali${suffix} || exit 1;
fi

if (($stage <= 14)) && (($stop_stage > 14)); then
  echo "$0: get phone sequences from model-level alignments, alignments got from align_fmllr.sh "
  # Compose a dir $expdir_root/tri5_phone_ali
  set=$full_train_set
  dir=$expdir_root/tri5_ali${suffix}
  num_jobs=$(cat $dir/num_jobs) || exit 1
  output_dir=$expdir_root/tri5_phone_ali${suffix}
  mkdir -p $output_dir
  if $ctm_format; then
    $cmd JOB=1:$num_jobs $output_dir/log/get_ctm.JOB.log \
      ali-to-phones --ctm-output $dir/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" "|utils/int2sym.pl -f 5- $dir/phones.txt > $output_dir/ctm.JOB.txt " || exit 1;
      cat $output_dir/ctm.*.txt > $output_dir/ctm.txt || exit 1
      rm -f $output_dir/ctm.*.txt
  else
    $cmd JOB=1:$num_jobs $output_dir/log/get_phone_ali.JOB.log \
      ali-to-phones --per-frame $dir/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" ark,t:- \| utils/int2sym.pl -f 2- $dir/phones.txt \> $output_dir/phone_ali.JOB.ali
      cat $output_dir/phone_ali.*.ali > $output_dir/phone_ali.ali || exit 1
      rm -f $output_dir/phone_ali.*.ali
  fi
  cp $dir/phones.txt $output_dir/phones.txt
fi

if (($stage <= 15)) && (($stop_stage > 15)); then
  echo "$0: evaluate OOD phone label  AUD tasks"
  eval_set=full
  if [ ! $eval_set = $full_train_set ]; then
    echo "$0: only full set evaluation supported "
    exit 1;
  fi
  output_dir=$expdir_root/tri5_phone_ali${suffix}
 #(to do: reduce to 1/5) hyp_trans=$output_dir/phone_ali.ali
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

echo "$0: ended"

