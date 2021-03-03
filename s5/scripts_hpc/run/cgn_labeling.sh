#!/bin/bash
cmd="run.pl"
nj=1
stage=0
stop_stage=1
num_threads_decode=1
post_decode_acwt=10.0
acwt=10.0
decode_nj=1
ctm_format=false
hires_conf_set=cgn # or hkust, or aidatatang
data_dir_name=data_hires_conf_${hires_conf_set}
model=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/chain/tdnnf_related/tdnn1g_sp_bi_epoch4
gmm_model=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/train/tri4
align_fmllr_stage=0
ood_model_suffix="" # out-of-domain ASR model suffix, can be "_aug" or "_aug_sp"
. ./utils/parse_options.sh
. ./path.sh

ivector_extractor=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/$hires_conf_set/s5/exp/nnet3/extractor${ood_model_suffix}
graph_dir=$model/graph
if [ $stage -le -1 ] && [ $stop_stage -gt -1 ];then
    for set in  train dev; do
        steps/make_mfcc.sh  --nj $nj data/$set || exit 1;
        steps/compute_cmvn_stats.sh data/$set || exit 1;
    done
    utils/combine_data.sh data/full data/train data/dev || exit 1;
fi

if [ $stage -le 0 ] && [ $stop_stage -gt 0 ];then
    for set in  train dev; do
        utils/copy_data_dir.sh data/$set ${data_dir_name}/$set || exit 1;
        steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_from_${hires_conf_set}.conf --nj $nj ${data_dir_name}/$set || exit 1;
        steps/compute_cmvn_stats.sh ${data_dir_name}/$set || exit 1;
    done
    utils/combine_data.sh $data_dir_name/full $data_dir_name/train $data_dir_name/dev
fi

if [ $stage -le 1 ] && [ $stop_stage -gt 1 ];then
   echo "$0: extracting ivectors"
   data_root_path=$data_dir_name
   for set in train dev full; do
       input_data=$data_root_path/$set
       output_data=$ivector_extractor/../ivectors${ood_model_suffix}_mboshi_hires/$set
       steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
         $input_data $ivector_extractor $output_data || exit 1;
   done
fi

if [ $stage -le 2 ] && [ $stop_stage -gt 2 ];then
  echo "$0: decoding mboshi speech data with OOD ASR $hires_conf_set "
  data_root_path=$data_dir_name
  for set in train dev full; do
    # decode 
    input_data=$data_root_path/$set
    nspk=$(wc -l <$input_data/spk2utt)
    [ "$nspk" -gt "$nj" ] && nspk=$decode_nj 
    decoding_dir=$model/decoding_for_mboshi_${set}_acwt${acwt}
    input_ivec_dir=$ivector_extractor/../ivectors${ood_model_suffix}_mboshi_hires/$set
    steps/nnet3/decode.sh --num-threads $num_threads_decode --nj $nspk \
      --cmd "run.pl" --online-ivector-dir $input_ivec_dir \
      --skip-scoring true --acwt ${acwt} --post-decode-acwt ${post_decode_acwt} \
      $graph_dir $input_data $decoding_dir || exit 1; 
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -gt 3 ];then
  echo "$0: lattice to best path"
  for set in train dev full; do
    input_dir=$model/decoding_for_mboshi_${set}_acwt${acwt}
    num_jobs=$(cat $input_dir/num_jobs) || exit 1;
    $cmd JOB=1:$num_jobs $input_dir/log/lattice_best_path.JOB.log \
      lattice-best-path --lm-scale=0.001 "ark:gunzip -c $input_dir/lat.JOB.gz|" "ark,t:|utils/int2sym.pl -f 2- $input_dir/../graph/words.txt >   $input_dir/text.JOB" || exit 1;
    cat $input_dir/text.* > $input_dir/text || exit 1;
  done 
fi

if [ $stage -le 4 ] && [ $stop_stage -gt 4 ];then
  echo "$0: compose data_plus_${hires_conf_set}_transcripts/ dirs and convert best path to alignments"
  for set in train  full; do
#  for set in dev ; do
    text_dir=$model/decoding_for_mboshi_${set}_acwt${acwt}
    utils/copy_data_dir.sh data/$set data_plus_${hires_conf_set}_transcripts/acwt${acwt}/$set || exit 1
    cp $text_dir/text data_plus_${hires_conf_set}_transcripts/acwt${acwt}/$set/
    utils/fix_data_dir.sh data_plus_${hires_conf_set}_transcripts/acwt${acwt}/$set/
#    if [ ! -f data_plus_${hires_conf_set}_transcripts/acwt${acwt}/$set/feats.scp ]; then
 
    ali_dir=exp/chain_${hires_conf_set}_lat_label/$set/lat_gen_acwt${acwt}/${set}_ali
    steps/align_fmllr.sh --nj $nj --cmd run.pl --stage $align_fmllr_stage \
      data_plus_${hires_conf_set}_transcripts/acwt${acwt}/$set $gmm_model/../../../data/lang_s $gmm_model $ali_dir || exit 1
  done

fi

if [ $stage -le 5 ] && [ $stop_stage -gt 5 ];then
  echo "$0: get phone sequences from model-level alignments"
  for set in train full; do
#  for set in dev; do
    dir=exp/chain_${hires_conf_set}_lat_label/$set/lat_gen_acwt${acwt}/${set}_ali
    num_jobs=$(cat $dir/num_jobs) || exit 1
    output_dir=exp/chain_${hires_conf_set}_lat_label/$set/lat_gen_acwt${acwt}/${set}_phone_ali
    mkdir -p $output_dir
    if $ctm_format; then
      # CTM format alignment
      $cmd JOB=1:$num_jobs $output_dir/log/get_ctm.JOB.log \
        ali-to-phones --ctm-output $dir/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" "|utils/int2sym.pl -f 5- $dir/phones.txt > $output_dir/ctm.JOB.txt " || exit 1;
      cat $output_dir/ctm.*.txt > $output_dir/ctm.txt || exit 1
      rm -f $output_dir/ctm.*.txt 
    else
      # regular kaldi-style alignment
      $cmd JOB=1:$num_jobs $output_dir/log/get_phone_ali.JOB.log \
        ali-to-phones --per-frame $dir/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" ark,t:- \| utils/int2sym.pl -f 2- $dir/phones.txt \> $output_dir/phone_ali.JOB.ali
      cat $output_dir/phone_ali.*.ali > $output_dir/phone_ali.ali || exit 1
      rm -f $output_dir/phone_ali.*.ali
    fi
    cp $dir/phones.txt $output_dir/phones.txt
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -gt 6 ];then
  echo "$0: convert position-dependent phone alignments to -independent ones"
  for set in dev train full; do
#  for set in dev; do
    output_dir=exp/chain_${hires_conf_set}_lat_label/$set/lat_gen_acwt${acwt}/${set}_phone_ali
    if [ -f $output_dir/ctm.txt ]; then
      echo "processing $output_dir/ctm.ali"
      sed -e "s/_[B,E,I,S] $/ /g" $output_dir/ctm.txt > $output_dir/ctm_pos_ind.txt || exit 1
    fi
    if [ -f $output_dir/phone_ali.ali ]; then
      echo "processing $output_dir/phone_ali.ali"
      sed -e "s/_[B,E,I,S] / /g" $output_dir/phone_ali.ali > $output_dir/phone_ali_pos_ind.ali
    fi
    if [ -f $output_dir/phones.txt ]; then
      echo "processing $output_dir/phones.txt"
      sed -e "/_[E,I,S]/d" -e "/SIL 1/d" $output_dir/phones.txt > $output_dir/temp_phones_pos_ind.txt
      sed -i -e "s/_B / /g" $output_dir/temp_phones_pos_ind.txt
      cut -d ' ' -f 1 $output_dir/temp_phones_pos_ind.txt | awk '{print $1,NR-1}' - > $output_dir/phones_pos_ind.txt
      rm -f $output_dir/temp_phones_pos_ind.txt
    fi
  done 
fi

if [ $stage -le  7 ] && [ $stop_stage -gt 7 ]; then
  echo "$0: evaluate OOD phone label  AUD tasks"
  eval_set=full
  output_dir=exp/chain_${hires_conf_set}_lat_label/$eval_set/lat_gen_acwt${acwt}/${eval_set}_phone_ali
  eval_code_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/BEER/beer/recipes/hshmm/
  mboshi_ref_ali=$eval_code_root/data/mboshi/$eval_set/ali
  hyp_trans=$output_dir/phone_ali_pos_ind.ali
  source activate beer
  cwd=$(pwd)
  cd $eval_code_root
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud
  source deactivate 
  cd $cwd
fi
