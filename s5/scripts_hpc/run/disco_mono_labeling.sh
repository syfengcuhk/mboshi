#/bin/bash

# To do: leverage phone-level ASR models trained in discophone project to generate cross-lingual phone labels for mboshi.
# This script focuses on using 13 monolingual ASRs , 
# During label generation, some alternatives in choosing language models: (1) phonotactic LM (2) word LM. Also uni-/bi-/tri-gram 
#
discophone_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/discophone/v1_phones_lms
cmd="run.pl"
nj=1
stage=0
stop_stage=1
num_threads_decode=1
post_decode_acwt=10.0
acwt=10.0
decode_max_active=7000
gpu_decode=false
eval_from_lattice=true # if false, evaluated based on alignment by align_fmllr.sh, default by based on alignment from lattice-best-path
decode_nj=1
decode_beam=15.0
lattice_beam=8.0
ctm_format=false
babel_langs="" #"307 103 101 402 107 206 404 203"
babel_recog="${babel_langs}"
gp_langs="Czech" #"Czech French Mandarin Spanish Thai"
gp_recog="${gp_langs}"
hires_conf_set=discophone
align_fmllr_stage=0
align_fmllr_beam=10
align_fmllr_retry_beam=40
nnet3_affix=""
tdnn_affix="1g"
tree_affix=""
data_aug_suffix="" # or _sp
LMTYPE=phn_ug_mono # phn_{ug,bg,tg}_mono or words
. ./utils/parse_options.sh
. ./path.sh
data_dir_name=data_hires_pitch_conf_${hires_conf_set}
data_dir_name_nopitch=data_hires_conf_${hires_conf_set}
for l in  ${babel_langs}; do
  train_set="$l/data/train_${l} ${train_set}"
  dev_set="$l/data/dev_${l} ${dev_set}"
done
for l in ${gp_langs}; do
  train_set="GlobalPhone/gp_${l}_train ${train_set}"
  dev_set="GlobalPhone/gp_${l}_dev ${dev_set}"
done
train_set=${train_set%% }
dev_set=${dev_set%% }
function langname() {
  # Utility
  echo "$(basename "$1")"
}
lang_name=$(langname $train_set)
model=$discophone_root_path/exp/chain${nnet3_affix}/$lang_name/tdnn${tdnn_affix}${data_aug_suffix}
gmm_model=$discophone_root_path/exp/gmm/$lang_name/tri5
ivector_extractor=$discophone_root_path/exp/nnet3${nnet3_affix}/$lang_name/extractor
tree_dir=$discophone_root_path/exp/chain${nnet3_affix}/$lang_name/tree${tree_affix}
graph_dir=$tree_dir/graph_${LMTYPE}
echo "language name: $lang_name"
echo "model: $model"
echo "gmm_model: $gmm_model"
echo "LM used to generate labels: $graph_dir"

if [ $stage -le 0 ]  && [ $stop_stage -gt 0 ]; then
  for set in dev train; do
    utils/copy_data_dir.sh data/$set ${data_dir_name}/$set || exit 1;
    steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires_from_${hires_conf_set}.conf --nj $nj ${data_dir_name}/$set || exit 1;
    steps/compute_cmvn_stats.sh ${data_dir_name}/$set || exit 1;
  done
  utils/combine_data.sh $data_dir_name/full $data_dir_name/train $data_dir_name/dev
fi

if [ $stage -le 1 ] && [ $stop_stage -gt 1 ];then
   echo "$0: extracting ivectors"
   data_root_path_nopitch=$data_dir_name_nopitch
   for set in train dev full; do
      input_data=$data_root_path_nopitch/$set
      if [ ! -d $input_data ]; then
          echo "$0: first compose High Resolution MFCC Without pitch"
          utils/data/limit_feature_dim.sh 0:39 $data_dir_name/$set $input_data || exit 1;
          steps/compute_cmvn_stats.sh $input_data || exit 1;
      fi
      output_data=exp/chain_${hires_conf_set}_lat_label/nnet3/ivectors_${lang_name}_mboshi_hires/$set
      echo "Then extract ivectors using $ivector_extractor"
      steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
         $input_data $ivector_extractor $output_data || exit 1;
   done
fi

if [ $stage -le 2 ] && [ $stop_stage -gt 2 ];then
  echo "$0: decoding mboshi speech data with OOD ASR $hires_conf_set "
  data_root_path=$data_dir_name
  for set in   dev train ; do
    input_data=$data_root_path/$set
    nspk=$(wc -l <$input_data/spk2utt)
    [ "$nspk" -gt "$nj" ] && nspk=$decode_nj
    decoding_dir=$model/decoding_for_mboshi_${LMTYPE}_${set}_acwt${acwt}
    input_ivec_dir=exp/chain_${hires_conf_set}_lat_label/nnet3/ivectors_${lang_name}_mboshi_hires/$set
    steps/nnet3/decode.sh  --max-active $decode_max_active --beam $decode_beam --lattice-beam $lattice_beam --num-threads $num_threads_decode --nj $nspk \
      --use-gpu $gpu_decode  --cmd "run.pl" --online-ivector-dir $input_ivec_dir \
      --skip-scoring true --acwt ${acwt} --post-decode-acwt ${post_decode_acwt} \
      $graph_dir $input_data $decoding_dir || exit 1;
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -gt 3 ];then
  echo "$0: lattice to best path"
  for set in    dev train ; do
    input_dir=$model/decoding_for_mboshi_${LMTYPE}_${set}_acwt${acwt}
    num_jobs=$(cat $input_dir/num_jobs) || exit 1;
    if [[ $LMTYPE == *"phn"* ]]; then
    $cmd JOB=1:$num_jobs $input_dir/log/lattice_best_path.JOB.log \
      lattice-best-path --lm-scale=0.001 "ark:gunzip -c $input_dir/lat.JOB.gz|" "ark,t:|utils/int2sym.pl -f 2- $graph_dir/words.txt > $input_dir/text.JOB" "ark:|gzip -c - > $input_dir/ali.JOB.gz" || exit 1;
    elif [[ $LMTYPE == *"words"* ]]; then
    # word constraints used in decoding
    $cmd JOB=1:$num_jobs $input_dir/log/lattice_best_path.JOB.log \
      lattice-to-phone-lattice $model/final.mdl "ark:gunzip -c $input_dir/lat.JOB.gz|" ark:- \| lattice-best-path --lm-scale=0.001 ark:- "ark,t:|utils/int2sym.pl -f 2- $graph_dir/phones.txt > $input_dir/text.JOB" "ark:|gzip -c - > $input_dir/ali.JOB.gz" || exit 1
    else
      echo "LMTYPE $LMTYPE not supported" || exit 1
    fi
    cat $input_dir/text.* > $input_dir/text || exit 1;
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -gt 4 ];then
  echo "$0: compose data_plus_${hires_conf_set}_transcripts/ dirs and convert best path to alignments"
  for set in  full  ; do
    if [ ! -d data_plus_${hires_conf_set}_transcripts/${lang_name}_${LMTYPE}/acwt${acwt}/$set ]; then
      text_dir=$model/decoding_for_mboshi_${LMTYPE}_${set}_acwt${acwt}
      utils/copy_data_dir.sh data/$set data_plus_${hires_conf_set}_transcripts/${lang_name}_${LMTYPE}/acwt${acwt}/$set || exit 1
      cp $text_dir/text data_plus_${hires_conf_set}_transcripts/${lang_name}_${LMTYPE}//acwt${acwt}/$set/
      utils/fix_data_dir.sh data_plus_${hires_conf_set}_transcripts/${lang_name}_${LMTYPE}/acwt${acwt}/$set/
    fi
    ali_dir=exp/chain_${hires_conf_set}_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${acwt}/${set}_ali
    steps/align_fmllr.sh --beam $align_fmllr_beam --retry-beam $align_fmllr_retry_beam --nj $nj --cmd run.pl --stage $align_fmllr_stage \
      data_plus_${hires_conf_set}_transcripts/${lang_name}_${LMTYPE}//acwt${acwt}/$set $discophone_root_path/data/lang_combined_test $gmm_model $ali_dir || exit 1
    # collect utterances that are not successfully aligned
    awk '/Did not/ {print}' $ali_dir/log/align_pass2.*.log > $ali_dir/log/utt_unsuccessful.log
    if [ ! -z $ali_dir/log/utt_unsuccessful.log ]; then
      echo "number of unsuccessfully aligned files: $(wc -l $ali_dir/log/utt_unsuccessful.log)"
    fi
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -gt 5 ];then
  echo "$0: get phone sequences from model-level alignments, alignments got from align_fmllr.sh"
  for set in  full; do
    dir=exp/chain_${hires_conf_set}_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${acwt}/${set}_ali
    num_jobs=$(cat $dir/num_jobs) || exit 1
    output_dir=exp/chain_${hires_conf_set}_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${acwt}/${set}_phone_ali
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
  done
fi
if [ $stage -le 6 ] && [ $stop_stage -gt 6 ];then
  echo "$0: get phone sequences from model-level alignments, alignments got from lattice-best-path"
  for set in  full; do
    dir=$model/decoding_for_mboshi_${LMTYPE}_${set}_acwt${acwt}
    num_jobs=$(cat $dir/num_jobs) || exit 1
    output_dir=exp/chain_${hires_conf_set}_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${acwt}/${set}_phone_ali_from_lattice
    mkdir -p $output_dir
    if $ctm_format; then
      $cmd JOB=1:$num_jobs $output_dir/log/get_ctm.JOB.log \
        ali-to-phones --frame-shift=0.03 --ctm-output $model/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" "|utils/int2sym.pl -f 5- $graph_dir/phones.txt > $output_dir/ctm.JOB.txt " || exit 1;
      cat $output_dir/ctm.*.txt > $output_dir/ctm.txt || exit 1
      rm -f $output_dir/ctm.*.txt
    else
      $cmd JOB=1:$num_jobs $output_dir/log/get_phone_ali.JOB.log \
        ali-to-phones --per-frame $model/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" ark,t:- \| utils/int2sym.pl -f 2- $graph_dir/phones.txt \> $output_dir/phone_ali.JOB.ali
      cat $output_dir/phone_ali.*.ali > $output_dir/phone_ali.ali || exit 1
      rm -f $output_dir/phone_ali.*.ali   
      python scripts_hpc/run/align_duplicate.py $output_dir/phone_ali.ali 3 # because phone_ali.ali has a subsampling factor 3. convert to normal (10ms) 
    fi 
    cp $graph_dir/phones.txt $output_dir/phones.txt
  done

fi
if [ $stage -le  8 ] && [ $stop_stage -gt 8 ]; then
  echo "$0: evaluate OOD phone label  AUD tasks"
  eval_set=full
  if $eval_from_lattice; then
    output_dir=exp/chain_${hires_conf_set}_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${acwt}/${eval_set}_phone_ali_from_lattice
    hyp_trans=$output_dir/phone_ali.ali.rep3
  else
    output_dir=exp/chain_${hires_conf_set}_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${acwt}/${eval_set}_phone_ali
    hyp_trans=$output_dir/phone_ali.ali
  fi
  eval_code_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/BEER/beer/recipes/hshmm/
  mboshi_ref_ali=$eval_code_root/data/mboshi/$eval_set/ali
  source activate beer
  cwd=$(pwd)
  cd $eval_code_root
  bash $eval_code_root/steps/score_aud.sh $mboshi_ref_ali $cwd/$hyp_trans $cwd/$output_dir/score_aud
  source deactivate
  cd $cwd

fi
if [ $stage -le 9  ] && [ $stop_stage -gt 9 ]; then
  echo "based on stage 6, create transcript for mboshi based on lattice-best-path alignment results. Collapsing repetitive, and discard <silence>"
  for set in  full; do
    dir=$model/decoding_for_mboshi_${LMTYPE}_${set}_acwt${acwt}
    num_jobs=$(cat $dir/num_jobs) || exit 1
    output_dir=exp/chain_${hires_conf_set}_lat_label/${lang_name}_${LMTYPE}/lat_gen_acwt${acwt}/${set}_phone_ali_from_lattice
    $cmd JOB=1:$num_jobs $output_dir/log/get_phone_ali_transcript.JOB.log \
      ali-to-phones  $model/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" ark,t:- \| utils/int2sym.pl -f 2- $graph_dir/phones.txt \> $output_dir/phone_ali_transcript.JOB
    cat $output_dir/phone_ali_transcript.* > $output_dir/phone_ali_transcript || exit 1
    rm -f $output_dir/phone_ali_transcript.*
    sed -e "s/<silence> //g" -e "s/<unk> //g" $output_dir/phone_ali_transcript > $output_dir/phone_ali_transcript.cleaned
    mkdir -p $output_dir
    if [ ! -d data_plus_${hires_conf_set}_transcripts_collapse_ali/${lang_name}_${LMTYPE}/acwt${acwt}/$set ]; then
      utils/copy_data_dir.sh data/$set data_plus_${hires_conf_set}_transcripts_collapse_ali/${lang_name}_${LMTYPE}/acwt${acwt}/$set || exit 1
      cp $output_dir/phone_ali_transcript.cleaned data_plus_${hires_conf_set}_transcripts_collapse_ali/${lang_name}_${LMTYPE}//acwt${acwt}/$set/text
      utils/fix_data_dir.sh data_plus_${hires_conf_set}_transcripts_collapse_ali/${lang_name}_${LMTYPE}//acwt${acwt}/$set/
    else
      echo "Skip creating dir as already exitts: data_plus_${hires_conf_set}_transcripts_collapse_ali/${lang_name}_${LMTYPE}/acwt${acwt}/$set"
    fi
  done
fi

# Next I'm going to use discophone GMM model to decode Mboshi
if [ $stage -le 10  ] && [ $stop_stage -gt 10 ]; then
  echo "$0: decode using discophone GMM "

fi

echo "$0: succeeded ..."
