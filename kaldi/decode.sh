#!/bin/bash
stage=1

data=$1
logdir=$2

pwddir=`pwd`
datadir=$1
gmmdir=exp/tri4b
##srcdir=/home/bliu/SRC/workspace/kaldi-master-1/egs/chime4/s5_1ch
srcdir=/home/bliu/SRC/workspace/kaldi-master-1/egs/wsj/s5
##dir=exp/dnn_py
dir=exp/dnn5b_pretrain-dbn_dnn_py
rescoredir=exp/dnn_py_lmrescore
rnn_rescoredir=exp/dnn_py_rnn_lmrescore
##python utils/feats_trans.py ${datadir}/feats.scp_tmp $pwddir ${datadir}/feats.scp

order=5
hidden=300
rnnweight=0.5
nbest=100

lm_suffix=${order}gkn_5k
rnnlm_suffix=rnnlm_5k_h${hidden}

##cat ${datadir}/feats.scp_tmp | awk -v aa=" $pwddir/" '{print $1 aa $2}' >${datadir}/feats.scp
cp ${datadir}/feats.scp_tmp ${datadir}/feats.scp
cd $srcdir

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
##. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.  
. ./path.sh




##utils/fix_data_dir.sh ${datadir}

##steps/nnet/decode_py.sh --nj 2 --cmd "$decode_cmd"  --acwt 0.1 --config conf/decode_dnn.config \
##    $dir/graph_bd_tgpr $datadir $dir/$logdir || exit 1;
##steps/nnet/decode_py.sh --nj 2 --cmd "$decode_cmd"  --acwt 0.1 --config conf/decode_dnn.config \
##    $gmmdir/graph_bd_tgpr $datadir $dir/$logdir || exit 1;   
if [ $stage -le 1 ]; then
  steps/nnet/decode_py.sh --num-threads 4 --nj 4 --cmd "$decode_cmd"  --acwt 0.1 --config conf/decode_dnn.config \
    $dir/graph_tgpr_5k $datadir $dir/$logdir || exit 1;    
  cp $dir/$logdir/scoring_kaldi/best_wer $datadir

fi


