#!/bin/bash

filter_regexp=.
[ $# -ge 1 ] && filter_regexp=$1


exp_dir=work/nnet/split_*/dnn_dbn_dnn
exp_dir=work/nnet/less_20/d_order_2/depth_2/dim_1024/dnn_dbn_dnn
exp_dir=exp/tri2

exp_dir=work/nnet/less_80/d_order_2/depth_2/dim_1024/dnn_dbn_dnn

for x in $exp_dir/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  for x in $exp_dir/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null | grep $filter_regexp
exit 0

# Results from Nikolay, using kaldi scoring:
# %WER 35.17 [ 9677 / 27512, 1267 ins, 1681 del, 6729 sub ] exp/tri1/decode/wer_13
# %WER 30.03 [ 8262 / 27512, 1255 ins, 1367 del, 5640 sub ] exp/tri2/decode/wer_15
# %WER 24.99 [ 6876 / 27512, 1314 ins, 1015 del, 4547 sub ] exp/tri3/decode/wer_14
# %WER 30.10 [ 8281 / 27512, 1257 ins, 1368 del, 5656 sub ] exp/tri3/decode.si/wer_16
# %WER 21.31 [ 5863 / 27512, 1121 ins, 875 del, 3867 sub ] exp/tri3_mmi_b0.1/decode_it4/wer_12

# GMMs (Cantab LM)
# DEV SPEAKERS:
%WER 30.9 | 507 17792 | 73.6 20.1 6.3 4.4 30.9 96.8 | 0.012 | exp/tri1/decode_dev/score_11/ctm.filt.filt.sys
%WER 26.8 | 507 17792 | 77.5 17.1 5.4 4.3 26.8 96.1 | -0.057 | exp/tri2/decode_dev/score_13/ctm.filt.filt.sys
%WER 22.3 | 507 17792 | 81.6 14.0 4.4 3.9 22.3 94.1 | -0.102 | exp/tri3/decode_dev/score_14/ctm.filt.filt.sys
%WER 19.1 | 507 17792 | 84.1 12.4 3.6 3.2 19.1 92.3 | 0.005 | exp/tri3_mmi_b0.1/decode_dev_it4/score_12/ctm.filt.filt.sys

# TEST SPEAKERS:
%WER 25.7 | 1155 27512 | 77.7 17.5 4.8 3.4 25.7 91.6 | -0.007 | exp/tri2/decode_test/score_13/ctm.filt.filt.sys
%WER 20.6 | 1155 27512 | 82.5 13.8 3.7 3.1 20.6 89.4 | -0.038 | exp/tri3/decode_test/score_13/ctm.filt.filt.sys
%WER 30.9 | 1155 27512 | 73.0 21.4 5.5 4.0 30.9 93.8 | 0.025 | exp/tri1/decode_test/score_11/ctm.filt.filt.sys
%WER 17.6 | 1155 27512 | 85.1 11.8 3.2 2.7 17.6 87.6 | 0.038 | exp/tri3_mmi_b0.1/decode_test_it4/score_11/ctm.filt.filt.sys

# Karel's DNN
%WER 15.0 | 1155 27512 | 87.2 9.7 3.0 2.2 15.0 83.6 | -0.014 | exp/dnn4_pretrain-dbn_dnn/decode_test/score_10/ctm.filt.filt.sys
%WER 13.3 | 1155 27512 | 88.9 8.8 2.3 2.2 13.3 81.4 | -0.063 | exp/dnn4_pretrain-dbn_dnn_smbr_i1lats/decode_test_it4/score_11/ctm.filt.filt.sys
%WER 12.8 | 1155 27512 | 89.0 8.4 2.6 1.8 12.8 80.2 | -0.122 | exp/dnn8e_BN_pretrain-dbn_dnn/decode_test/score_10/ctm.filt.filt.sys
%WER 11.7 | 1155 27512 | 90.2 7.7 2.1 2.0 11.7 78.7 | -0.110 | exp/dnn8f_BN_pretrain-dbn_dnn_smbr/decode_test_it4/score_12/ctm.filt.filt.sys

# Karel's DNN + rescore
%WER 13.5 | 1155 27512 | 88.4 8.5 3.1 1.9 13.5 81.6 | -0.074 | exp/dnn4_pretrain-dbn_dnn/decode_test.rescore/score_10/ctm.filt.filt.sys
%WER 11.9 | 1155 27512 | 90.0 7.6 2.3 1.9 11.9 78.2 | -0.110 | exp/dnn4_pretrain-dbn_dnn_smbr_i1lats/decode_test_it4.rescore/score_11/ctm.filt.filt.sys
%WER 10.6 | 1155 27512 | 91.0 6.7 2.3 1.6 10.6 74.4 | -0.194 | exp/dnn8f_BN_pretrain-dbn_dnn_smbr/decode_test_it4.rescore/score_12/ctm.filt.filt.sys

# multi-splice + i-vector + perturbed
%WER 14.0 | 507 17792 | 88.5 8.4 3.0 2.5 14.0 88.0 | -0.074 | exp/nnet2_online/nnet_ms_sp/decode_dev/score_12/ctm.filt.filt.sys
%WER 13.3 | 1155 27512 | 88.7 8.7 2.6 2.0 13.3 81.5 | -0.097 | exp/nnet2_online/nnet_ms_sp/decode_test/score_10/ctm.filt.filt.sys
%WER 13.2 | 1155 27512 | 88.6 8.5 2.8 1.9 13.2 81.6 | -0.102 | exp/nnet2_online/nnet_ms_sp_online/decode_test/score_11/ctm.filt.filt.sys
%WER 13.6 | 1155 27512 | 88.5 8.9 2.6 2.1 13.6 82.7 | -0.095 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt/score_10/ctm.filt.filt.sys

# multi-splice + i-vector + perturbed + rescore
%WER 11.9 | 1155 27512 | 89.9 7.5 2.6 1.8 11.9 77.9 | -0.177 | exp/nnet2_online/nnet_ms_sp/decode_test.rescore/score_10/ctm.filt.filt.sys
%WER 11.8 | 1155 27512 | 90.0 7.4 2.6 1.8 11.8 77.6 | -0.300 | exp/nnet2_online/nnet_ms_sp_online/decode_test.rescore/score_10/ctm.filt.filt.sys
%WER 11.8 | 1155 27512 | 89.9 7.4 2.7 1.8 11.8 79.0 | -0.233 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt_offline.rescore/score_11/ctm.filt.filt.sys
%WER 12.3 | 1155 27512 | 89.5 7.6 3.0 1.8 12.3 80.5 | -0.200 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt.rescore/score_12/ctm.filt.filt.sys

# multi-splice + i-vector + perturbed + sMBR training
%WER 13.2 | 1155 27512 | 88.6 8.5 2.8 1.9 13.2 81.6 | -0.102 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch0_test/score_11/ctm.filt.filt.sys
%WER 12.8 | 1155 27512 | 89.1 8.4 2.5 2.0 12.8 81.1 | -0.108 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch1_test/score_10/ctm.filt.filt.sys
%WER 12.6 | 1155 27512 | 89.3 8.2 2.6 1.9 12.6 81.1 | -0.062 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch2_test/score_11/ctm.filt.filt.sys
%WER 12.5 | 1155 27512 | 89.4 8.1 2.4 1.9 12.5 81.3 | -0.064 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch3_test/score_11/ctm.filt.filt.sys
%WER 12.5 | 1155 27512 | 89.6 8.1 2.2 2.1 12.5 80.8 | -0.107 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch4_test/score_10/ctm.filt.filt.sys

# multi-splice + i-vector + perturbed + sMBR training + rescore
%WER 11.8 | 1155 27512 | 90.0 7.4 2.6 1.8 11.8 77.6 | -0.300 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch0_test.rescore/score_10/ctm.filt.filt.sys
%WER 11.4 | 1155 27512 | 90.4 7.1 2.5 1.7 11.4 76.9 | -0.253 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch1_test.rescore/score_10/ctm.filt.filt.sys
%WER 11.2 | 1155 27512 | 90.6 7.0 2.4 1.7 11.2 76.5 | -0.240 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch2_test.rescore/score_10/ctm.filt.filt.sys
%WER 11.0 | 1155 27512 | 90.7 6.8 2.5 1.7 11.0 75.8 | -0.232 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch3_test.rescore/score_11/ctm.filt.filt.sys
%WER 10.9 | 1155 27512 | 90.7 6.8 2.4 1.7 10.9 75.8 | -0.230 | exp/nnet2_online/nnet_ms_sp_smbr_0.000005/decode_epoch4_test.rescore/score_11/ctm.filt.filt.sys

#---------------------------------(Old LM) Provided for reference-----------------------------------------------
# DEV SPEAKERS:
%WER 36.0 | 507 17792 | 69.7 23.3 7.0 5.7 36.0 98.2 | 0.009 | exp/tri1/decode_dev/score_12/ctm.filt.filt.sys
%WER 32.1 | 507 17792 | 73.8 20.1 6.1 6.0 32.1 97.2 | -0.092 | exp/tri2/decode_dev/score_15/ctm.filt.filt.sys
%WER 27.4 | 507 17792 | 78.3 16.7 5.0 5.7 27.4 96.4 | -0.232 | exp/tri3/decode_dev/score_16/ctm.filt.filt.sys
%WER 24.0 | 507 17792 | 80.6 14.7 4.7 4.6 24.0 94.1 | -0.098 | exp/tri3_mmi_b0.1/decode_dev_it4/score_15/ctm.filt.filt.sys
# Karel's DNN
%WER 21.7 | 507 17792 | 83.3 13.1 3.7 5.0 21.7 93.5 | -0.175 | exp/dnn4_pretrain-dbn_dnn/decode_dev/score_11/ctm.filt.filt.sys
%WER 20.0 | 507 17792 | 84.7 12.1 3.1 4.7 20.0 92.5 | -0.202 | exp/dnn4_pretrain-dbn_dnn_smbr/decode_dev/score_12/ctm.filt.filt.sys
%WER 19.4 | 507 17792 | 85.2 12.0 2.8 4.6 19.4 91.9 | -0.223 | exp/dnn4_pretrain-dbn_dnn_smbr_i1lats/decode_dev/score_13/ctm.filt.filt.sys

# TEST SPEAKERS:
# GMMs
%WER 34.7 | 1155 27512 | 69.9 23.7 6.4 4.6 34.7 97.3 | 0.080 | exp/tri1/decode_test/score_13/ctm.filt.filt.sys
%WER 29.8 | 1155 27512 | 74.8 20.2 5.0 4.6 29.8 95.9 | -0.015 | exp/tri2/decode_test/score_14/ctm.filt.filt.sys
%WER 24.6 | 1155 27512 | 79.6 16.3 4.1 4.2 24.6 93.6 | -0.050 | exp/tri3/decode_test/score_16/ctm.filt.filt.sys
%WER 21.6 | 1155 27512 | 82.3 14.2 3.5 3.9 21.6 91.9 | 0.043 | exp/tri3_mmi_b0.1/decode_test_it4/score_13/ctm.filt.filt.sys
# Karel's DNN
%WER 19.1 | 1155 27512 | 84.4 12.1 3.5 3.5 19.1 90.0 | -0.025 | exp/dnn4_pretrain-dbn_dnn/decode_test/score_12/ctm.filt.filt.sys
%WER 17.7 | 1155 27512 | 85.8 11.3 2.9 3.5 17.7 88.7 | -0.049 | exp/dnn4_pretrain-dbn_dnn_smbr/decode_test/score_13/ctm.filt.filt.sys
%WER 17.2 | 1155 27512 | 86.2 11.0 2.7 3.4 17.2 87.9 | -0.063 | exp/dnn4_pretrain-dbn_dnn_smbr_i1lats/decode_test/score_14/ctm.filt.filt.sys

# multi-splice + i-vector
%WER 17.9 | 1155 27512 | 85.6 11.2 3.1 3.6 17.9 88.1 | -0.105 | exp/nnet2_online/nnet_ms_a/decode_test/score_12/ctm.filt.filt.sys
%WER 17.9 | 1155 27512 | 85.6 11.2 3.1 3.6 17.9 88.1 | -0.187 | exp/nnet2_online/nnet_ms_a_online/decode_test/score_12/ctm.filt.filt.sys
%WER 18.0 | 1155 27512 | 85.3 11.3 3.4 3.3 18.0 87.9 | -0.196 | exp/nnet2_online/nnet_ms_a_online/decode_test_utt/score_13/ctm.filt.filt.sys
%WER 17.7 | 1155 27512 | 85.6 11.1 3.3 3.3 17.7 87.4 | -0.177 | exp/nnet2_online/nnet_ms_a_online/decode_test_utt_offline/score_13/ctm.filt.filt.sys
%WER 17.9 | 1155 27512 | 85.6 11.2 3.1 3.6 17.9 88.1 | -0.187 | exp/nnet2_online/nnet_ms_a_smbr_0.000005/decode_epoch0_test/score_12/ctm.filt.filt.sys
%WER 17.6 | 1155 27512 | 86.2 11.0 2.7 3.9 17.6 87.7 | -0.191 | exp/nnet2_online/nnet_ms_a_smbr_0.000005/decode_epoch1_test/score_12/ctm.filt.filt.sys
%WER 17.3 | 1155 27512 | 86.4 10.9 2.7 3.7 17.3 87.8 | -0.191 | exp/nnet2_online/nnet_ms_a_smbr_0.000005/decode_epoch2_test/score_13/ctm.filt.filt.sys
%WER 17.2 | 1155 27512 | 86.5 10.9 2.6 3.7 17.2 87.6 | -0.190 | exp/nnet2_online/nnet_ms_a_smbr_0.000005/decode_epoch3_test/score_13/ctm.filt.filt.sys
%WER 17.2 | 1155 27512 | 86.6 10.8 2.6 3.8 17.2 87.6 | -0.193 | exp/nnet2_online/nnet_ms_a_smbr_0.000005/decode_epoch4_test/score_13/ctm.filt.filt.sys

# multi-splice + i-vector + perturbed
%WER 17.2 | 1155 27512 | 86.3 10.8 2.9 3.5 17.2 87.1 | -0.126 | exp/nnet2_online/nnet_ms_sp/decode_test/score_12/ctm.filt.filt.sys
%WER 17.2 | 1155 27512 | 86.3 10.8 3.0 3.5 17.2 87.0 | -0.229 | exp/nnet2_online/nnet_ms_sp_online/decode_test/score_12/ctm.filt.filt.sys
%WER 17.6 | 1155 27512 | 85.9 11.1 3.0 3.5 17.6 87.8 | -0.210 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt/score_12/ctm.filt.filt.sys
%WER 17.2 | 1155 27512 | 86.5 10.8 2.7 3.7 17.2 87.4 | -0.236 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt_offline/score_11/ctm.filt.filt.sys

# new dict, lm,
%WER 13.3 | 1155 27512 | 88.7 8.7 2.6 2.0 13.3 81.5 | -0.097 | exp/nnet2_online/nnet_ms_sp/decode_test/score_10/ctm.filt.filt.sys
%WER 13.2 | 1155 27512 | 88.6 8.5 2.8 1.9 13.2 81.6 | -0.102 | exp/nnet2_online/nnet_ms_sp_online/decode_test/score_11/ctm.filt.filt.sys
%WER 13.6 | 1155 27512 | 88.5 8.9 2.6 2.1 13.6 82.7 | -0.095 | exp/nnet2_online/nnet_ms_sp_online/decode_test_utt/score_10/ctm.filt.filt.sys

### LSTM vs. DNN ###
# DNN on MFCC-fMLLR
%WER 19.1 | 1155 27512 | 84.4 12.1 3.5 3.5 19.1 90.0 | -0.025 | exp/dnn4_pretrain-dbn_dnn/decode_test/score_12/ctm.filt.filt.sys
# DNN on FBANK-pitch (we see pitch compensated degradation of not having fMLLR),
%WER 19.2 | 1155 27512 | 84.4 12.3 3.3 3.6 19.2 89.2 | -0.021 | exp/dnn4d-fbank_pretrain-dbn_dnn/decode_test/score_12/ctm.filt.filt.sys
# LSTM
%WER 20.3 | 1155 27512 | 83.3 13.2 3.5 3.6 20.3 90.7 | -0.176 | exp/lstm4f_ClipGradient5_lrate1e-4/decode_test/score_11/ctm.filt.filt.sys
# 2xLSTM
TODO...

