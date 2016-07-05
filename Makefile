#TODO:
#1. Mix validate & train
#2. Leave test
#3. 

include common.mk 

.SECONDEXPANSION:

all: mfcc.done

dir/%:
	@mkdir -p $*

p-%:
	@echo $($*)

SHELL:=/bin/bash

WORK_DIR=work
#Features selection
DATA_DIR=$(WORK_DIR)/data
TRAIN_DIR=$(DATA_DIR)/train
DEV_DIR=$(DATA_DIR)/dev
TEST_DIR=$(DATA_DIR)/test

LOCAL_DIR=$(DATA_DIR)/local

DATA_DIRS=$(TRAIN_DIR) $(DEV_DIR) $(TEST_DIR)

NJ=12 
DEC_NJ=8
DECODE_NJ=$(DEC_NJ)

TRAIN_CMD=run.pl
DEC_CMD=run.pl
DECODE_CMD=$(DEC_CMD)

$(DATA_DIR)/data.done: | dir/$$(@D)
	local/prepare_data.sh $(@D) 
	touch $@

$(LOCAL_DIR)/dict.done: $(DATA_DIR)/data.done | dir/$$(@D)
	local/prepare_dict.sh $(@D)
	touch $@

lang.done: $(LOCAL_DIR)/lang.done
$(LOCAL_DIR)/lang.done: $(LOCAL_DIR)/dict.done | dir/$$(@D)
	utils/prepare_lang.sh $(LOCAL_DIR)/dict_nosp \
	"<UNK>" $(LOCAL_DIR)/lang_nosp $(DATA_DIR)/lang_nosp
	touch $@

$(DATA_DIR)/lm.done: $(LOCAL_DIR)/lang.done | dir/$$(@D)
	local/prepare_lm.sh $(@D) touch $@


######Feats######

FEAT_NAME=mfcc

mfcc.done: $(DATA_DIR)/mfcc.done
$(DATA_DIR)/mfcc.done: $(DATA_DIR)/lm.done | dir/$$(@D)
	for dir in $(DATA_DIRS); do \
	  steps/make_mfcc.sh --nj $(NJ) --cmd "$(TRAIN_CMD)" $$dir $$dir/log $$dir/data; \
	  steps/compute_cmvn_stats.sh $$dir $$dir/log $$dir/data; \
	done
	touch $@


######Feats######
# Train

EXP_DIR=exp
SHORT_TRAIN_DIR=$(TRAIN_DIR)_10kshort
SHORT_TRAIN_NODUP_DIR=$(TRAIN_DIR)_10kshort_nodup

MONO_DIR=$(EXP_DIR)/mono0a
MONO_ALI_DIR=$(MONO_DIR)_ali

short.done: $(MONO_DIR)/short.done
$(MONO_DIR)/short.done: | dir/$$(@D)
	utils/subset_data_dir.sh --shortest $(TRAIN_DIR) 10000 $(SHORT_TRAIN_DIR)
	local/remove_dup_utts.sh 10 $(SHORT_TRAIN_DIR) $(SHORT_TRAIN_NODUP_DIR)
	touch $@

LANG_DIR=$(DATA_DIR)/lang_nosp
LANG_CLEAN_DIR=$(DATA_DIR)/lang
$(MONO_DIR)/mono.done: $(MONO_DIR)/short.done | dir/$$(@D)
	steps/train_mono.sh --nj $(NJ) --cmd "$(TRAIN_CMD)" \
	$(SHORT_TRAIN_NODUP_DIR) $(LANG_DIR) $(MONO_DIR); 
	steps/align_si.sh --nj $(NJ) --cmd "$(TRAIN_CMD)" \
	  $(TRAIN_DIR) $(LANG_DIR) $(MONO_DIR) $(MONO_ALI_DIR); 

TRI1_DIR=$(EXP_DIR)/tri1
TEST_LANG_DIR=$(LANG_DIR)_test
TRI1_GRAPH_DIR=$(TRI1_DIR)/graph_nosp
TRI1_DEC_TEST_DIR=$(TRI1_DIR)/decode_nosp_test
TRI1_DEC_DEV_DIR=$(TRI1_DIR)/decode_nosp_dev

deltas.test.decode.done: $(TRI1_DIR)/deltas.test.decode.done
deltas.dev.decode.done: $(TRI1_DIR)/deltas.dev.decode.done
deltas.train.done: $(TRI1_DIR)/deltas.train.done

$(TRI1_DIR)/deltas.train.done: $(MONO_DIR)/mono.done | dir/$$(@D)
	steps/train_deltas.sh --cmd "$(TRAIN_CMD)" \
	  2500 30000 $(TRAIN_DIR) $(LANG_DIR) $(MONO_ALI_DIR) $(TRI1_DIR)
	touch $@
	
$(TRI1_DIR)/deltas.graph.done: | dir/$$(@D)
	utils/mkgraph.sh $(TEST_LANG_DIR) $(TRI1_DIR) $(TRI1_GRAPH_DIR)
	touch $@

TRI1_ALI_DIR=$(TRI1_DIR)_ali
ali.done: $(TRI1_DIR)/ali.done
$(TRI1_ALI_DIR)/ali.done: | dir/$$(@D)
	steps/align_si.sh --nj $(NJ) --cmd "$(TRAIN_CMD)" \
	      $(TRAIN_DIR) $(LANG_DIR) $(TRI1_DIR) $(TRI1_ALI_DIR)
	touch $@

$(TRI1_DIR)/deltas.dev.decode.done: | dir/$$(@D)
	steps/decode.sh --nj 1 --cmd "$(DEC_CMD)" --num-threads 12 \
	  $(TRI1_GRAPH_DIR) $(DEV_DIR) $(TRI1_DEC_DEV_DIR)
	touch $@
	
$(TRI1_DIR)/deltas.test.decode.done: | dir/$$(@D)
	steps/decode.sh --nj 1 --cmd "$(DEC_CMD)" \
	  --num-threads 12 \
	  $(TRI1_GRAPH_DIR) $(TEST_DIR) $(TRI1_DEC_TEST_DIR)
	touch $@

TRI2_DIR=$(EXP_DIR)/tri2
TEST_LANG_DIR=$(LANG_DIR)_test
TRI2_GRAPH_DIR=$(TRI2_DIR)/graph_nosp
TRI2_DEC_TEST_DIR=$(TRI2_DIR)/decode_nosp_test
TRI2_DEC_DEV_DIR=$(TRI2_DIR)/decode_nosp_dev

lda.test.decode.done: $(TRI2_DIR)/lda.test.decode.done
lda.dev.decode.done: $(TRI2_DIR)/lda.dev.decode.done
lda.train.done: $(TRI2_DIR)/lda.train.done

$(TRI2_DIR)/lda.train.done: $(TRI1_DIR)/deltas.train.done| dir/$$(@D)
	steps/train_lda_mllt.sh --cmd "$(TRAIN_CMD)" \
	  5000 50000 $(TRAIN_DIR) $(LANG_DIR) $(TRI1_ALI_DIR) $(TRI2_DIR)
	touch $@
	
$(TRI2_DIR)/lda.graph.done: $(TRI2_DIR)/lda.train.done | dir/$$(@D)
	utils/mkgraph.sh $(TEST_LANG_DIR) $(TRI2_DIR) $(TRI2_GRAPH_DIR)
	touch $@

THREADS=12
TRI2_ALI_DIR=$(TRI2_DIR)_ali
ali.done: $(TRI2_DIR)/ali.done
$(TRI2_ALI_DIR)/ali.done: $(TRI2_DIR)/lda.graph.done | dir/$$(@D)
	steps/align_si.sh --nj $(NJ) --cmd "$(TRAIN_CMD)" \
	      $(TRAIN_DIR) $(LANG_DIR) $(TRI2_DIR) $(TRI2_ALI_DIR)
	touch $@

$(TRI2_DIR)/lda.dev.decode.done: $(TRI2_DIR)/ali.done | dir/$$(@D)
	steps/decode.sh --nj 1 --cmd "$(DEC_CMD)" --num-threads $(THREADS) \
	$(TRI2_GRAPH_DIR) $(DEV_DIR) $(TRI2_DEC_DEV_DIR)
	touch $@
	
$(TRI2_DIR)/lda.test.decode.done: $(TRI2_DIR)/lda.dev.decode.done | dir/$$(@D)
	steps/decode.sh --nj 1 --cmd "$(DEC_CMD)" \
	  --num-threads $(THREADS) \
	  $(TRI2_GRAPH_DIR) $(TEST_DIR) $(TRI2_DEC_TEST_DIR)
	touch $@



#Subset data dir
SEED=777

SPLITS=5
SPLIT=1

HID_DIM=1024
NN_DEPTH=2

#HID_DIM=2048
#NN_DEPTH=6

D_ORDER=0

CV_TRAIN=20
CV_HELD=$(shell perl -e 'print 100-$(CV_TRAIN)')
CV_HELD=20

EXP_DATA=split_$(SPLIT)
EXP_DATA=less_$(CV_TRAIN)

SPLIT_TYPE=less_$(CV_TRAIN)

DATA_LDA_DIR=data-lda
DATA_FBANK_DIR=data-fbank
DATA_TRANS_DIR=$(DATA_LDA_DIR)
DATA_TRANS_DIR=$(DATA_FBANK_DIR)

NNET_INIT_DIR=$(WORK_DIR)/nnet/$(DATA_TRANS_DIR)
NNET_DIR=$(NNET_INIT_DIR)/$(EXP_DATA)
NNET_SPLIT_DIR=$(NNET_INIT_DIR)/$(SPLIT_TYPE)

DBN_PRE_DIR=$(NNET_DIR)/d_order_$(D_ORDER)/depth_$(NN_DEPTH)/dim_$(HID_DIM)
PRETRAIN_TYPE=dnn_dbn
PRETRAIN_TYPE=no_pre
PRETRAIN_TYPE=lstm
PRETRAIN_TYPE=auto
PRETRAIN_TYPE=cnn
DBN_DIR=$(DBN_PRE_DIR)/$(PRETRAIN_TYPE)

NNET_TRAIN_CLEAN_DIR=$(NNET_DIR)/train_tr
NNET_CV_CLEAN_DIR=$(NNET_DIR)/train_cv

NNET_TRAIN_DIR=$(NNET_DIR)/train_tr.$(SPLIT)
NNET_CV_DIR=$(NNET_DIR)/train_cv.$(SPLIT)

split.subset.done: $(NNET_INIT_DIR)/$(EXP_DATA)/split.subset.done
$(NNET_INIT_DIR)/split_$(SPLIT)/split.subset.done: | dir/$$(@D)
	inhouse/subset_data_split.sh --seed $(SEED) --split $(SPLITS) $(TRAIN_DIR) $(NNET_TRAIN_CLEAN_DIR) $(NNET_CV_CLEAN_DIR) 
	touch $@

$(NNET_INIT_DIR)/less_$(CV_TRAIN)/split.subset.done: | dir/$$(@D)
	inhouse/subset_data_reduce.sh --cv-spk-percent $(CV_HELD) --tr-spk-percent $(CV_TRAIN) --seed $(SEED) $(TRAIN_DIR) $(NNET_TRAIN_DIR) $(NNET_CV_DIR) 
	touch $@

$(WORK_DIR)/nnet/$(DATA_LDA_DIR)/less_$(CV_TRAIN)/split.subset.done: | dir/$$(@D)
	inhouse/subset_data_reduce.sh --cv-spk-percent $(CV_HELD) --tr-spk-percent $(CV_TRAIN) --seed $(SEED) $(DATA_LDA_DIR)/train $(NNET_TRAIN_DIR) $(NNET_CV_DIR) 
	touch $@

$(WORK_DIR)/nnet/$(DATA_FBANK_DIR)/less_$(CV_TRAIN)/split.subset.done: | dir/$$(@D)
	inhouse/subset_data_reduce.sh --cv-spk-percent $(CV_HELD) --tr-spk-percent $(CV_TRAIN) --seed $(SEED) $(DATA_FBANK_DIR)/train $(NNET_TRAIN_DIR) $(NNET_CV_DIR) 
	touch $@

CUDA_CMD=run.pl
dbn.done: $(DBN_DIR)/dbn.done

FEATURE_TRANSFORM_DBN=$(DBN_DIR)/final.feature_transform 
DBN=$(DBN_DIR)/$(NN_DEPTH).dbn

dbn.done: $(DBN_DIR)/dbn.done
$(DBN_PRE_DIR)/dnn_dbn/dbn.done: $(NNET_SPLIT_DIR)/split.subset.done | dir/$$(@D)
	  $(CUDA_CMD) $(DBN_DIR)/log/pretrain_dbn.log steps/nnet/pretrain_dbn.sh --delta-opts "--delta-order=$(D_ORDER)" --hid_dim $(HID_DIM) --rbm-iter 1 --nn_depth $(NN_DEPTH) $(NNET_TRAIN_DIR) $@
	  touch $@

auto.done: $(DBN_PRE_DIR)/auto/dbn.done
$(DBN_PRE_DIR)/auto/dbn.done: $(NNET_SPLIT_DIR)/split.subset.done | dir/$$(@D)
	#labels="add-deltas --delta-order=$(D_ORDER) scp:$(NNET_TRAIN_DIR)/feats.scp ark:- | steps/nnet/train.sh --left-context 5 --right-context 5 ark:- ark:- | ark:feat-to-post ark:- ark:- | "; 
	#steps/nnet/train.sh --delta-opts "--delta-order=$(D_ORDER)" --hid-layers $(NN_DEPTH) --hid-dim $(HID_DIM) --learn-rate 0.00001
	labels_tr="ark:feat-to-post scp:$(NNET_TRAIN_DIR)/feats.scp ark:- |"; \
	labels_cv="ark:feat-to-post scp:$(NNET_CV_DIR)/feats.scp ark:- |"; \
	steps/nnet/train.sh --hid-layers $(NN_DEPTH) --hid-dim $(HID_DIM) --learn-rate 0.00001 \
	--labels_tr "$$labels_tr" --labels_cv "$$labels_cv" --num-tgt 40 --train-tool "nnet-train-frmshuff --objective-function=mse" \
	--proto-opts "--no-softmax --activation-type=<Tanh> --hid-bias-mean=0.0 --hid-bias-range=1.0 --param-stddev-factor=0.01" \
	$(NNET_TRAIN_DIR) $(NNET_CV_DIR) dummy-dir dummy-dir dummy-dir $(@D)
	#touch $@

$(DBN_PRE_DIR)/no_pre/dbn.done: $(NNET_SPLIT_DIR)/split.subset.done | dir/$$(@D)
	  touch $@

$(DBN_PRE_DIR)/cnn/dbn.done: $(NNET_SPLIT_DIR)/split.subset.done | dir/$$(@D)
	  touch $@

GMM_DIR=$(TRI2_DIR)
ALI_NAME=tri2_ali
ALI_DIR=$(EXP_DIR)/$(ALI_NAME)
DBN_DNN_DIR=$(DBN_DIR)/$(ALI_NAME)

#ifeq ($(PRETRAIN_TYPE),gcc)
#	$(CC) -o foo $(objects) $(libs_for_gcc)
#else
#	$(CC) -o foo $(objects) $(normal_libs)
#endif

FEATURE_TRANSFORM_DNN=$(DBN_DIR)/final.feature_transform 
dbn.train.done: $(DBN_DNN_DIR)/dbn.train.done
$(DBN_PRE_DIR)/dnn_dbn/$(ALI_NAME)/dbn.train.done: $(DBN_DIR)/dbn.done $(ALI_DIR)/ali.done | dir/$$(@D)
	steps/nnet/train.sh --dbn $(DBN) --hid-layers 0 --learn-rate 0.008 --feature-transform $(FEATURE_TRANSFORM_DNN) $(NNET_TRAIN_DIR) $(NNET_CV_DIR) $(LANG_DIR) $(ALI_DIR) $(ALI_DIR) $(DBN_DNN_DIR)
	touch $@

no_pre.done: $(DBN_PRE_DIR)/no_pre/dbn.train.done
$(DBN_PRE_DIR)/no_pre/$(ALI_NAME)/dbn.train.done: $(ALI_DIR)/ali.done
	steps/nnet/train.sh --hid-layers $(NN_DEPTH) --learn-rate 0.008 $(NNET_TRAIN_DIR) $(NNET_CV_DIR) $(LANG_DIR) $(ALI_DIR) $(ALI_DIR) $(@D)
	touch $@

auto.train.done: $(DBN_PRE_DIR)/no_pre/$(ALI_NAME)/dbn.train.done
$(DBN_PRE_DIR)/no_pre/$(ALI_NAME)/dbn.train.done: $(ALI_DIR)/ali.done
	steps/nnet/train.sh --hid-layers $(NN_DEPTH) --learn-rate 0.008 $(NNET_TRAIN_DIR) $(NNET_CV_DIR) $(LANG_DIR) $(ALI_DIR) $(ALI_DIR) $(@D)
	touch $@

cnn.train.done: $(DBN_PRE_DIR)/cnn/$(ALI_NAME)/dbn.train.done
$(DBN_PRE_DIR)/cnn/$(ALI_NAME)/dbn.train.done: $(ALI_DIR)/ali.done
	steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" \
      --delta-opts "--delta-order=2" --splice 5 \
      --network-type cnn1d --cnn-proto-opts "--patch-dim1 8 --pitch-dim 3" \
      --hid-layers $(NN_DEPTH) --learn-rate 0.008 \
      $(NNET_TRAIN_DIR) $(NNET_CV_DIR) data/lang $(ALI_DIR) $(ALI_DIR) $(@D) 
	touch $@

lstm: $(DBN_PRE_DIR)/lstm/$(ALI_NAME)/dbn.train.done
$(DBN_PRE_DIR)/lstm/$(ALI_NAME)/dbn.train.done: $(ALI_DIR)/ali.done
	steps/nnet/train.sh --network-type lstm --learn-rate 0.0001 \
	--cmvn-opts "--norm-means=true --norm-vars=true" --feat-type plain --splice 0 \
	--scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
	--train-tool "nnet-train-lstm-streams" \
	--train-tool-opts "--num-stream=4 --targets-delay=5" \
	--proto-opts "--num-cells 512 --num-recurrent 200 --num-layers 2 --clip-gradient 5.0" \
	$(NNET_TRAIN_DIR) $(NNET_CV_DIR) $(LANG_DIR) $(ALI_DIR) $(ALI_DIR) $(@D)
	touch $@

LOCAL_DICT_DIR=$(LOCAL_DIR)/dict
lang2.done: $(DATA_DIR)/lang2.done
$(DATA_DIR)/lang2.done:
	steps/get_prons.sh --cmd "$(TRAIN_CMD)" $(TRAIN_DIR) $(LANG_DIR) $(TRI2_DIR)
	utils/dict_dir_add_pronprobs.sh --max-normalize true \
	$(LOCAL_DIR)/dict_nosp $(TRI2_DIR)/pron_counts_nowb.txt $(TRI2_DIR)/sil_counts_nowb.txt $(TRI2_DIR)/pron_bigram_counts_nowb.txt $(LOCAL_DICT_DIR)
	utils/prepare_lang.sh $(LOCAL_DICT_DIR) "<unk>" $(DATA_DIR)/local/lang $(DATA_DIR)/lang
	cp -rT $(DATA_DIR)/lang $(DATA_DIR)/lang_test
	cp -rT $(DATA_DIR)/lang $(DATA_DIR)/lang_rescore
	cp $(DATA_DIR)/lang_nosp_test/G.fst $(DATA_DIR)/lang_test
	cp $(DATA_DIR)/lang_nosp_rescore/G.carpa $(DATA_DIR)/lang_rescore
	utils/mkgraph.sh $(DATA_DIR)/lang_test exp/tri2 exp/tri2/graph 
	touch $@

DECODE_NNET_NJ=2
GRAPH_TYPE=graph_nosp
GRAPH_TYPE=graph
decode.dev.dnn.done: $(DBN_DNN_DIR)/$(GRAPH_TYPE)_decode_dev/decode.dev.dnn.done 
$(DBN_DNN_DIR)/$(GRAPH_TYPE)_decode_dev/decode.dev.dnn.done: $(DBN_DNN_DIR)/dbn.train.done
	steps/nnet/decode.sh --nj $(DECODE_NNET_NJ) --cmd "$(DECODE_CMD)" --config conf/decode_dnn.config --acwt 0.1 $(GMM_DIR)/$(GRAPH_TYPE) $(DATA_TRANS_DIR)/dev $(@D)
	touch $@

decode.test.dnn.done: $(DBN_DNN_DIR)/$(GRAPH_TYPE)_decode_test/decode.test.dnn.done
$(DBN_DNN_DIR)/$(GRAPH_TYPE)_decode_test/decode.test.dnn.done: $(DBN_DNN_DIR)/$(GRAPH_TYPE)_decode_dev/decode.dev.dnn.done
	steps/nnet/decode.sh --nj $(DECODE_NNET_NJ) --cmd "$(DECODE_CMD)" --config conf/decode_dnn.config --acwt 0.1 $(GMM_DIR)/$(GRAPH_TYPE) $(DATA_TRANS_DIR)/test $(@D)
	touch $@


	


MAT_LDA_GMM_DIR=$(TRI2_ALI_DIR)
lda.feats.done: $(DATA_LDA_DIR)/lda.feats.done
$(DATA_LDA_DIR)/lda.feats.done:
	for dir in $(DATA_DIRS); do \
	name=$(DATA_LDA_DIR)/`basename $$dir`; \
	inhouse/make_lda_feats.sh --nj $(DECODE_NNET_NJ) $$name $$dir $(MAT_LDA_GMM_DIR) $$name/log $$name/data; done
	touch $@

fbank.feats.done: $(DATA_FBANK_DIR)/fbank.feats.done
$(DATA_FBANK_DIR)/fbank.feats.done:
	for source_dir in $(DATA_DIRS); do \
	target_dir=$(@D)/`basename $$source_dir`; \
	utils/copy_data_dir.sh $$source_dir $$target_dir;  rm $$target_dir/{cmvn,feats}.scp; \
	steps/make_fbank_pitch.sh --nj $(DECODE_NNET_NJ) $$target_dir $$target_dir/log $$target_dir/data; \
	steps/compute_cmvn_stats.sh $$target_dir $$target_dir/log $$target_dir/data; done
	touch $@
