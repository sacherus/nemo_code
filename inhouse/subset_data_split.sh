#!/bin/bash
# Copyright 2013  Hong Kong University of Science and Technology (Author: Ricky Chan Ho Yin);
#                 Brno University of Technology (Author: Karel Vesely);
#                 Johns Hopkins University (Author: Daniel Povey);
# Apache 2.0

# Begin configuration.
seed=777 # use seed for speaker shuffling
split=5
# End configuration.

echo "$0 $@"  # Print the command line for logging

uttbase=true; # by default, we choose last 10% utterances for CV

if [ "$1" == "--cv-spk-percent" ]; then
  uttbase=false;
  spkbase=true;
fi

[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [--cv-spk-percent P|--cv-utt-percent P] <srcdir> <traindir> <crossvaldir>"
  echo "  --cv-spk-percent P  Cross Validation portion of the total speakers, recommend value is 10% (i.e. P=10)"
  echo "  --cv-utt-percent P  Cross Validation portion of the total utterances, default is 10% (i.e. P=10)"
  echo "  "
  exit 1;
fi

srcdir=$1
trndir=$2
cvdir=$3

## use random chosen P% speakers for CV
if [ ! -f $srcdir/spk2utt ]; then
  echo "$0: no such file $srcdir/spk2utt" 
  exit 1;
fi

#total, cv, train number of speakers
N=$(cat $srcdir/spk2utt | wc -l)
#N_spk_cv=$((N * cv_spk_percent / 100))
#N_spk_trn=$((N - N_spk_cv))

mkdir -p $cvdir $trndir

#shuffle the speaker list
awk '{print $1}' $srcdir/spk2utt | shuffle_list.pl --srand $seed > $trndir/_tmpf_randspk

echo $N
echo $((N/split))

i=1
rm -f trndir/xxx*
split -l$((N/split+1)) $trndir/_tmpf_randspk $trndir/xxx
read_xxx=`ls $trndir/xxx* | sort`
for f in $read_xxx; do
	echo "cat `ls $trndir/xxx* | grep -v $f` > $f.tr"
	mkdir -p $cvdir.$i $trndir.$i
	cat `echo "$read_xxx" | grep -v $f` > $f.tr
	subset_data_dir.sh --spk-list $f.tr $srcdir $trndir.$i
	subset_data_dir.sh --spk-list $f $srcdir $cvdir.$i
	i=$((i+1))
done

#clean-up
rm -f $trndir/_tmpf_randspk $trndir/xxx* $trndir/_tmpf_trainspk $cvdir/_tmpf_cvspk
