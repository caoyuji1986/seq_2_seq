#!/bin/bash

. `dirname $0`/../common/vars

src=de
tgt=en
pair=$src-$tgt

tok() {
  path=$1
  name=$2
  cat=$3
  for lang in $src $tgt; do
    $cat $path.$lang | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $moses_scripts/tokenizer/tokenizer.perl -a -l $lang  \
    >> corpus.tok.$lang
  done
  lines=`wc -l $path.$src | cut -d' ' -f 1`
  yes $name | head -n $lines >> corpus.tok.domain
}

rm -f corpus.tok.*

# extract paracrawl
for lang in $src $tgt; do
  tar  xzvOf  $pc_dir/paracrawl-release1.$tgt-$src.zipporah0-dedup-clean.tgz\
       paracrawl-release1.$tgt-$src.zipporah0-dedup-clean.$lang > paracrawl.$lang
done

# Tokenise
tok paracrawl PARACRAWL cat
tok $rapid_dir/rapid2016.$pair RAPID cat
tok $ep_dir/europarl-v7.$pair EUROPARL cat
tok $nc_dir/news-commentary-v13.$pair NEWSCOMM cat
tok $cc_dir/commoncrawl.$pair COMMONCRAWL zcat



#
###
#### Clean
`dirname $0`/../common/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 1 $max_len corpus.retained
###
#
#### Train truecaser and truecase
for lang in $src $tgt; do
  $moses_scripts/recaser/train-truecaser.perl -model truecase-model.$lang -corpus corpus.tok.$lang
  $moses_scripts/recaser/truecase.perl < corpus.clean.$lang > corpus.tc.$lang -model truecase-model.$lang
done
#

# dev sets
for devset in test2014 test2015 test2016 test2017 ; do
  for lang  in $src $tgt; do
    side="ref"
    if [ $lang == $tgt ]; then
      side="src"
    fi
    $moses_scripts/ems/support/input-from-sgm.perl < $dev_dir/news$devset-$tgt$src-$side.$lang.sgm | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
    $moses_scripts/recaser/truecase.perl   -model truecase-model.$lang \
    > news$devset.tc.$lang

  done
  cp $dev_dir/news$devset-$src$tgt*sgm .
  cp $dev_dir/news$devset-$tgt$src*sgm .
done


## Tidy up and compress
paste corpus.tc.$src corpus.tc.$tgt corpus.clean.domain | gzip -c > corpus.gz
for lang in $src $tgt; do
  rm -f corpus.tc.$lang corpus.tok.$lang corpus.clean.$lang corpus.retained paracrawl.$lang
done
rm corpus.clean.domain corpus.tok.domain
tar zcvf dev.tgz news* &&  rm news*
tar zcvf true.tgz truecase-model.* && rm truecase-model*