#!/usr/bin/env bash
# ./patch-python3.6.sh
set -o errexit -o nounset -o pipefail -o xtrace

command -v strip-hints

for f in $(find datastream/ -name '*.py');
do
    sed -i 's/#.*//g' $f;
    strip-hints $f --to-empty --strip-nl > /tmp/.stripped_hints;
    mv /tmp/.stripped_hints $f;
    sed -i '/from __future__ import annotations/d' $f;
    sed -i 's/Generic\(\[[A-Z]\]\)\?\(\s*,\s*\)\?//g' $f;
    sed -i 's/#//g' $f;
done

sed -i 's/Dataset\[[A-Z]\]/Dataset/g' datastream/datastream.py;

sed -i -E 's/(Programming *Language *:: *Python *:: *)3.7/\13.6/' setup.cfg
sed -i '/Programming Language :: Python :: 3.8/d' setup.cfg

sed -i 's/python/python3.6/g' publish.sh
sed -i -E 's/(VERSION_TAG=)(.*)(")$/\1\2+python3.6\3/' publish.sh
sed -i '/test -z.*exit 2/d' publish.sh
sed -i '/test -z.*exit 3/d' publish.sh
