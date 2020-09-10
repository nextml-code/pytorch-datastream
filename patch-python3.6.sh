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
