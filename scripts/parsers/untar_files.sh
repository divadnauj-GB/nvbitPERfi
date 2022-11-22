#!/bin/bash
set -x
set -e

cd data/ || exit

for file in *.tar.gz; do
  tar xzf "$file"
done
