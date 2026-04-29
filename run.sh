#!/usr/bin/env bash
set -e

# sciknoweval (4 domain)
for d in biology chemistry material physics; do
  python data/preprocess.py --data_source datasets/sciknoweval/$d
done

# tooluse
python data/preprocess.py --data_source datasets/tooluse