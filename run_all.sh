#!/usr/bin/env bash
# Menjalankan seluruh studi kasus secara berurutan
set -e
for d in 01-analytics-ai 02-hpc 03-gis 04-machine-learning 05-hci-dataviz 06-deforestation-ai; do
  echo "=== $d ==="
  python "$d/main.py"
done
