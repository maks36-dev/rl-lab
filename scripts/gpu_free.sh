#!/usr/bin/env bash
# Показывает GPU с загрузкой ниже порога (по умолчанию 20% util и < 4 GB занято).
UTIL_MAX=${1:-20}; MEM_MAX_MB=${2:-4000}
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits |
awk -F', ' -v u=$UTIL_MAX -v m=$MEM_MAX_MB '{ mark=($2<u && $3<m) ? "FREE" : "busy"; printf "GPU %s  util %3s%%  mem %5s/%5s MB  %s\n", $1, $2, $3, $4, mark }'
