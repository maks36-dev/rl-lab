#!/usr/bin/env bash
# Создаёт скелет главы: chapters/chNN_name (пакет Python, поэтому префикс ch), docs/NN_name (публичный results.md)
# и notes/NN_name (личные черновики: theory, journal, tasks, post — вне git; в docs/ попадают на ступени writeup).
# Использование: scripts/new_chapter.sh 01 snn_ppo "Spiking-нейросети как актор PPO"
set -euo pipefail
NUM=$1; SLUG=$2; TITLE=${3:-$SLUG}; NAME="${NUM}_${SLUG}"; PKG="ch${NAME}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
C="$ROOT/chapters/$PKG"; D="$ROOT/docs/$NAME"; N="$ROOT/notes/$NAME"
mkdir -p "$C"/{configs,src,scripts,tests} "$D" "$N"
touch "$C/src/__init__.py"
cat > "$C/README.md" <<MD
# $NUM. $TITLE

**Формат:** полная / мини
**Статус:** не начата

## Цель
## Исследовательский вопрос
## Инфра-новинка главы
## Критерии готовности
- [ ] ...

## Как запустить
\`\`\`
source .venvs/base/bin/activate
python -m chapters.$PKG.src.train
\`\`\`
MD
printf '# %s — results\n\n' "$TITLE" > "$D/results.md"
for f in theory journal tasks post; do
  printf '# %s — %s\n\n' "$TITLE" "$f" > "$N/$f.md"
done
echo "created $C, $D and $N (notes, not in git)"
