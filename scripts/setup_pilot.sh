#!/usr/bin/env bash
# Set up an ISOLATED environment for one TopoScout portability pilot (M7B).
# Never touches the main project environment or any research environment.
#
#   bash scripts/setup_pilot.sh satellite|materials|microscopy|pathology|industrial
#
# After setup, run the pilot with the printed interpreter, e.g.:
#   .pilot-envs/satellite/bin/python scripts/pilot_demo.py run satellite
set -euo pipefail

DOMAIN="${1:-}"
case "$DOMAIN" in
  satellite)  PKGS="tensorflow-cpu keras huggingface_hub opencv-python-headless scikit-image scipy pillow numpy" ;;
  materials)  PKGS="torch --extra-index-url https://download.pytorch.org/whl/cpu segmentation-models-pytorch huggingface_hub opencv-python-headless scikit-image scipy pillow numpy" ;;
  microscopy) PKGS="cellpose opencv-python-headless scikit-image scipy pillow numpy" ;;
  pathology)  PKGS="tiatoolbox opencv-python-headless scikit-image scipy pillow numpy" ;;
  industrial) PKGS="anomalib opencv-python-headless scikit-image scipy pillow numpy" ;;
  *) echo "usage: bash scripts/setup_pilot.sh {satellite|materials|microscopy|pathology|industrial}" >&2; exit 2 ;;
esac

ENVDIR=".pilot-envs/$DOMAIN"
echo "creating isolated env: $ENVDIR"
python3 -m venv "$ENVDIR"
"$ENVDIR/bin/pip" install --quiet --upgrade pip
# shellcheck disable=SC2086
"$ENVDIR/bin/pip" install --quiet $PKGS

if [ "$DOMAIN" = "industrial" ]; then
  cat <<'EOF'

industrial extra step (dataset NOT redistributed with this repository):
  MVTec AD is CC BY-NC-SA — obtain it from MVTec's official site, then:
    export TOPOSCOUT_MVTEC_ROOT=/path/to/mvtec_root   # containing MVTecAD/bottle/...
  PatchCore fits its memory bank from the category's normal images (~30 min CPU).
EOF
fi

echo
echo "done. run the pilot with:"
echo "  $ENVDIR/bin/python scripts/pilot_demo.py run $DOMAIN"
