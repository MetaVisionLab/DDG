#/bin/bash
set -e

folder="1"
if [[ $# == 1 ]]; then
  folder=$1
fi

python train.py --root ../../Dassl.data --trainer Vanilla --source-domains              cartoon photo sketch --target-domains art_painting --dataset-config-file configs/datasets/ddg_with_mixstyle_PACS_resnet50.yaml --config-file configs/trainers/PACS.yaml --output-dir ./result/ddg_with_mixstyle_PACS_resnet50/art_painting/$folder
python train.py --root ../../Dassl.data --trainer Vanilla --source-domains art_painting         photo sketch --target-domains cartoon      --dataset-config-file configs/datasets/ddg_with_mixstyle_PACS_resnet50.yaml --config-file configs/trainers/PACS.yaml --output-dir ./result/ddg_with_mixstyle_PACS_resnet50/cartoon/$folder
python train.py --root ../../Dassl.data --trainer Vanilla --source-domains art_painting cartoon       sketch --target-domains photo        --dataset-config-file configs/datasets/ddg_with_mixstyle_PACS_resnet50.yaml --config-file configs/trainers/PACS.yaml --output-dir ./result/ddg_with_mixstyle_PACS_resnet50/photo/$folder
python train.py --root ../../Dassl.data --trainer Vanilla --source-domains art_painting cartoon photo        --target-domains sketch       --dataset-config-file configs/datasets/ddg_with_mixstyle_PACS_resnet50.yaml --config-file configs/trainers/PACS.yaml --output-dir ./result/ddg_with_mixstyle_PACS_resnet50/sketch/$folder
