#/bin/bash
set -e

folder="1"
if [[ $# == 1 ]]; then
  folder=$1
fi

python train.py --root ../../Dassl.data --trainer DomainMix --source-domains     clipart product real_world --target-domains art        --dataset-config-file configs/datasets/ddg_OfficeHome_resnet50.yaml --config-file configs/trainers/OfficeHome.yaml --output-dir ./result/ddg_OfficeHome_resnet50/art/$folder
python train.py --root ../../Dassl.data --trainer DomainMix --source-domains art         product real_world --target-domains clipart    --dataset-config-file configs/datasets/ddg_OfficeHome_resnet50.yaml --config-file configs/trainers/OfficeHome.yaml --output-dir ./result/ddg_OfficeHome_resnet50/clipart/$folder
python train.py --root ../../Dassl.data --trainer DomainMix --source-domains art clipart         real_world --target-domains product    --dataset-config-file configs/datasets/ddg_OfficeHome_resnet50.yaml --config-file configs/trainers/OfficeHome.yaml --output-dir ./result/ddg_OfficeHome_resnet50/product/$folder
python train.py --root ../../Dassl.data --trainer DomainMix --source-domains art clipart product            --target-domains real_world --dataset-config-file configs/datasets/ddg_OfficeHome_resnet50.yaml --config-file configs/trainers/OfficeHome.yaml --output-dir ./result/ddg_OfficeHome_resnet50/real_world/$folder
