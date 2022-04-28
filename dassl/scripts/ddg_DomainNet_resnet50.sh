#/bin/bash
set -e

folder="1"
if [[ $# == 1 ]]; then
  folder=$1
fi

python train.py --root ../../Dassl.data --trainer DomainMix --source-domains         infograph painting quickdraw real sketch --target-domains clipart   --dataset-config-file configs/datasets/ddg_DomainNet_resnet50.yaml --config-file configs/trainers/DomainNet.yaml --output-dir ./result/ddg_DomainNet_resnet50/clipart/$folder
python train.py --root ../../Dassl.data --trainer DomainMix --source-domains clipart           painting quickdraw real sketch --target-domains infograph --dataset-config-file configs/datasets/ddg_DomainNet_resnet50.yaml --config-file configs/trainers/DomainNet.yaml --output-dir ./result/ddg_DomainNet_resnet50/infograph/$folder
python train.py --root ../../Dassl.data --trainer DomainMix --source-domains clipart infograph          quickdraw real sketch --target-domains painting  --dataset-config-file configs/datasets/ddg_DomainNet_resnet50.yaml --config-file configs/trainers/DomainNet.yaml --output-dir ./result/ddg_DomainNet_resnet50/painting/$folder
python train.py --root ../../Dassl.data --trainer DomainMix --source-domains clipart infograph painting           real sketch --target-domains quickdraw --dataset-config-file configs/datasets/ddg_DomainNet_resnet50.yaml --config-file configs/trainers/DomainNet.yaml --output-dir ./result/ddg_DomainNet_resnet50/quickdraw/$folder
python train.py --root ../../Dassl.data --trainer DomainMix --source-domains clipart infograph painting quickdraw      sketch --target-domains real      --dataset-config-file configs/datasets/ddg_DomainNet_resnet50.yaml --config-file configs/trainers/DomainNet.yaml --output-dir ./result/ddg_DomainNet_resnet50/real/$folder
python train.py --root ../../Dassl.data --trainer DomainMix --source-domains clipart infograph painting quickdraw real        --target-domains sketch    --dataset-config-file configs/datasets/ddg_DomainNet_resnet50.yaml --config-file configs/trainers/DomainNet.yaml --output-dir ./result/ddg_DomainNet_resnet50/sketch/$folder
