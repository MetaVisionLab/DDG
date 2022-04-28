#/bin/bash
set -e

folder="1"
if [[ $# == 1 ]]; then
  folder=$1
fi

python ddg/trainer/domain_mix.py --dataset OfficeHome --transform TransformForOfficeHome --source-domains     Clipart Product RealWorld --target-domains Art       --models model=resnet50_dynamic --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_OfficeHome_resnet50/Art/$folder
python ddg/trainer/domain_mix.py --dataset OfficeHome --transform TransformForOfficeHome --source-domains Art         Product RealWorld --target-domains Clipart   --models model=resnet50_dynamic --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_OfficeHome_resnet50/Clipart/$folder
python ddg/trainer/domain_mix.py --dataset OfficeHome --transform TransformForOfficeHome --source-domains Art Clipart         RealWorld --target-domains Product   --models model=resnet50_dynamic --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_OfficeHome_resnet50/Product/$folder
python ddg/trainer/domain_mix.py --dataset OfficeHome --transform TransformForOfficeHome --source-domains Art Clipart Product           --target-domains RealWorld --models model=resnet50_dynamic --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_OfficeHome_resnet50/RealWorld/$folder
