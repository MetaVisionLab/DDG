#/bin/bash
set -e

folder="1"
if [[ $# == 1 ]]; then
  folder=$1
fi

python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains             Cartoon Photo Sketch --target-domains ArtPainting --models model=resnet50_dynamic --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_PACS_resnet50/ArtPainting/$folder
python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting         Photo Sketch --target-domains Cartoon     --models model=resnet50_dynamic --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_PACS_resnet50/Cartoon/$folder
python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting Cartoon       Sketch --target-domains Photo       --models model=resnet50_dynamic --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_PACS_resnet50/Photo/$folder
python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting Cartoon Photo        --target-domains Sketch      --models model=resnet50_dynamic --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_PACS_resnet50/Sketch/$folder
