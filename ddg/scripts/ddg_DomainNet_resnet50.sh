#/bin/bash
set -e

folder="1"
if [[ $# == 1 ]]; then
  folder=$1
fi

python ddg/trainer/domain_mix.py --dataset DomainNet --sampler RandomDomainSampler --transform TransformForDomainNet --source-domains         Infograph Painting Quickdraw Real Sketch --target-domains Clipart   --models model=resnet50_dynamic --epochs 15 --batch-size 64 --optimizer name=SGD lr=0.002 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_DomainNet_resnet50/Clipart/$folder
python ddg/trainer/domain_mix.py --dataset DomainNet --sampler RandomDomainSampler --transform TransformForDomainNet --source-domains Clipart           Painting Quickdraw Real Sketch --target-domains Infograph --models model=resnet50_dynamic --epochs 15 --batch-size 64 --optimizer name=SGD lr=0.002 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_DomainNet_resnet50/Infograph/$folder
python ddg/trainer/domain_mix.py --dataset DomainNet --sampler RandomDomainSampler --transform TransformForDomainNet --source-domains Clipart Infograph          Quickdraw Real Sketch --target-domains Painting  --models model=resnet50_dynamic --epochs 15 --batch-size 64 --optimizer name=SGD lr=0.002 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_DomainNet_resnet50/Painting/$folder
python ddg/trainer/domain_mix.py --dataset DomainNet --sampler RandomDomainSampler --transform TransformForDomainNet --source-domains Clipart Infograph Painting           Real Sketch --target-domains Quickdraw --models model=resnet50_dynamic --epochs 15 --batch-size 64 --optimizer name=SGD lr=0.002 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_DomainNet_resnet50/Quickdraw/$folder
python ddg/trainer/domain_mix.py --dataset DomainNet --sampler RandomDomainSampler --transform TransformForDomainNet --source-domains Clipart Infograph Painting Quickdraw      Sketch --target-domains Real      --models model=resnet50_dynamic --epochs 15 --batch-size 64 --optimizer name=SGD lr=0.002 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_DomainNet_resnet50/Real/$folder
python ddg/trainer/domain_mix.py --dataset DomainNet --sampler RandomDomainSampler --transform TransformForDomainNet --source-domains Clipart Infograph Painting Quickdraw Real        --target-domains Sketch    --models model=resnet50_dynamic --epochs 15 --batch-size 64 --optimizer name=SGD lr=0.002 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/ddg_DomainNet_resnet50/Sketch/$folder
