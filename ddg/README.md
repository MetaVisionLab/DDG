# DDG

This repo contains the code for our IJCAI 2022 paper: Dynamic Domain Generalization.

# Install

## Install as package

```shell script
git clone https://github.com/MetaVisionLab/DDG.git
cd DDG/ddg
python setup.py install
```

## Install as develop mode

```shell script
git clone https://github.com/MetaVisionLab/DDG.git
cd DDG/ddg
python setup.py develop
```

# Data preprocessing

Enter the project directory and run the `ddg/utils/datasets_download.py` script to automatically perform data preprocessing.

```shell script
cd /path/to/DDG/ddg
python ddg/utils/datasets_download.py
```

If your environment encounters network issues, you can manually download the required files to the corresponding place, and the files with the same md5 will skip the download process and directly perform data preprocessing. 

# Training example

```shell script
cd /path/to/DDG/ddg
bash ./scripts/ddg_PACS_resnet50.sh
```

# Citation

If this project is helpful to you, you can cite our paper:

```
@inproceedings{sun2022ddg,
  title={Dynamic Domain Generalization},
  author={Sun, Zhishu and Shen, Zhifeng and Lin, Luojun and Yu, Yuanlong and Yang, Zhifeng and Yang, Shicai and Chen, Weijie},
  booktitle={IJCAI},
  year={2022}
}
```

# Contact us

For any questions, please feel free to contact Dr. [Zhishu Sun](mailto:siaimes@163.com) or Dr. [Luojun Lin](mailto:linluojun2009@126.com).

# Copyright

This code is free to the academic community for research purpose only. For commercial purpose usage, please contact Dr. [Luojun Lin](mailto:linluojun2009@126.com).
