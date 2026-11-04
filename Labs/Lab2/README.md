# DL Lab2 Baseline

## Directory structure

```text
DL_Lab2/
├── dataset/
│   └── oxford-iiit-pet/
├── src/
│   ├── models/
│   │   ├── unet.py
│   │   └── resnet34_unet.py
│   ├── oxford_pet.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── saved_models/
└── requirements.txt
```

## Training

```bash
cd src
python train.py --model unet --dataset_root ../dataset/oxford-iiit-pet --save_dir ../saved_models
python train.py --model resnet34_unet --dataset_root ../dataset/oxford-iiit-pet --save_dir ../saved_models
```

## Evaluation

```bash
cd src
python evaluate.py --model unet --checkpoint ../saved_models/unet_best.pth --dataset_root ../dataset/oxford-iiit-pet
python evaluate.py --model resnet34_unet --checkpoint ../saved_models/resnet34_unet_best.pth --dataset_root ../dataset/oxford-iiit-pet
```

## Inference

```bash
cd src
python inference.py --model unet --checkpoint ../saved_models/unet_best.pth --dataset_root ../dataset/oxford-iiit-pet --output_dir ../outputs_unet
python inference.py --model resnet34_unet --checkpoint ../saved_models/resnet34_unet_best.pth --dataset_root ../dataset/oxford-iiit-pet --output_dir ../outputs_resnet34_unet
```
