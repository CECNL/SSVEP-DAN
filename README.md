# SSVEP-DAN
This repository is the official implementation of "SSVEP-DAN: Data Alignment Network for SSVEP-based Brain Computer Interfaces". 

## Requirements
#### Step 1:
To install requirements:
```setup
git clone https://github.com/CECNL/SSVEP-DAN.git
cd SSVEP-DAN
conda env create -f SSVEP_DAN_env.yaml
conda activate SSVEP_DAN
```
#### Step 2:
Download [Benchmark dataset](http://bci.med.tsinghua.edu.cn/download.html) and put them to the folder "Benchmark".

Download [Wearable SSVEP BCI dataset](http://bci.med.tsinghua.edu.cn/download.html) and put them to the folder "Wearable".

## Performance comparison for different calibration trials per stimulus 

To obtain SSVEP-DAN performance on the 'Benchmark' scenarios, run this command:
```
python3 DANet_benchmark.py --gpu 0 --tps 2 --method DANet --file_path Testing/Benchmark_diff_ntps/2tps/ --model_path Testing/Benchmark_diff_ntps/2tps/
```

To obtain SSVEP-DAN performance on the 'Dry to dry' scenarios, run this command:
```
python3 DANet_wearable.py --gpu 0 --tps 2 --device dryTOdry --method DANet --file_path Testing/Wearable_diff_npts/dryTOdry/ --model_path Testing/Wearable_diff_npts/dryTOdry/
```
## Performance comparison for different supplementary subjects

To obtain SSVEP-DAN performance on the 'Benchmark' scenarios, run this command:
```
python3 DANet_benchmark.py --gpu 0 --supp 5 --tps 2 --method DANet --file_path Testing/Benchmark_diff_supp/5supp/ --model_path Testing/Benchmark_diff_supp/5supp/
```

To obtain SSVEP-DAN performance on the 'Dry to dry' scenarios, run this command:
```
python3 DANet_wearable.py --gpu 0 --supp 5 --tps 2 --device dryTOdry --method DANet --file_path Testing/Wearable_diff_supp/dryTOdry/5supp --model_path Testing/Wearable_diff_supp/dryTOdry/5supp
```
## Ablation study

To obtain SSVEP-DAN w/o pre-training performance on the 'Benchmark' scenarios, run this command:
```
python3 DANet_benchmark.py --gpu 0 --tps 2 --ablation wo1 --file_path Testing/Ablation/Benchmark/ --model_path Testing/Ablation/Benchmark/
```

To obtain SSVEP-DAN w/o fine-tuning performance on the 'Benchmark' scenarios, run this command:
```
python3 DANet_wearable.py --gpu 0 --tps 2 --device dryTOdry --ablation wo1 --file_path Testing/Ablation/dryTOdry/ --model_path Testing/Ablation/dryTOdry/
```

## Reference

If you use this our codes in your research, please cite our paper and the related references in your publication as:
```bash
@article{,
  title={},
  author={},
  journal={arXiv preprint},
  year={2022}
}
```
If you use the TRCA, please cite the following:
```bash
@article{nakanishi2017enhancing,
  title={Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis},
  author={Nakanishi, Masaki and Wang, Yijun and Chen, Xiaogang and Wang, Yu-Te and Gao, Xiaorong and Jung, Tzyy-Ping},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={65},
  number={1},
  pages={104--112},
  year={2017},
  publisher={IEEE}
}
```
If you use the LST, please cite the following:
```bash
@article{chiang2021boosting,
  title={Boosting template-based SSVEP decoding by cross-domain transfer learning},
  author={Chiang, Kuan-Jung and Wei, Chun-Shu and Nakanishi, Masaki and Jung, Tzyy-Ping},
  journal={Journal of Neural Engineering},
  volume={18},
  number={1},
  pages={016002},
  year={2021},
  publisher={IOP Publishing}
}
```
