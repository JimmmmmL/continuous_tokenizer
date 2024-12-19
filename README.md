# SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.01756-b31b1b.svg)](https://arxiv.org/abs/2412.10958v1)&nbsp;
[![huggingface models](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/SoftVQVAE)&nbsp;

</div>

![Images generated with 32 and 64 tokens](assets/figure1.jpg)


## Change Logs

* [12/18/2024] All models have been released at: https://huggingface.co/SoftVQVAE. Checkout [demo](demo/sit.ipynb) here. 


## Setup
```
conda create -n softvq python=3.10 -y
conda activate softvq
pip install -r requirements.txt
```


## Models

### Tokenizers


| Tokenizer 	| rFID 	| Huggingface 	|
|:---:	|:---:	|:---:	|
| SoftVQ-L-64 	| 0.61 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-l-64) 	|
| SoftVQ-BL-64 	| 0.65 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-bl-64) 	|
| SoftVQ-B-64 	| 0.88 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-b-64) 	|
| SoftVQ-L-32 	| 0.74 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-l-32) 	|
| SoftVQ-BL-32 	| 0.68 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-bl-32) 	|
| SoftVQ-B-32 	| 0.89 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-b-32) 	|


### SiT-XL Models

| Genenerative Model 	| Tokenizer 	| gFID (w/o CFG) 	| Huggingface 	|
|:---:	|:---:	|:---:	|:---:	|
| SiT-XL 	| SoftVQ-L-64 	| 5.35 	| [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-l-64) 	|
| SiT-XL 	| SoftVQ-BL-64 	| 5.80 	| [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-bl-64) 	|
| SiT-XL 	| SoftVQ-B-64 	| 5.98 	| [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-b-64) 	|
| SiT-XL 	| SoftVQ-L-32 	| 7.59 	| [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-l-32) 	|
| SiT-XL 	| SoftVQ-BL-32 	| 7.69 	| [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-bl-32) 	|
| SiT-XL 	| SoftVQ-B-32 	| 7.99 	| [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-b-32) 	|


### DiT-XL Models



## Training 

Will be released soon.


## Inference


**Reconstruction**



**SiT Generation**



**DiT Generation**




## Reference
```
@article{chen2024softvqvae,
    title={SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer},
    author={Hao Chen and Ze Wang and Xiang Li and Ximeng Sun and Fangyi Chen and Jiang Liu and Jindong Wang and Bhiksha Raj and Zicheng Liu and Emad Barsoum},
    year={2024},
}
```

## Acknowledge 
A large portion of our code are borrowed from [Llamagen](https://github.com/FoundationVision/LlamaGen), [VAR](https://github.com/FoundationVision/VAR/tree/main), [ImageFolder](https://github.com/lxa9867/ImageFolder), [DiT](https://github.com/facebookresearch/DiT/tree/main), [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA/tree/main)
