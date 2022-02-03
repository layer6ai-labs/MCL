<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## WWW'22 MCL: Mixed-Centric Loss for Collaborative Filtering 

Authors: Zhaolin Gao*, Zhaoyue Cheng*, Felipe Perez, Jianing Sun, Maksims Volkovs
[[paper](https://github.com/layer6ai-labs/MCL/blob/master/pdf/www2022_mcl.pdf)]

<a name="Environment"/>

## Environment:

The code was developed and tested on the following python environment:
```
python 3.7.7
pytorch 1.9.0
scikit-learn 0.23.2
numpy 1.19.1
scipy 1.5.4
tqdm 4.48.2
tensorboard 2.7.0
```
<a name="instructions"/>

## Instructions:

Train and evaluation LightGCN + MCL:

Training on `Amazon-Digital-Music` dataset:
```
python main.py --dataset amazon-digital-music --alpha 1.25 --beta 5 --lamb_p 6.5 --lamb_n -0.5
```

Training on `Amazon-Grocery` dataset:
```
python main.py --dataset amazon-grocery --alpha 1.25 --beta 5 --lamb_p 6.5 --lamb_n -0.5
```

Training on `Amazon-Books` dataset:
```
python main.py --dataset amazon-book --alpha 1 --beta 4 --lamb_p 8 --lamb_n -1
```

Training on `Yelp2021` dataset:
```
python main.py --dataset yelp --alpha 1 --beta 4 --lamb_p 8 --lamb_n -1
```

<a name="citation"/>

## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{gao2022mcl,
      title={MCL: Mixed-Centric Loss for Collaborative Filtering},
      author={Zhaolin Gao, Zhaoyue Cheng, Felipe Perez, Jianing Sun, Maksims Volkovs},
      booktitle={Proceedings of the International World Wide Web Conference},
      year={2022}
    }


