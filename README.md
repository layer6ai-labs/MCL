<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## WWW'22 MCL: Mixed-Centric Loss for Collaborative Filtering 

Authors: Zhaolin Gao*, Zhaoyue Cheng*, Felipe Perez, Jianing Sun, Maksims Volkovs


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
```
<a name="instructions"/>

## Instructions (TODO):

Train and evaluation LightGCN + MCL:

Training on `Amazon-Digital-Music` dataset:
```
python main.py
```

Training on `Amazon-Grocery` dataset:
```
python main.py
```

Training on `Amazon-Books` dataset:
```
python main.py
```

Training on `Yelp2021` dataset:
```
python main.py
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


