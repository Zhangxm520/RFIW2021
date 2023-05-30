# RFIW 2021
Official implementation of "Supervised Contrastive Learning for Facial Kinship Recognition".       
The code and data we (TeamCNU) used to participate in the RFIW 2021 Data Challenge.     
We achieved first place in all three tasks.  
Experimental data are provided by RFIW official [RFIW 2021](https://competitions.codalab.org/competitions/21843#learn_the_details).  
Datasets and some introduction about RFIW 2020 [RFIW 2020 Website](https://web.northeastern.edu/smilelab/rfiw2020/).  

## Dataset
We use The Families In the Wild dataset, we align the faces and crop them to 112*112 pixels, there is a dataset.zip for each task that contains the dataset used for the current task, please unzip it before running the code.  The dataset is completely official, we just align and crop the faces.  

## Pre-training models
We use the ArcFace pre-trained ResNet101 as the feature extraction network, and the code is mainly from [here](https://github.com/dmlc/gluon-cv).The models were pre-trained using mxnet, and we used MMdnn to convert the pre-trained models into pytorch version and Tensorflow 2.0, corresponding to backbone/kit_resnet101.pkl and backbone/ArcFace_r100_v1.h5, respectively.  

## GPU
Requires a GPU with 10G of video memory.  

## Track1 
We use pytorch to implement Track 1.  
If you want to run the code for Track 1, you need to:  
  
**1. Copy backbone/kit_resnet101.pkl to the Track1 directory.**  
**2. unzip Track1/dataset.zip**  


### Main Documents:  
#### 1. Track1/sample0     
Our method requires that the sample pairs in each minimum batch come from different families, and to ensure that the results are reproducible, we have sorted the sample pairs. The current folder keeps the sorted sample pairs.These sample pairs are taken from the official documents provided.  
>>**train_sort.txt**: save the sample pairs for training.  
**val_choose.txt**:selected partial samples from the validation set are used for model selection.  
**val.txt**: the validation set sample pairs are used to derive the threshold values.  
**test.txt**: for testing.  


#### 2. Track1/train.py  
Training Model and the main parameters are:  
>>batch_size : default 25.  
sample  : corresponding to the sorted sample pair folder, corresponding to /Track1/sample0 just mentioned.  
save_path : model save path.  
epochs : default 80.  
beta : temperature parameters default 0.08.  
log_path : log file save path.  
gpu : which gpu you want to use.  
```
python train.py --batch_size 25 --sample Track1/sample0 \  
                --save_path Track1/model_name.pth \  
                --epochs 80 --beta 0.08 --log_path Track1/log_name.txt --gpu 0  
```
#### 3. Track1/find.py    
Finding the threshold and the main parameters are:  
>>sample: Track1/sample0.  
save_path: model paths saved via train.py.  
batch_size :default 40.  
log_path : log file path ,the calculated threshold values will be saved here.  
gpu :which gpu you want to use.  
```
python find.py  --sample Track1/sample0 \  
                --save_path Track1/model_name.pth \  
                --batch_size 40 --log_path Track1/log_name.txt --gpu 0 
```

#### 4. Track1/test.py    
Test the model   
```
python test.py  --sample Track1/sample0 \  
                --save_path Track1/model_name.pth \  
                --threshold 0.1( calculated by find.py) \  
                --batch_size 40 --log_path Track1/log_name.txt --gpu 0 
```
In our environment, we achieved the best model at the 42nd epoch. We think that it is normal for the final results obtained to fluctuate up and down due to software version differences.  
  
## Track2
We have implemented Track 2 using tensorflow 2.  
Basically the same as task 1, only need to copy ArcFace_r100_v1.h5 to the Track2 folder.  
train.py can directly calculate the model prediction threshold, and the threshold will be saved in the log file.  
We get the best model at epoch 13.  
  
## Track3
Use the trained model from Track 1 to complete the prediction of Track 3. PyTorch do it.     
#### 1. Track3/test.py    
Get Track 3 results and  the main parameters are:
>>sample_root: Track3/sample.  
model_path: Track 1 trained model path.      
batch_size :default 40.  
score : fusion method  mean or max.  
log_path : log file path.  
pred_path : result file path.  
gpu :which gpu you want to use.  
```
python test.py  --sample_root Track3/sample \  
                --model_path Track3/model_name.pth \  
                --batch_size 40 --score mean --log_path Track1/log_name.txt \  
                --pred_path Track3/predictions.csv --gpu 0
```
## Citations  

```
@inproceedings{zhang2021supervised,
  title={Supervised contrastive learning for facial kinship recognition},
  author={Zhang, Ximiao and Min, XU and Zhou, Xiuzhuang and Guo, Guodong},
  booktitle={2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)},
  pages={01--05},
  year={2021},
  organization={IEEE}
}
```
