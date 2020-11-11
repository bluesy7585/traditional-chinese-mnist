# traditional-chinese-mnist
A simple Pytorch example for traditional chinese handwriting recognition.

## Dataset
[Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset) contains common traditional chinese charactors, which has more than 4800 classes.
For simplicity, codes trains on a subset of the full data, about 700 classes which are charactors used in 20000 random chinese names.
Data are split with 90/10 into training/test set.

## Data depoly
- git clone https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset.git
- decompress *.zip files in the dataset into folder 'cleaned_data'
- then call
```
python data_deploy.py
```
It will move all the images into corresponding folder.
```
{project_dir}/cleaned_data
    [CHAR1]/
        CHAR1_0.png
        CHAR1_1.png
        ...
    [CHAR2]/
        CHAR2_0.png
        CHAR2_1.png
        ...
    [CHAR3]/
        CHAR3_0.png
        CHAR3_1.png
        ...
```
## Training
train from scratch
```
python main.py 
```
resume training from a weight file
```
python main.py -w [path to weight file]
```
## Test
```
python main.py -w [path to weight file] -t
eg. 
python main.py -w .\weights\ConvNet_val_loss0.105 -t
...
test acc: 0.977784 3829-3916
```
## Demo
Predict on a random name in the text file and save image. Each charactor in name is randomly pick from test set.
```
python demo.py -w [path to weight file]
e.g
python demo.py -w .\weights\ConvNet_val_loss0.105
...
name 洪俊佑 pred name 洪俊佑
```
![demo image](https://github.com/bluesy7585/traditional-chinese-mnist/blob/main/demo.jpg)
## Reference
[Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)

cnn model used here is modified from [https://github.com/chineseocr/chineseocr](https://github.com/chineseocr/chineseocr)

[random chinese name generator](http://www.richyli.com/name/index.asp)
