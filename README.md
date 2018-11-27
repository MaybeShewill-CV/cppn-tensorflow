# cppn-tensorflow
Simple tensorflow implemention of "Compositional Pattern Producing Networks: A Novel Abstraction of Development" model"

Thanks for the related project [hardmaru/cppn](https://github.com/hardmaru/cppn-tensorflow)

## Install

This software has only been tested on ubuntu 16.04(x64), python3.5, 
cuda-9.0, cudnn-7.0 with a GTX-1070 GPU. To install this software 
you need tensorflow 1.10.0 and other version of tensorflow has not 
been tested but I think it will be able to work properly 
in tensorflow above version 1.0. Other required package you may install 
them by

```angular2html
pip3 install -r requirements.txt
```

## Test model
You can try the model as follows

```
python tools/test_model.py
```
If you want to chanage some of the parameters you can refer the param 
details as follows

```angular2html
python tools/test_model.py --help
```

## Personalized customization
You can implement your own generate function by modifying the 
cppn_model/cppn_net.py script. There has been three different
generating functions you can implement your own version follow 
those examples

## Some of the test result
# gray image

`Test output 1`
![Test_output_1](/data/gray/generated_11.jpg)

`Test output 2`
![Test_output_2](/data/gray/generated_16.jpg)

# color image

`Test output 3`
![Test_output_3](/data/color/generated_15.jpg)

`Test output 4`
![Test_output_4](/data/color/generated_8.jpg)