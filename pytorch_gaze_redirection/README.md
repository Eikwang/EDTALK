## 參考論文
連結：https://github.com/HzDmS/gaze_redirection

### 本次專案簡介
參考論文所使用的框架是基於 TensorFlow 1.7 版本開發的。但TensorFlow 1.7 現在已經停止支持，框架本身也缺乏更新的功能，我們嘗試多個環境但都有兼容性的問題，修復成本很高。

有嘗試尋找其他基於眼球校正的專案。但我們發現現有網路上這類專案數量不多，且大多數專案都不太能用。所以我們沒有現成的工具可以直接拿來使用或改進。

所以嘗試將這整個專案從頭改寫，並選擇了 PyTorch 作為我們的開發框架。選擇 PyTorch是因為在靈活性、可擴展性以及實用性方面都表現出色。

所以此次專案的目標是想讓這個專案運行起來，且也希望通過重構code，去提升眼球校正技術的應用潛力。
<img width="1046" alt="image" src="https://github.com/user-attachments/assets/967da833-8009-44cf-acf8-2cb0b2991704" />


## Dependencies

Python == 3.10.15 <br />
torch == 2.5.1 <br />
dlib == 19.24.6 <br />
pillow == 10.4.0

## Train

```Bash
python main.py --mode train --data_path /home/andy/AILab/AIfinal/pytorch_gaze_redirection/eyespatch_dataset --log_dir ./log/ --vgg_path ./vgg16_reducedfc.pth
```

## Use the model
```
python main.py --mode eval --client_pictures_dir ./client_pictures/ --log_dir ./log/
```

## VGG16 pretrained weights
```Bash
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```
