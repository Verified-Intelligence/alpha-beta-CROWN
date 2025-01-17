Please download models from here:

```bash
wget https://download.huan-zhang.com/models/crown-ibp/crown_ibp_cifar_2px.tar.gz
tar xf crown_ibp_cifar_2px.tar.gz
wget https://download.huan-zhang.com/models/crown-ibp/models_crown-ibp_dm-large.tar.gz
tar xf models_crown-ibp_dm-large.tar.gz
mv models_crown-ibp_dm-large/cifar_dm-large_2_255/IBP_large_best.pth cifar_model_dm_large_2px.pth
mv models_crown-ibp_dm-large/cifar_dm-large_8_255/IBP_large_best.pth cifar_model_dm_large_8px.pth
mv models_crown-ibp_dm-large/mnist_dm-large_0.2/IBP_large_best.pth mnist_model_dm_large_0.2.pth
mv models_crown-ibp_dm-large/mnist_dm-large_0.4/IBP_large_best.pth mnist_model_dm_large_0.4.pth
wget -O cifar_model_dm_large_bn_8px.pth http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/cifar/cnn_7layer_bn_cifar
wget http://d.huan-zhang.com/storage/models/cifar_model_dm_large_bn_full_8px.pth
```

These models were originally from the
[CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP),
[auto\_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) and [IBP with short
warmup](https://github.com/shizhouxing/Fast-Certified-Robust-Training)
repository and renamed here.
