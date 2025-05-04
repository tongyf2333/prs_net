首先运行 ```preprocess.py```：

```bash
python preprocess.py --data_dir meshes/chair --output_dir preprocessed/chair --rotate True
```
三个选项分别表示加工前数据目录，加工后数据目录以及是否对数据随机旋转

之后开始训练，运行```train.py```:

```bash
python train.py --train_dir preproccesed/chair --model_dir models --epoch 300 --batch_size 32 --weight 25 --hasaxis False --use_chamfer False --lr 0.01
```

```train_dir```表示训练数据目录，```model_dir```表示模型存储位置，```epoch```表示训练轮数，```batch_size```表示训练时一个批次的大小，```weight```表示正则化损失的权重，```hasaxis```表示是否预测对称轴（默认只预测对称平面），```use_chamfer```表示是否将对称损失替换为倒角距离，```lr```表示学习率

之后测试，输出预测的平面或者对称轴，运行```test.py```:

```bash
python test.py --test_dir preprocessed/test --model_dir models/model_weights_final.pth --batch_size 1 --weight 25 --has_axis False --use_chamfer False 
```

```test_dir```表示测试目录，```model_dir```表示加载的模型路径，其余选项含义与上面一致