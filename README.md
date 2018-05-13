# Pretrained_network

Using Pretrained InceptionV3 network to predict employment labels of villages.

Model:

We used keras  InceptionV3 network which is pretrained on imagenet dataset. We removed the top layer of network, added a fc layer with 1024 nodes and an output layer with 3 nodes. The input image size is 224*224*3. We used only R,G,B bands in LANDSAT data. The dataset is then split into training and test data with 80% as training data and 20% as test data.

Link to trained model:

https://drive.google.com/open?id=1tw7o5LMoPhBvfe6ztPinupxsLIy4kVUP

![Inceptionnet_v3](Results/Inceptionnet_V3.png)

Training:
In training, only the weight of FC layer is learned keeping weight of the remaining architecture fixed.
The network is trained for 2 epochs.

to train network:

nohup python inceptionnet_pretrained.py > Results/pretrained_out1.txt 2>&1 &

To plot training loss:

python plot_loss.py

![training_loss](Results/pretrained_train_loss.png)


To plot confusion matrix:

python plot_confusion_matrix.py


Results:

Unnormalized confusion matrix:

![unnormalized_confusion_matrix](Results/pretrained_unnormalized_confusion_matrix.png)

Normalized confusion matrix:

![Normalized_confusion_matrix](Results/pretrained_normalized_confusion_matrix.png)


Test accuracy is 47.11%

