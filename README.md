# Shoppster
Multimodal Shopping Assistant, A project made for the course [Machine Learning and Discovery](https://web.cs.ucdavis.edu/~hpirsiav/courses/MLf22/) taught by Professor [Hamed Pirsiavash](https://web.cs.ucdavis.edu/~hpirsiav/) at UC Davis. 


## Motivation
This project aims at illustrating transformer-based multi-modal retrieval for the fashion domain. Given an index of images and their descriptions, our system can represent the images and their text-descriptions in a joint vector space using contrastive learning and query the index using text-based queries or queries based on a combination of texts and im- ages. We use the Vision Transformer (ViT) as our Image Backbone and MP-Net as our Text Backbone and also report how tuning the Vision Transformer using Masked Image Modelling (SimMIM) and Masked Autoencoding (MAE) on fashion data affects the recall of our system.


## Jump To
* <a id="jumpto"></a> [Project Structure](#project-structure-)
* <a id="jumpto"></a> [Dataset](#dataset-)
* <a id="jumpto"></a> [Triplet-Mining](#triplet-mining-)
* <a id="jumpto"></a> [Network](#network-)
* <a id="jumpto"></a> [Contrastive-Learning](#contrastive-learning-)
* <a id="jumpto"></a> [Improving the Vision Encoder](#improving-the-vision-encoder-)
* <a id="jumpto"></a> [Training](#training-)
* <a id="jumpto"></a> [Results](#results-)
* <a id="jumpto"></a> [Text Queries](#text-queries-)
* <a id="jumpto"></a> [Text Plus Image Queries](#text-plus-image-queries-)
* <a id="jumpto"></a> [Limitations](#limitations-)
* <a id="jumpto"></a> [Usage and Tips](#usage-and-tips-)
* <a id="jumpto"></a> [References](#references-)

# Project Structure [`↩`](#jumpto)
- The **data** directory contains the triplet data split according to train,val and test. It also contains a csv file called *search.csv* which contains data about all the unique images in all the three splits. We use the images in this dataframe as our index.
- The **MLD_Project_Triplet_Mining.ipynb** notebook deals with Data Analysis, Cleaning and construction of the Triplet Dataset. It contains the rules used for mining different levels of triplets.
- The **MLD_Project_ViT_Base.ipynb** notebook contains the Data Splitting, the Network and the Training of the Network with the Base ViT model
- The **Pretraining.ipynb** notebook contains the script for pre-training the ViT Image Backbone on Fashion Images using Algorithms from [SimMIM](https://arxiv.org/pdf/2111.09886.pdf) and [MAE](https://arxiv.org/pdf/2111.06377.pdf) using HuggingFace for **4 epochs**
- The **MLD_Project_MIM.ipynb** notebook contains the training of the Network with the ViT model pre-trained using the techniques in SimMIM
- The **MLD_Project_MAE.ipynb** notebook contains the training of the Network with the ViT model pre-trained using the techniques in MAE
- The **Plots_And_Figures.ipynb** notebook contains the plots for the validation and training curves, recall curves, and the results using some qualitative evaluation. This notebook contains some sample queries for **Text -> Image** search and **Text + Image -> Image** search


# Dataset [`↩`](#jumpto)
We use the Small Version of the Fashion Product Images Dataset from Kaggle which can be found 
[here](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

We further clean the dataset and retain only the most common 25 clothing categories. We also preprocess the text to remove brand names of the clothing items from the product descriptions. We modify the product descriptions to include the Gender for which the product is designed by prepending either ”Men” or ”Women” to the product description. The decision to only hold on to popular categories is that we do not have enough images for all the categories and this will lead to poor training for those categories. We are using the Small version of the dataset (280 MB) which is much smaller than the Large version (15.71 GB). Note that we do this due to a restriction on compute, and that the number of datapoints in both versions of the dataset is the same. In the large version, the image quality is much higher than in the version we use which also hurts performance.

# Triplet-Mining [`↩`](#jumpto)
In order to prepare the dataset for contrastive learning, we structure the data such that every image (considered the anchor) has a positive text de- scription and K negative text descriptions. These negatives correspond to descriptions of other images.

**While mining for negative descriptions we choose different toughness levels of negative descriptions. For instance for a product *P* of a certain colour *Col* belonging to a gender *G* and product-category *C*:** 
1) For easy negative descriptions we obtain a random description of a product with product-category not equal to *C*
2) For semi-hard negative descriptions we obtain a description of the product with the same product-category *C* or similar product categories but different Color *Col* and different Gender *G*
3) For hard negative descriptions we obtain a description of the product with the same product-category *C* and same Color *Col* and same Gender *G*. These are essentially images of different products that look very similar

**In our experiments for all categories we choose a K of 9 on average**


# Network [`↩`](#jumpto)
Our neural network contains the Google ViT-Base trained on ImageNet and the Mi- crosoft MPNet-All model from Sentence- Transformers. The model encodes the image using the Image Encoder and normalizes the embeddings before passing them through another projection layer and then passes them through a non-linearity like ReLU and then uses Dropout for regularization. The model uses the text encoder and does the same for the descriptions. Note that the same text encoder encodes the positive and negative text descriptions separately. After having projected the images and text to the same number of dimensions, they are passed onto the loss. The loss is described in detail in the next section.

![](/images/network.png)



# Contrastive-Learning [`↩`](#jumpto)
 
We use the Triplet Margin Distance Loss and set the margin to 0.2. Using a larger margin can make the training tougher and longer but improve the final representations. We did not have a chance to play around with the margin due to our limited compute resources. The loss tries to minimize the distance between the embeddings of the image and the embeddings of the positive text description and maximize the distance between the embeddings of the image and the embeddings of the negative text descriptions with respect to the margin.


![](/images/loss.png)

Here d() denotes the distance function which in our case is the Euclidean Distance. a denotes the anchor, p denotes the positive text description and n denotes the negative text description.

# Improving the Vision Encoder [`↩`](#jumpto)
Since our Image Encoder is originally trained on ImageNet and our application is in the fashion do- main, we use methods from recent literature to pre-train the Vision Encoder on these fashion images. This section looks at two methods described in -

1) SimMIM: a Simple Framework for Masked Image Modeling and
2) Masked Autoencoders Are Scalable Vision Learners

**Both the papers recommend pretraining a fine-tuned checkpoint for atleast 50 epochs. However, due to our limited compute budget we could only pre-train the models on the fashion dataset for 4 epochs which is a possible reason why our results are not very different across all methods.**
 
**SimMIM: a Simple Framework for Masked Image Modeling**

In this paper, the authors randomly mask large patches (32 X 32) in the images and use a simple linear layer to predict the raw pixel values at the masked locations. The authors show that a simple setting such as this is very performant and robust to image sizes as well provided that a large pro- portion (about 80 percent) of the image is being masked.
The authors use a L1 loss while training between the predicted masked pixel values and the origi- nal pixel values at the masked positions. They say that it is important to only calculate the loss over the masked values since otherwise the task is no longer prediction but reconstruction.

**Masked Autoencoders Are Scalable Vision Learners**

In this paper the authors randomly mask large patches in the images as in the paper above but the difference is that they use an autoencoder ar- chitecture to calculate the pixel values of the re- consutructed image.
First, after masking the image only the non- masked patches of the image are fed into the En- coder.
Then a shared learnabale vector which represents the mask tokens is fed into the decoder along with all the encoded image patches from the Encoder. Since the mask representation is shared, positional encoding are also injected into the decoder so that the masked tokens can be distinguished. The au- thors here use the Mean Squared Error (MSE) loss between the predictions and the original pixel val- ues only on the masked regions


# Training [`↩`](#jumpto)
We train three variants of the Image Encoder and thus Three Networks in Total. The training, validation and test data are available on GitHub and care has been taken to split them so that all the triplets belonging to a certain anchor are in the same data split so as to avoid data-leakage.
The Image Encoder was first pre-trained on the Image Dataset on a **NVIDIA Tesla T4 using mixed-precision (FP-16) training**. For both the pre-training algorithms, we trained for 4 epochs each (as opposed to the suggested 50 due to a limited computational budget). The data was split such that roughly 80 percent of the triplets are in the training set and 10 percent each in the validation and test set.

**Training was carried out on a NVIDIA A100- SXM4-40GB with a batch-size of 256.**
The total training time including all the variants and pre-training was about 24 hours. We report our results in the section below.

# Results [`↩`](#jumpto)

## Training and Validation Losses
We call our Base Model without any pre-training on the fashion images Base. The model pre-trained using SimMIM is called Base + SimMIM and the model pre-trained using the masked au- toencoder technique is called Base + MAE.
Below are the training and validation losses for all the models across 6 epochs.

![](/images/train_loss.png)

![](/images/val_loss.png)

## Recall on Test Set
We calculate the Recall on the Testset at 1,5,20 and 100. Here is the Recall for all the three variants.
![](/images/recall.png)




# Text Queries [`↩`](#jumpto)
![](/images/qualitative_queries_men.png)
![](/images/qualitative_queries_women.png)


# Text Plus Image Queries [`↩`](#jumpto)
![](/images/mm_men.png)
![](/images/mm_women.png)


# Limitations [`↩`](#jumpto)
## Poor Performance for Certain Colours and Product Categories
We find that certain categories and colours which occur less frequently during training compared to other colours and categories are not well captured by the model. This is only natural as in an ideal setting we would have equal amounts of data to represent all products and colours.
## No Improvements Using Pre-Training Techniques
We see that pre-training using the two algorithms in sections 3.5 and 3.6 actually end up hurting per- formance. This is possibly due to the fact that they were only pre-trained for 4 epochs due to our limited computational budget. The authors of these algorithms suggest that fine-tuned check- points must be trained for about 50 epochs. Since we cannot train for 50 epochs, we report the re- sults using 4 epochs which lead to no peformance boost.
## Compute Limits
Since we only had limited access to GPUs, we could not train for longer and try out certain hyper- parameter searches. For instance, we strongly be- lieve that increasing the margin in the Triplet Loss and training the model for longer would lead to a better latent representation space but we could not try this out since our access to Colab Pro Plus expired. Even after 6 epochs we observed that the Train- ing and Validation Losses were decreasing which indicates that we did not train to an optimal parameter setting as we did not have the compute to train beyond 6 epochs for each of the variants.

# Usage and Tips [`↩`](#jumpto)
Please note that I have connected my Google Drive to the Colab Notebooks to enable eay data transfer. Please change the paths accordingly for yourself (in case you mount a drive).

- To replicate our results, first run the **MLD_Project_Triplet_Mining.ipynb** notebook. You can also directly use our data from the *data* folder.
- If you want to pre-train the vision backbone, open **Pretraining.ipynb** and run the script for either **vit_mae** or *SimMIM**. Consider tweaking the *mask_ratio* but remember to keep it above 70 percent at all times for good performance. If you have the compute consider training for atleast 50 epochs by changing the *num_training_epochs* parameter.
- Now if you had pretrained, choose the training notebook to run according to your pre-training choice. If you choose not to pre-train, directly run **MLD_Project_ViT_Base.ipynb**
- Use the code in **Plots_And_Figures** to set up the index and query it using the model. Keep in mind that all the data paths must be changed according to where you save your models during training.

## Accessing our Pre-Trained Models
If you want to directly access our pre-trained models (on which the current results are reported), please shoot a mail to **rsangani@ucdavis.edu** and I shall share them with you.

# References [`↩`](#jumpto)
[[1] Prithiviraj Damodaran’s Tutorial on Multi- Modal Learning](https://www.youtube.com/watch?v=QtMOUlXUPc8) 

[[2] "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ,**v2 2021**](https://arxiv.org/abs/1810.04805) Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby

[[3] “SimMIM: a Simple Framework for Masked Image Modeling” ,**v2 2022**](https://arxiv.org/pdf/2111.09886.pdf) Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, Han Hu

[[4] “Masked Autoencoders Are Scalable Vision Learners” ,**v3 2021**](https://arxiv.org/pdf/2111.06377.pdf) Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick

[[5] “MPNet: Masked and Permuted Pre-training for Language Understanding” ,**v2 2020**](https://arxiv.org/abs/2004.09297) Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu

[[6] “Learning local feature descriptors with triplets and shallow convolutional neural networks” ,**2016**](http://www.bmva.org/bmvc/2016/papers/paper119/index.html) Vassileios Balntas, Edgar Riba, Daniel Ponsa and Krystian Mikolajczyk






