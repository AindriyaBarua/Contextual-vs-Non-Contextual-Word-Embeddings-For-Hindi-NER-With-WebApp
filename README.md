# Contextual vs Non-Contextual Word Embeddings For Hindi NER With WebApp
 [![forthebadge made-with-python 2](https://img.shields.io/badge/Made%20with-MATLAB%20-brightgreen.svg)](https://in.mathworks.com/products/matlab.html?requestedDomain=) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]() [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]() 
This repo consists of all the codes and dataset of the research paper, **"[Analysis Of Contextual and Non-Contextual
Word Embedding Models For Hindi NER With
Web Application For Data Collection](https://www.researchgate.net/publication/349190662_Analysis_of_Contextual_and_Non-contextual_Word_Embedding_Models_for_Hindi_NER_with_Web_Application_for_Data_Collection)"**.




## Abstract :
Abstract. Named Entity Recognition (NER) is the process of taking a string and identifying relevant proper nouns in it. In this paper ‡ we report the development of the Hindi NER system, in Devanagari script, using various embedding models. We categorize embeddings as Contextual and Non-contextual, and further compare them inter and intra-category. Un-
der non-contextual type embeddings, we experiment with Word2Vec and FastText, and under the contextual embedding category, we experiment with BERT and its variants, viz. RoBERTa, ELECTRA, CamemBERT, Distil-BERT, XLM-RoBERTa. For non-contextual embeddings, we use five machine learning algorithms namely Gaussian NB, Adaboost Classifier, Multi-layer Perceptron classifier, Random Forest Classifier, and Decision Tree Classifier for developing ten Hindi NER systems, each,
once with Fast Text and once with Gensim Word2Vec word embedding models. These models are then compared with Transformers based contextual NER models, using BERT and its variants. A comparative study among all these NER models is made. Finally, the best of all these models is used and a web app is built, that takes a Hindi text of any length and
returns NER tags for each word and takes feedback from the user about the correctness of tags. These feed-backs aid our further data collection. 

## Keywords : 
Gaussian NB · Adaboost Classifier · Multi-layer Perceptron classifier · Random Forest Classifier · Decision Tree Classifier · Gensim Word2Vec · FastText · Transformer · BERT · RoBERTa · ELECTRA · CamemBERT · Distil-BERT · XLM-RoBERTa 

## Authors :
**[Aindriya Barua]()**<sup>∗</sup>, [Thara.S]()<sup>†</sup>, [Premjith B]()<sup>†</sup> and [Soman KP]()<sup>‡</sup> 
<!---
**<sup>∗</sup>Department of Computer Science Engineering, Amrita Vishwa Vidyapeetham, India.** <br/> 
<sup>†</sup>Center for Computational Engineering and Networking (CEN), Amrita School of Engineering, Coimbatore.<br/> 
<sup>‡</sup>Center for Cyber Security Systems and Networks, Amrita School of Engineering, Amritapuri Amrita Vishwa Vidyapeetham, India.

## How to run the code?
### For **Classical Machine Learning**
* Run `all.py` [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/blob/master/all.py)
### For **Deep Neural Network (100 iterations)** 
* Run `dnn1.py` for 1-hidden layer network and run `dnn1acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)
* Run `dnn2.py` for 2-hidden layer network and run `dnn2acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)
* Run `dnn3.py` for 3-hidden layer network and run `dnn3acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)
* Run `dnn4.py` for 4-hidden layer network and run `dnn4acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)
* Run `dnn5.py` for 5-hidden layer network and run `dnn5acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)

### For **Deep Neural Network (1000 iterations)** 
* Run `dnn1.py` for 1-hidden layer network and run `dnn1acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)
* Run `dnn2.py` for 2-hidden layer network and run `dnn2acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)
* Run `dnn3.py` for 3-hidden layer network and run `dnn3acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)
* Run `dnn4.py` for 4-hidden layer network and run `dnn4acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)
* Run `dnn5.py` for 5-hidden layer network and run `dnn5acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)



## Recommended Citation :
If you use this repository in your research, cite this paper - "[Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security](https://ieeexplore.ieee.org/document/8494096)".
```bib
{
  @InProceedings{Rahul2018,
  author       = {Rahul-Vigneswaran, K and Vinayakumar, R and Soman, KP and Poornachandran, Prabaharan},
  title        = {Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security},
  booktitle    = {2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT)},
  year         = {2018},
  pages        = {1--6},
  organization = {IEEE},
  abstract     = {Intrusion detection system (IDS) has become an essential layer in all the latest ICT system due to an urge towards cyber safety in the day-to-day world. Reasons including uncertainty in ﬁnding the types of attacks and increased the complexity of advanced cyber attacks, IDS calls for the need of integration of Deep Neural Networks (DNNs). In this paper, DNNs have been utilized to predict the attacks on Network Intrusion Detection System (N-IDS). A DNN with 0.1 rate of learning is applied and is run for 1000 number of epochs and KDDCup-’99’ dataset has been used for training and benchmarking the network. For comparison purposes, the training is done on the same dataset with several other classical machine learning algorithms and DNN of layers ranging from 1 to 5. The results were compared and concluded that a DNN of 3 layers has superior performance over all the other classical machine learning algorithms.},
  doi          = {https://doi.org/10.1109/ICCCNT.2018.8494096},
  keywords     = {Intrusion detection, deep neural networks, machine learning, deep learning},
  url          = {https://github.com/rahulvigneswaran/Intrusion-Detection-Systems},
}
```
-->
## Issue / Want to Contribute ? :
Open a new issue or do a pull request incase your are facing any difficulty with the code base or you want to contribute to it.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)]()
