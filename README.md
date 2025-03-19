# A Dual Two-Stage Attention-based Model for Interpretable Hard Landing Prediction from Flight Data

This project contains the source code for the work: **A Dual Two-Stage Attention-based Model for Interpretable Hard Landing Prediction from Flight Data**

**Authors**: Jiaxing Shang, Xiaoquan Li, Ruixiang Zhang, et al.

**Institution**: College of Computer Science, Chongqing University, Chongqing, China

**Abstract**: Hard landings are a significant safety concern in aviation, with potential consequences ranging from poor passenger experiences to serious injuries or fatalities. Predicting and explaining hard landing events are equally important for enhancing flight safety, the former makes it possible to give proactive warnings, while the latter helps pilots identify the reasons and refine their skills. However, existing studies generally lack a comprehensive consideration for the fine-grained characteristics of flight data containing both inter-temporal and inter-parametric relationships, resulting in suboptimal prediction performance. In addition, most of existing approaches aim at improving the prediction performance but fail to provide interpretability for the causes of hard landing. To address the above problems, we propose DUTSAM, a DUal Two-Stage Attention-based interpretable Model for hard landing prediction from quick access recorder (QAR) data. The model consists of dual parallel modules, each of which combines a convolutional feature encoder and a two-stage attention mechanism. The two encoders capture fine-grained characteristics by encoding multivariate data from temporal domain and parametric domain respectively. After that, the dual two-stage attention mechanism captures the inter-temporal and inter-parametric correlations in reverse order to predict hard landing and provide interpretation from both temporal and parametric perspectives. Experimental results on a real QAR dataset with 37,920 flights show that DUTSAM achieves better prediction performance compared with other state-of-the-art baselines in terms of Precision, Recall, and F1-score. Additionally, case study demonstrates that DUTSAM can uncover key flight parameters and moments strongly correlated to the hard landing events.

**Keywords**: Interpretable AI, Attention mechanism, Flight safety, Hard landing, Deep learning

## Table of Contents

- Datagenerator.py: the code for dateset generation
- tools.py: tool functions for data processing
- TrainUtils.py: the code for training settings
- main.py: the main procedure for model training and testing
- DUTSAM.py: the overall model architecture

## Execution

python main.py

## Experimental environment

- Operating System--Ubuntu 16.04
  
- Python version==3.7.7
  
- Pytorch version==1.13.1
  
- CUDA version==12.4
  
- GPU specification--1 × NVIDIA GeForce RTX 2080 Ti (11GB VRAM)
  

## Contact

- shangjx@cqu.edu.cn (Jiaxing Shang, Corresponding author)
  
- li.xiaoquan@foxmail.com (Xiaoquan Li)
