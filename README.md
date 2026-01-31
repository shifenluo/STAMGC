# STAMGC
STAMGC: A Dual-Contrastive Learning Framework for Spatial Domain Identification in Spatial Transcriptomics

# Overview
In this study, we propose STAMGC, a dual-contrastive learning approach based on graph convolutional networks. This method jointly optimizes the model through regional and topological contrastive learning to process ST data, effectively reducing domain noise and enhancing detailed features. Moreover, this study is the first to apply Gaussian smoothing from the image processing field to ST data, providing a novel perspective to address the spatial discontinuity of gene expression signals.We demonstrate the effectiveness of STAMGC across multiple tissue types and technological platforms. In both qualitative and quantitative analyses, STAMGC consistently achieves superior performance. Furthermore, STAMGC not only identifies finer structures in the mouse sagittal brain, formed by stitching together the anterior and posterior brain slices, but also provides new insights for human breast cancer research.
![image_text](https://github.com/shifenluo/STAMGC/blob/main/STAMGC.tif)

# Requirements
- Python: 3.8.16
- Python packages: anndata = 0.9.2, json5 = 0.12.1, jupyter-client = 8.6.3, matplotlib = 3.7.5, pandas = 2.0.3, numpy = 1.24.4, pillow = 10.4.0, rpy2 = 3.5.11, torch=2.4.1, scipy = 1.10.1, scanpy = 1.9.8, sklearn = 1.3.2

# Usage
First, you need to download the dataset from the public URL. All subsequent examples are located in the ./examples directory. Select or set the corresponding parameter group in the ./train_config.yaml file, and then click to run.

# Dataset
(1) Human DLPFC within the spatialLIBD (http://spatial.libd.org/spatialLIBD);  
(2) Human breast cancer and mouse brain tissue sections datasets (https://www.10xgenomics.com/resources/datasets);  
(3)Stereo-seq dataset for mouse olfactory bulb tissue (https://github.com/BGIResearch/stereopy);  
(4) other datasets from the Spatial Omics DataBase(SODB)(https://gene.ai.tencent.com/SpatialOmics/). 
