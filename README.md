# Awesome-Surgical-Video-Analysis

_**Under development. Welcome to contribute.**_

TO DO:
- Add paper / code links
- Add 2020/2021 works for skill assessment
- Elaborate on surgical phase/action recognition
- Elaborate on surgical instrument/scene segmentation
- Better classification of categories
- De-duplicate

## Contents

[Survey and The Big Picture](#survey)

[Surgical Skill Assessment](#skill)

[Surgical Phase Recognition](#phase)

[Surgical Gesture Recognition](#gesture)

[Surgical Instrument Recognition / Localization](#instrument)

[Surgical Scene / Organ Recognition](#scene)

[Human / Activity in Operating Rooms](#operatingrooms)

[Others](#others)



## <span id = "survey"> **Survey and The Big Picture** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Surgical Data Science: from Concepts to Clinical Translation | **Arxiv 2020** | |
| Surgical Data Science: A Consensus Perspective | **Arxiv 2018** | [Paper](https://arxiv.org/abs/1806.03184) |
| Surgical data science for next-generation interventions | **NBE 2017** | |
| OR Black Box and Surgical Control Tower: Recording and Streaming data and Analytics to Improve Surgical Care | **JVS 2021** | [Paper](https://www.sciencedirect.com/science/article/pii/S1878788621000163?via%3Dihub) |
| Supporting laparoscopic general surgery training with digital technology: The United Kingdom and Ireland paradigm | **BMC 2021** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10124234/) |
| Objective assessment of surgical technical skill and competency in the operating room | **ARBE 2017** | |
| Vision-based and marker-less surgical tool detection and tracking: a review of the literature | **MIA 2017** | |
| Gesture Recognition in Robotic Surgery: a Review | **TBE 2021** | [Paper](https://iris.ucl.ac.uk/iris/publication/1845816/1) |

## <span id = "skill"> **Surgical Skill Assessment** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Towards Unified Surgical Skill Assessment | **CVPR 2021** | |
| Clearness of Operating Field: A Surrogate for Surgical Skills on In-Vivo Clinical Data | **IJCARS 2020** | [Paper](https://link.springer.com/article/10.1007/s11548-020-02267-z) |
| The Pros and Cons: Rank-aware Temporal Attention for Skill Determination in Long Videos | **CVPR 2019** | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Doughty_The_Pros_and_Cons_Rank-Aware_Temporal_Attention_for_Skill_Determination_CVPR_2019_paper.pdf) [Code](https://github.com/hazeld/rank-aware-attention-network) |
| Surgical Skill Assessment on In-Vivo Clinical Data via the Clearness of Operating Field | **MICCAI 2019** | [Paper](https://arxiv.org/abs/2008.11954) |
| Modeling Surgical Technical Skill Using Expert Assessment for Automated Computer Rating | **AnnS 2019** | |
| Towards Optimizing Convolutional Neural Networks for Robotic Surgery Skill Evaluation | **IJCNN 2019** | |
| Video-based surgical skill assessment using 3D convolutional neural networks | **IJCARS 2019** | |
| Accurate and interpretable evaluation of surgical skills from kinematic data using fully convolutional neural networks | **IJCARS 2019** | [Paper](https://arxiv.org/pdf/1908.07319.pdf) [Code](https://github.com/hfawaz/ijcars19) |
| Machine learning methods for automated technical skills assessment with instructional feedback in ultrasound-guided interventions | **IJCARS 2019** | |
| Objective assessment of intraoperative technical skill in capsulorhexis using videos of cataract surgery | **IJCARS 2019** | |
| Objective classification of psychomotor laparoscopic skills of surgeons based on three different approaches | **IJCARS 2019** | |
| Who’s Better? Who’s Best? Pairwise Deep Ranking for Skill Determination | **CVPR 2018** | |
| Tool Detection and Operative Skill Assessment in Surgical Videos Using Region-Based Convolutional Neural Networks | **WACV 2018** | |
| Evaluating surgical skills from kinematic data using convolutional neural networks | **MICCAI 2018** | [Paper](https://arxiv.org/abs/1806.02750) [Code](https://github.com/hfawaz/miccai18) |
| Automated surgical skill assessment in RMIS training | **IJCARS 2018** | |
| Video and accelerometer-based motion analysis for automated surgical skills assessment | **IJCARS 2018** | |
| Deep learning with convolutional neural network for objective skill evaluation in robot-assisted surgery | **IJCARS 2018** | |
| Automated robot‐assisted surgical skill evaluation: Predictive analytics approach | **IJMRCAR 2018** | |
| Meaningful Assessment of Surgical Expertise: Semantic Labeling with Data and Crowds | **MICCAI 2016** | |
| Automated video-based assessment of surgical skills for training and evaluation in medical schools | **IJCARS 2016** | |
| Task-Level vs. Segment-Level Quantitative Metrics for Surgical Skill Assessment | **Surg Educ 2016** | |
| Relative Hidden Markov Models for Video-Based Evaluation of Motion Skills in Surgical Training | **TPAMI 2015** | |
| Automated Assessment of Surgical Skills Using Frequency Analysis | **MICCAI 2015** | |
| Automated objective surgical skill assessment in the operating room from unstructured tool motion in septoplasty | **IJCARS 2015** | |
| Automated Surgical OSATS Prediction From Videos | **ISBI 2014** | |
| Pairwise Comparison-Based Objective Score for Automated Skill Assessment of Segments in a Surgical Task | **IPCAI 2014** | |
| Video Based Assessment of OSATS Using Sequential Motion Textures | **MICCAIW 2014** | |
| Augmenting Bag-of-Words: Data-Driven Discovery of Temporal and Structural Information for Activity Recognition | **CVPR 2013** | |
| String Motif-Based Description of Tool Motion for Detecting Skill and Gestures in Robotic Surgery | **MICCAI 2013** | |
| Robotic Path Planning for Surgeon Skill Evaluation in Minimally-Invasive Sinus Surgery | **MICCAI 2012** | |
| An objective and automated method for assessing surgical skill in endoscopic sinus surgery using eye-tracking and tool-motion data | **IFAR 2012** | |
| Sparse Hidden Markov Models for Surgical Gesture Classification and Skill Evaluation | **IPCAI 2012** | |
| Towards integrating task information in skills assessment for dexterous tasks in surgery and simulation | **ICRA 2011** | |
| Video-based Motion Expertise Analysis in Simulation-based Surgical Training Using Hierarchical Dirichlet Process Hidden Markov Model | **MMAR 2011** | |
| Eye Metrics as an Objective Assessment of Surgical Skill | **AnnS 2010** | |
| Surgical Task and Skill Classification from Eye Tracking and Tool Motion in Minimally Invasive Surgery | **MICCAI 2010** | |
| Task versus Subtask Surgical Skill Evaluation of Robotic Minimally Invasive Surgery | **MICCAI 2009** | |
| Data-derived models for segmentation with application to surgical assessment and training | **MICCAI 2009** | |

## <span id = "phase"> **Surgical Phase Recognition** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Temporal Memory Relation Network for Workflow Recognition from Surgical Video | **TMI 2021** | [Paper](https://arxiv.org/abs/2103.16327) [Code](https://github.com/YuemingJin/TMRNet) |
| Multi-Task Temporal Convolutional Networks for Joint Recognition of Surgical Phases and Steps in Gastric Bypass | **IJCARS 2021** | [Paper](https://arxiv.org/abs/2102.12218v1) |
| TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks | **MICCAI 2020** | |
| Multi-Task Recurrent Convolutional Network with Correlation Loss for Surgical Video Analysis | **MIA 2020** | [Paper](https://arxiv.org/abs/1907.06099) [Code](https://github.com/YuemingJin/MTRCNet-CL) |
| LRTD: Long-Range Temporal Dependency based Active Learning for Surgical Workflow Recognition | **IJCARS 2020** | [Paper](https://link.springer.com/content/pdf/10.1007/s11548-020-02198-9.pdf) [Code](https://github.com/xmichelleshihx/AL-LRTD) |
| Assisted phase and step annotation for surgical videos | **IJCARS 2020** | |
| Hard Frame Detection and Online Mapping for Surgical Phase Recognition | **MICCAI 2019** | |
| MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation | **CVPR 2019** | |
| Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition | **IPCAI 2019** | [Paper](https://arxiv.org/abs/1812.00033) |
| Machine and deep learning for workflow recognition during surgery | **MITAT 2019** | |
| SV-RCNet: Workflow Recognition from Surgical Videos using Recurrent Convolutional Network | **TMI 2018** | [Paper](https://ieeexplore.ieee.org/document/8240734) [Code](https://github.com/YuemingJin/SV-RCNet) |
| DeepPhase: Surgical Phase Recognition in CATARACTS Videos | **MICCAI 2018** | |
| Less is More: Surgical Phase Recognition with Less Annotations through Self-Supervised Pre-training of CNN-LSTM Networks | **Arxiv 2018** | |
| EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos | **TMI 2017**| |
| Unsupervised temporal context learning using convolutional neural networks for laparoscopic workflow analysis | **Arxiv 2017** | [Paper](https://arxiv.org/pdf/1702.03684.pdf) |
| MICCAI Workflow Challenge: Convolutional neural networks with time smoothing and Hidden Markov Model for video frames classification | **Arxiv 2016** | |
| Statistical modeling and recognition of surgical workflow | **MIA 2012** | |
| An application-dependent framework for the recognition of high-level surgical tasks in the OR | **TBE 2011** | |
| Modeling and Segmentation of Surgical Workflow from Laparoscopic Video | **MICCAI 2010** | |
| Surgical phases detection from microscope videos by combining SVM and HMM | **MICCAIW 2010** | |
| On-line recognition of surgical activity for monitoring in the operating room | **IAAI 2008** | |

## <span id = "gesture"> **Surgical Gesture Recognition** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Relational Graph Learning on Visual and Kinematics Embeddings for Accurate Gesture Recognition in Robotic Surgery | **ICRA 2021** | [Paper](https://arxiv.org/abs/2011.01619) |
| Automatic Gesture Recognition in Robot-assisted Surgery with Reinforcement Learning and Tree Search | **ICRA 2020** | [Paper](https://arxiv.org/abs/2002.08718) |
| Multi-Task Recurrent Neural Network for Surgical Gesture Recognition and Progress Prediction | **ICRA 2020** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10111216/) |
| Deep Reinforcement Learning for Surgical Gesture Segmentation and Classification | **MICCAI 2018** | [Paper](https://arxiv.org/abs/1806.08089v1) [Code](https://github.com/Finspire13/RL-Surgical-Gesture-Segmentation) | 
| Unsupervised learning for surgical motion by learning to predict the future | **MICCAI 2018** | | 
| A dataset and benchmarks for segmentation and recognition of gestures in robotic surgery | **TBE 2017**| |
| Temporal Convolutional Networks for Action Segmentation and Detection | **CVPR 2017** | |
| Temporal convolutional networks: A unified approach to action segmentation | **ECCVW 2016** | |
| Recognizing surgical activities with recurrent neural networks | **MICCAI 2016** | |
| Segmental spatiotemporal cnns for fine-grained action segmentation | **ECCV 2016** | |
| Learning convolutional action primitives for fine-grained action recognition | **ICRA 2016** | |
| An improved model for segmentation and recognition of fine-grained activities with application to surgical training tasks | **WACV 2015** | |
| Learning shared, discriminative dictionaries for surgical gesture segmentation and classification | **MICCAIW 2015** | |
| JHU-ISI gesture and skill assessment working set (JIGSAWS): A surgical activity dataset for human motion modeling | **MICCAIW 2014**| |
| Surgical gesture segmentation and recognition | **MICCAI 2013** | |
| Sparse hidden markov models for surgical gesture classification and skill evaluation | **IPCAI 2011** | |
| Data-derived models for segmentation with application to surgical assessment and training | **MICCAI 2009** | |
| Automatic detection and segmentation of robot-assisted surgical motions | **MICCAI 2005** | |


## <span id = "instrument"> **Surgical Instrument Recognition / Localization** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| One to Many: Adaptive Instrument Segmentation via Meta Learning and Dynamic Online Adaptation in Robotic Surgical Video | **ICRA 2021** | [Paper](https://arxiv.org/abs/2103.12988) |
| A Kinematic Bottleneck Approach For Pose Regression of Flexible Surgical Instruments directly from Images | **ICRA 2021** | [Paper](https://arxiv.org/abs/2103.00586) |
| Unsupervised Surgical Instrument Segmentation via Anchor Generation and Semantic Diffusion | **MICCAI 2020** | [Paper](https://arxiv.org/abs/2008.11946) [Code](https://github.com/Finspire13/AGSD-Surgical-Instrument-Segmentation) |
| Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video| **MICCAI 2020** | [Paper](https://arxiv.org/abs/2007.02501) [Code](https://github.com/zxzhaoeric/Semi-InstruSeg) |
| Synthetic and Real Inputs for Tool Segmentation in Robotic Surgery | **MICCAI 2020** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10113753/) | 
| Automated Surgical Instrument Detection from Laparoscopic Gastrectomy Video Images Using an Open Source Convolutional Neural Network Platform | **JACS 2020** | |
| BARNet: Bilinear Attention Network with Adaptive Receptive Field for Surgical Instrument Segmentation | **Arxiv 2020** | |
| Multi-Task Recurrent Convolutional Network with Correlation Loss for Surgical Video Analysis | **MIA 2020** | [Paper](https://arxiv.org/abs/1907.06099) [Code](https://github.com/YuemingJin/MTRCNet-CL) |
| Incorporating Temporal Prior from Motion Flow for Instrument Segmentation in Minimally Invasive Surgery Video | **MICCAI 2019** | [Paper](https://arxiv.org/abs/1907.07899) [Code](https://github.com/keyuncheng/MF-TAPNet) |
| Weakly supervised convolutional LSTM approach for tool tracking in laparoscopic videos | **IJCARS 2019** | |
| Self-supervised surgical tool segmentation using kinematic information | **ICRA 2019** | |
| Learning Where to Look While Tracking Instruments in Robot-assisted Surgery | **MICCAI 2019** | |
| Patch-based adaptive weighting with segmentation and scale (PAWSS) for visual tracking in surgical video | **MIA 2019** | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841519300593) |
| CATARACTS: Challenge on automatic tool annotation for cataRACT surgery | **MIA 2019** | |
| Deep residual learning for instrument segmentation in robotic surgery | **MLMI 2019** | |
| 2017 robotic instrument segmentation challenge | **Arxiiv 2019** | |
| U-NetPlus: A Modified Encoder-Decoder U-Net Architecture for Semantic and Instance Segmentation of Surgical Instruments from Laparoscopic Images | **EMBC 2019**| |
| RASNet: Segmentation for tracking surgical instruments in surgical videos using refined attention segmentation network | **EMBC 2019** | |
| CFCM: Segmentation via coarse to fine context memory | **MICCAI 2018** | |
| 3D Pose Estimation of Articulated Instruments in Robotic Minimally Invasive Surgery | **TMI 2018** | [Paper](https://iris.ucl.ac.uk/iris/publication/1539968/1) |
| Exploiting the potential of unlabeled endoscopic video data with self-supervised learning | **IJCARS 2018** | [Paper](https://link.springer.com/article/10.1007/s11548-018-1772-0) |
| Comparative evaluation of instrument segmentation and tracking methods in minimally invasive surgery | **Arxiv 2018** | [Paper](https://arxiv.org/abs/1805.02475) |
| Weakly-supervised learning for tool localization in laparoscopic videos | **MICCAIW 2018** | |
| Automatic instrument segmentation in robot-assisted surgery using deep learning | **ICMLA 2018** | |
| Concurrent segmentation and localization for tracking of surgical instruments | **MICCAI 2017** | |
| Simultaneous recognition and pose estimation of instruments in minimally invasive surgery | **MICCAI 2017** | [Paper](https://discovery.ucl.ac.uk/id/eprint/1576315/) |
| Toolnet: holistically-nested real-time segmentation of robotic surgical tools | **IROS 2017** | |
| Real-time localization of articulated surgical instruments in retinal microsurgery | **MIA 2016** | |
| Combined 2D and 3D tracking of surgical instruments for minimally invasive and robotic-assisted surgery | **IJCARS 2016** | [Paper](https://discovery.ucl.ac.uk/id/eprint/1480794/) |
| Detecting surgical tools by modelling local appearance and global shape | **TMI 2015** | |
| Visual tracking of da vinci instruments for laparoscopic surgery | **MI 2014**| |
| Toward Detection and Localization of Instruments in Minimally Invasive Surgery | **TBE 2013** | [Paper](https://ieeexplore.ieee.org/document/6359786) |



## <span id = "scene"> **Surgical Scene / Organ Recognition** </span>
| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Artificial Intelligence for Surgical Safety: Automatic Assessment of the Critical View of Safety in Laparoscopic Cholecystectomy using Deep Learning | **AnnS 2020** | [Paper](https://journals.lww.com/annalsofsurgery/Abstract/9000/Artificial_Intelligence_for_Surgical_Safety_.94005.aspx) |
| EasyLabels: weak labels for scene segmentation in laparoscopic videos | **IJCARS 2019** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10084862/) |
| Kidney edge detection in laparoscopic image data for computer-assisted surgery | **IJCARS 2019** | [Paper](https://link.springer.com/article/10.1007/s11548-019-02102-0) |
| Uncertainty-Aware Organ Classification for Surgical Data Science Applications in Laparoscopy | **TBE 2018** | [Paper](https://ieeexplore.ieee.org/abstract/document/8310960) |
| Long Term Safety Area Tracking (LT-SAT) with online failure detection and recovery for robotic minimally invasive surgery | **MIA 2018** | [Paper](https://doi.org/10.1016/j.media.2017.12.010) |


## <span id = "operatingrooms"> **Human / Activity in Operating Rooms** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Self-supervision on Unlabelled OR Data for Multi-person 2D/3D Human Pose Estimation | **MICCAI 2020** | [Paper](https://arxiv.org/abs/2007.08354) |
| Automatic Operating Room Surgical Activity Recognition for Robot-Assisted Surgery | **MICCAI 2020** | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_37) |
| Privacy-Preserving Human Pose Estimation on Low-Resolution Depth Images | **MICCAI 2019** | [Paper](https://arxiv.org/abs/2007.08340) |
| Face Detection in the Operating Room: Comparison of State-of-the-art Methods and a Self-supervised Approach | **IPCAI 2019** | [Paper](https://arxiv.org/abs/1811.12296) |
| A Multi-view RGB-D Approach for Human Pose Estimation in Operating Rooms | **WACV 2017** | [Paper](https://arxiv.org/abs/1701.07372) |
| Pictorial Structures on RGB-D Images for Human Pose Estimation in the Operating Room | **MICCAI 2015** | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-24553-9_45) |
| Data-driven Spatio-temporal RGBD Feature Encoding for Action Recognition in Operating Rooms | **IJCARS 2015** | [Paper](https://link.springer.com/article/10.1007/s11548-015-1186-1) |


## <span id = "others"> **Others** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| A Computer Vision Platform to Automatically Locate Critical Events in Surgical Videos | **AnnS 2021** | [Paper](https://journals.lww.com/annalsofsurgery/Abstract/9000/A_Computer_Vision_Platform_to_Automatically_Locate.93788.aspx) |
| Recognition of Instrument-Tissue Interactions in Endoscopic Videos via Action Triplets | **2020 MICCAI** | [Paper](https://arxiv.org/abs/2007.05405) |
| Ethical implications of AI in robotic surgical training: A Delphi consensus statement | **EUF 2021** | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S2405456921001127) |
| Future Frame Prediction for Robot-assisted Surgery | **IPMI 2021** | [Paper](https://arxiv.org/abs/2103.10308) | 
| Surgical Visual Domain Adaptation: Results from the MICCAI 2020 SurgVisDom Challenge | **Arxiv 2021** | [Paper](https://arxiv.org/abs/2102.13644) | 
| Future-State Predicting LSTM for Early Surgery Type Recognition | **TMI 2019** | [Paper](https://arxiv.org/abs/1811.11727) |
| Global rigid registration of CT to video in laparoscopic liver surgery | **IJCARS 2018** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10048586/) |
| RSDNet: Learning to Predict Remaining Surgery Duration from Laparoscopic Videos Without Manual Annotations | **TMI 2018** | [Paper](https://ieeexplore.ieee.org/document/8509608) |
| Deep Neural Networks Predict Remaining Surgery Duration from Cholecystectomy Videos | **MICCAI 2017** | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_66) |
| Projective biomechanical depth matching for soft tissue registration in laparoscopic surgery | **IJCARS 2017** | [Paper](https://link.springer.com/article/10.1007/s11548-017-1613-6) | 
| Intelligent viewpoint selection for efficient CT to video registration in laparoscopic liver surgery | **IJCARS 2017**| [Paper](https://discovery.ucl.ac.uk/id/eprint/1549761/) |
| Classification Approach for Automatic Laparoscopic Video Database Organization | **IJCARS 2015** | [Paper](https://link.springer.com/article/10.1007/s11548-015-1183-4) |
| Fisher Kernel Based Task Boundary Retrieval in Laparoscopic Database with Single Video Query | **MICCAI 2014** | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-10443-0_52) |
| Real-time dense stereo reconstruction using convex optimisation with a cost-volume for image-guided robotic surgery | **MICCAI 2013** | [Paper](https://link.springer.com/chapter/10.1007/978-3-642-40811-3_6) |











<!-- 
Lena Maier-Hein
Danail Stoyanov
Nicolas Padoy
Gregory D. Hager
 Pierre Jannin
 Stefanie Speidel
 Nassir Navab
 Swaroop S. Vedula -->
