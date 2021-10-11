# Awesome-Surgical-Video-Analysis 

<!-- [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
 -->
_**Keep updating. Welcome to contribute.**_

<!-- TO DO:
- Add links
- Search for more papers
- Better categories, Better curation
- Add 2020/2021 works for skill assessment
- Elaborate on surgical phase/action recognition
- Elaborate on surgical instrument/scene segmentation
- last big 4
- two pie charts (venue / Topic) -->

## Contents

[The Big Picture](#picture)

[Survey](#survey)

[Surgical Skill Assessment](#skill)

[Surgical Phase Recognition](#phase)

[Surgical Gesture Recognition](#gesture)

[Surgical Instrument Recognition / Localization](#instrument)

[Surgical Scene / Anatomy Recognition](#scene)

[Human / Activity in Operating Rooms](#operatingrooms)

<!-- [Video-Based Navigation in Surgery](#navigation)
 -->
[Others](#others)



## <span id = "picture"> **The Big Picture** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Surgical Data Science: from Concepts to Clinical Translation | **Arxiv 2020** | [Paper](https://arxiv.org/abs/2011.02284) |
| Surgical Data Science: A Consensus Perspective | **Arxiv 2018** | [Paper](https://arxiv.org/abs/1806.03184) |
| Surgical data science for next-generation interventions | **NBE 2017** | [Paper](https://www.nature.com/articles/s41551-017-0132-7) |
| CAI4CAI: The Rise of Contextual Artificial Intelligence in Computer-Assisted Interventions | **IEEE 2019** | [Paper](https://ieeexplore.ieee.org/document/8880624) | 
| OR Black Box and Surgical Control Tower: Recording and Streaming data and Analytics to Improve Surgical Care | **JVS 2021** | [Paper](https://www.sciencedirect.com/science/article/pii/S1878788621000163?via%3Dihub) |

<!-- | Supporting laparoscopic general surgery training with digital technology: The United Kingdom and Ireland paradigm | **BMC 2021** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10124234/) | -->



## <span id = "survey"> **Survey** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| A Review on Deep Learning in Minimally Invasive Surgery | **IEEE Access 2021** | [Paper](https://ieeexplore.ieee.org/abstract/document/9386091) |
| Deep neural networks for the assessment of surgical skills: A systematic review | **IEEE Access 2021** | [Paper](https://arxiv.org/pdf/2103.05113) |
| Current methods for assessing technical skill in cataract surgery | **JCRS 2021** | [Paper](https://journals.lww.com/jcrs/Abstract/2021/02000/Current_methods_for_assessing_technical_skill_in.18.aspx) |
| Gesture Recognition in Robotic Surgery: a Review | **TBE 2021** | [Paper](https://iris.ucl.ac.uk/iris/publication/1845816/1) |
| Objective assessment of surgical technical skill and competency in the operating room | **ARBE 2017** | [Paper](https://www.annualreviews.org/doi/10.1146/annurev-bioeng-071516-044435) |
| Vision-based and marker-less surgical tool detection and tracking: a review of the literature | **MIA 2017** | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841516301657?via%3Dihub) |
| Surgical process modelling: a review | **IJCARS 2013** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-013-0940-5) |


## <span id = "skill"> **Surgical Skill Assessment** </span>


| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Comparative Validation of Machine Learning Algorithms for Surgical Workflow and Skill Analysis with the HeiChole Benchmark | **Arxiv 2021** | [Paper](https://arxiv.org/abs/2109.14956) |
| Development and Validation of a 3-Dimensional Convolutional Neural Network for Automatic Surgical Skill Assessment Based on Spatiotemporal Video Analysis | **JAMA 2021** | [Paper](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2782991?utm_campaign=articlePDF&utm_medium=articlePDFlink&utm_source=articlePDF&utm_content=jamanetworkopen.2021.20786) |
| Towards Unified Surgical Skill Assessment | **CVPR 2021** | [Paper](http://www.vie.group/media/pdf/CVPR2021_Puz4Y7Z.pdf) [Code](https://github.com/Finspire13/Towards-Unified-Surgical-Skill-Assessment) |
| Clearness of Operating Field: A Surrogate for Surgical Skills on In-Vivo Clinical Data | **IJCARS 2020** | [Paper](https://link.springer.com/article/10.1007/s11548-020-02267-z) |
| Predicting the quality of surgical exposure using spatial and procedural features from laparoscopic videos | **IJCARS 2020** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-019-02072-3) |
| The Pros and Cons: Rank-aware Temporal Attention for Skill Determination in Long Videos | **CVPR 2019** | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Doughty_The_Pros_and_Cons_Rank-Aware_Temporal_Attention_for_Skill_Determination_CVPR_2019_paper.pdf) [Code](https://github.com/hazeld/rank-aware-attention-network) |
| Surgical Skill Assessment on In-Vivo Clinical Data via the Clearness of Operating Field | **MICCAI 2019** | [Paper](https://arxiv.org/abs/2008.11954) |
| Modeling Surgical Technical Skill Using Expert Assessment for Automated Computer Rating | **AnnS 2019** | [Paper](https://www.researchgate.net/publication/319558609_Modeling_Surgical_Technical_Skill_Using_Expert_Assessment_for_Automated_Computer_Rating) |
| Towards Optimizing Convolutional Neural Networks for Robotic Surgery Skill Evaluation | **IJCNN 2019** | [Paper](https://www.researchgate.net/publication/336167355_Towards_Optimizing_Convolutional_Neural_Networks_for_Robotic_Surgery_Skill_Evaluation) |
| Video-based surgical skill assessment using 3D convolutional neural networks | **IJCARS 2019** | [Paper](https://arxiv.org/abs/1903.02306) [Code](https://gitlab.com/nct_tso_public/surgical_skill_classification?utm_source=catalyzex.com) |
| Accurate and interpretable evaluation of surgical skills from kinematic data using fully convolutional neural networks | **IJCARS 2019** | [Paper](https://arxiv.org/pdf/1908.07319.pdf) [Code](https://github.com/hfawaz/ijcars19) |
| Machine learning methods for automated technical skills assessment with instructional feedback in ultrasound-guided interventions | **IJCARS 2019** | [Paper](https://www.researchgate.net/publication/332546556_Machine_learning_methods_for_automated_technical_skills_assessment_with_instructional_feedback_in_ultrasound-guided_interventions) |
| Objective assessment of intraoperative technical skill in capsulorhexis using videos of cataract surgery | **IJCARS 2019** | [Paper](https://www.cs.jhu.edu/~tkim60/files/IJCARS_FINAL.pdf) |
| Objective classification of psychomotor laparoscopic skills of surgeons based on three different approaches | **IJCARS 2019** | |
| Who’s Better? Who’s Best? Pairwise Deep Ranking for Skill Determination | **CVPR 2018** | |
| Tool Detection and Operative Skill Assessment in Surgical Videos Using Region-Based Convolutional Neural Networks | **WACV 2018** | |
| Evaluating surgical skills from kinematic data using convolutional neural networks | **MICCAI 2018** | [Paper](https://arxiv.org/abs/1806.02750) [Code](https://github.com/hfawaz/miccai18) |
| Automated surgical skill assessment in RMIS training | **IJCARS 2018** | |
| Video and accelerometer-based motion analysis for automated surgical skills assessment | **IJCARS 2018** | |
| Deep learning with convolutional neural network for objective skill evaluation in robot-assisted surgery | **IJCARS 2018** | |
| Surgical skills: Can learning curves be computed from recordings of surgical activities? | **IJCARS 2018** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-018-1713-y) |
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
| Self-supervised representation learning for surgical activity recognition | **IJCARS 2021** | [Paper](https://link.springer.com/content/pdf/10.1007/s11548-021-02493-z.pdf) |
| Surgical Workflow Anticipation Using Instrument Interaction | **MICCAI 2021** | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_59) |
| Comparative Validation of Machine Learning Algorithms for Surgical Workflow and Skill Analysis with the HeiChole Benchmark | **Arxiv 2021** | [Paper](https://arxiv.org/abs/2109.14956) |
| OperA: Attention-Regularized Transformers for Surgical Phase Recognition | **MICCAI 2021** | [Paper](https://arxiv.org/abs/2103.03873) |
| Temporal Memory Relation Network for Workflow Recognition from Surgical Video | **TMI 2021** | [Paper](https://arxiv.org/abs/2103.16327) [Code](https://github.com/YuemingJin/TMRNet) |
| Multi-Task Temporal Convolutional Networks for Joint Recognition of Surgical Phases and Steps in Gastric Bypass | **IJCARS 2021** | [Paper](https://arxiv.org/abs/2102.12218v1) |
| Train one, Classify one, Teach one"–Cross-surgery transfer learning for surgical step recognition | **MIDL 2021** | [Paper](https://arxiv.org/pdf/2102.12308v2.pdf) |
| TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks | **MICCAI 2020** | |
| Automated laparoscopic colorectal surgery workflow recognition using artificial intelligence: Experimental research | **IJS 2020** | [Paper](https://www.sciencedirect.com/science/article/pii/S1743919120303988) |
| Multi-Task Recurrent Convolutional Network with Correlation Loss for Surgical Video Analysis | **MIA 2020** | [Paper](https://arxiv.org/abs/1907.06099) [Code](https://github.com/YuemingJin/MTRCNet-CL) |
| LRTD: Long-Range Temporal Dependency based Active Learning for Surgical Workflow Recognition | **IJCARS 2020** | [Paper](https://link.springer.com/content/pdf/10.1007/s11548-020-02198-9.pdf) [Code](https://github.com/xmichelleshihx/AL-LRTD) |
| Assisted phase and step annotation for surgical videos | **IJCARS 2020** | |
| Accurate Detection of Out of Body Segments in Surgical Video using Semi-Supervised Learning | **MIDL 2020** | [Paper](https://openreview.net/pdf?id=k-ANsPQJxY) |
| Impact of data on generalization of AI for surgical intelligence applications | **SciReports 2020** | [Paper](https://www.nature.com/articles/s41598-020-79173-6) |
| Hard Frame Detection and Online Mapping for Surgical Phase Recognition | **MICCAI 2019** | |
| MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation | **CVPR 2019** | |
| Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition | **IPCAI 2019** | [Paper](https://arxiv.org/abs/1812.00033) |
| Machine and deep learning for workflow recognition during surgery | **MITAT 2019** | |
| Assessment of automated identification of phases in videos of cataract surgery using machine learning and deep learning techniques | **JAMA 2019** | [Paper](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2729808) |
| Multitask learning of temporal connectionism in convolutional networks using a joint distribution loss function to simultaneously identify tools and phase in surgical videos | **Arxiv 2019** | [Paper](https://arxiv.org/pdf/1905.08315.pdf) |
| SV-RCNet: Workflow Recognition from Surgical Videos using Recurrent Convolutional Network | **TMI 2018** | [Paper](https://ieeexplore.ieee.org/document/8240734) [Code](https://github.com/YuemingJin/SV-RCNet) |
| DeepPhase: Surgical Phase Recognition in CATARACTS Videos | **MICCAI 2018** | |
| Surgical activity recognition in robot-assisted radical prostatectomy using deep learning | **MICCAI 2018** | [Paper](https://arxiv.org/pdf/1806.00466v1.pdf) |
| Knowledge transfer for surgical activity prediction | **IJCARS 2018** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-018-1768-9) |
| Temporal coherence-based self-supervised learning for laparoscopic workflow analysis | **MICCAIW 2018** | [Paper](https://arxiv.org/abs/1806.06811) |
| “Deep-Onto” network for surgical workflow and context recognition | **IJCARS 2018** | [Paper](https://link.springer.com/article/10.1007/s11548-018-1882-8) |
| Less is More: Surgical Phase Recognition with Less Annotations through Self-Supervised Pre-training of CNN-LSTM Networks | **Arxiv 2018** | |
| EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos | **TMI 2017**| |
| Unsupervised temporal context learning using convolutional neural networks for laparoscopic workflow analysis | **Arxiv 2017** | [Paper](https://arxiv.org/pdf/1702.03684.pdf) |
| System events: readily accessible features for surgical phase detection | **IJCARS 2016** | [Paper](https://link.springer.com/article/10.1007/s11548-016-1409-0) |
| Automatic data-driven real-time segmentation and recognition of surgical workflow | **IJCARS 2016** | [Paper](https://hal.archives-ouvertes.fr/hal-01299344/document) | 
| MICCAI Workflow Challenge: Convolutional neural networks with time smoothing and Hidden Markov Model for video frames classification | **Arxiv 2016** | |
| Automatic phase prediction from low-level surgical activities | **IJCARS 2015** | [Paper](https://link.springer.com/article/10.1007/s11548-015-1195-0) |
| Lapontospm: an ontology for laparoscopic surgeries and its application to surgical phase recognition | **IJCARS 2015** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-015-1222-1) | 
| Real-time segmentation and recognition of surgical tasks in cataract surgery videos | **TMI 2014** | [Paper](https://ieeexplore.ieee.org/abstract/document/6860246) |
| Statistical modeling and recognition of surgical workflow | **MIA 2012** | |
| A framework for the recognition of high-level surgical tasks from video images for cataract surgeries | **TBE 2012** | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3432023/) |
| An application-dependent framework for the recognition of high-level surgical tasks in the OR | **TBE 2011** | |
| Modeling and Segmentation of Surgical Workflow from Laparoscopic Video | **MICCAI 2010** | |
<!-- | Surgical phases detection from microscope videos by combining SVM and HMM | **MICCAIW 2010** | | -->
<!-- | On-line recognition of surgical activity for monitoring in the operating room | **IAAI 2008** | |
 -->
<!-- | Toward a Neural-Symbolic Framework for Automated Workflow Analysis in Surgery | **MEDICON 2019** | [Paper](https://link.springer.com/chapter/10.1007%2F978-3-030-31635-8_192) | -->




## <span id = "gesture"> **Surgical Gesture Recognition** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Relational Graph Learning on Visual and Kinematics Embeddings for Accurate Gesture Recognition in Robotic Surgery | **ICRA 2021** | [Paper](https://arxiv.org/abs/2011.01619) |
| Automatic Gesture Recognition in Robot-assisted Surgery with Reinforcement Learning and Tree Search | **ICRA 2020** | [Paper](https://arxiv.org/abs/2002.08718) |
| Multi-Task Recurrent Neural Network for Surgical Gesture Recognition and Progress Prediction | **ICRA 2020** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10111216/) |
| Automated surgical activity recognition with one labeled sequence | **MICCAI 2019** | [Paper](https://arxiv.org/pdf/1907.08825.pdf) |
| Using 3d convolutional neural networks to learn spatiotemporal features for automatic surgical gesture recognition in video | **MICCAI 2019** | [Paper](https://link.springer.com/chapter/10.1007%2F978-3-030-32254-0_52) |
| Segmenting and classifying activities in robot-assisted surgery with recurrent neural networks | **IJCARS 2019** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-019-01953-x) |
| Deep Reinforcement Learning for Surgical Gesture Segmentation and Classification | **MICCAI 2018** | [Paper](https://arxiv.org/abs/1806.08089v1) [Code](https://github.com/Finspire13/RL-Surgical-Gesture-Segmentation) | 
| Unsupervised learning for surgical motion by learning to predict the future | **MICCAI 2018** | | 
| Surgical motion analysis using discriminative interpretable patterns | **AIM 2018** | [Paper](https://www.sciencedirect.com/science/article/pii/S0933365717306681?via%3Dihub) |
| A dataset and benchmarks for segmentation and recognition of gestures in robotic surgery | **TBE 2017**| |
| Temporal Convolutional Networks for Action Segmentation and Detection | **CVPR 2017** | |
| Temporal convolutional networks: A unified approach to action segmentation | **ECCVW 2016** | |
| Recognizing surgical activities with recurrent neural networks | **MICCAI 2016** | |
| Segmental spatiotemporal cnns for fine-grained action segmentation | **ECCV 2016** | |
| Learning convolutional action primitives for fine-grained action recognition | **ICRA 2016** | |
| Unsupervised surgical data alignment with application to automatic activity annotation | **ICRA 2016** | [Paper](https://ieeexplore.ieee.org/document/7487608) |
| An improved model for segmentation and recognition of fine-grained activities with application to surgical training tasks | **WACV 2015** | |
| Learning shared, discriminative dictionaries for surgical gesture segmentation and classification | **MICCAIW 2015** | |
| JHU-ISI gesture and skill assessment working set (JIGSAWS): A surgical activity dataset for human motion modeling | **MICCAIW 2014**| |
| Surgical gesture segmentation and recognition | **MICCAI 2013** | |
| Surgical gesture classification from video and kinematic data | **MIA 2013** | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841513000522) |
| Sparse hidden markov models for surgical gesture classification and skill evaluation | **IPCAI 2011** | |
| Data-derived models for segmentation with application to surgical assessment and training | **MICCAI 2009** | |
| Automatic detection and segmentation of robot-assisted surgical motions | **MICCAI 2005** | |





## <span id = "instrument"> **Surgical Instrument Recognition / Localization** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| hSDB-instrument: Instrument Localization Database for Laparoscopic and Robotic Surgeries |  **MICCAI 2021**  | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_38) |
| Prototypical Interaction Graph for Unsupervised Domain Adaptation in Surgical Instrument Segmentation |  **MICCAI 2021**  | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_26) |
| Co-Generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data | **MICCAI 2021** | [Paper](https://arxiv.org/pdf/2103.09276.pdf) |
| One to Many: Adaptive Instrument Segmentation via Meta Learning and Dynamic Online Adaptation in Robotic Surgical Video | **ICRA 2021** | [Paper](https://arxiv.org/abs/2103.12988) |
| A Kinematic Bottleneck Approach For Pose Regression of Flexible Surgical Instruments directly from Images | **ICRA 2021** | [Paper](https://arxiv.org/abs/2103.00586) |
| Simulation-to-real domain adaptation with teacher–student learning for endoscopic instrument segmentation | **IJCARS 2021** | [Paper](https://link.springer.com/article/10.1007/s11548-021-02383-4) | 
| Searching for Efficient Architecture for Instrument Segmentation in Robotic Surgery | **MICCAI 2020** | [Paper](https://arxiv.org/pdf/2007.04449v1.pdf) |
| Unsupervised Surgical Instrument Segmentation via Anchor Generation and Semantic Diffusion | **MICCAI 2020** | [Paper](https://arxiv.org/abs/2008.11946) [Code](https://github.com/Finspire13/AGSD-Surgical-Instrument-Segmentation) |
| Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video| **MICCAI 2020** | [Paper](https://arxiv.org/abs/2007.02501) [Code](https://github.com/zxzhaoeric/Semi-InstruSeg) |
| Synthetic and Real Inputs for Tool Segmentation in Robotic Surgery | **MICCAI 2020** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10113753/) | 
| BARNet: Bilinear Attention Network with Adaptive Receptive Field for Surgical Instrument Segmentation | **Arxiv 2020** | |
| Multi-Task Recurrent Convolutional Network with Correlation Loss for Surgical Video Analysis | **MIA 2020** | [Paper](https://arxiv.org/abs/1907.06099) [Code](https://github.com/YuemingJin/MTRCNet-CL) |
| Real-time surgical needle detection using region-based convolutional neural networks | **IJCARS 2020** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-019-02050-9) |
| Learning Representations of Endoscopic Videos to Detect Tool Presence Without Supervision | **ML-CDS 2020** | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-60946-7_6) [Code](https://github.com/zdavidli/tool-presence/) |
| Incorporating Temporal Prior from Motion Flow for Instrument Segmentation in Minimally Invasive Surgery Video | **MICCAI 2019** | [Paper](https://arxiv.org/abs/1907.07899) [Code](https://github.com/keyuncheng/MF-TAPNet) |
| Weakly supervised convolutional LSTM approach for tool tracking in laparoscopic videos | **IJCARS 2019** | |
| Self-supervised surgical tool segmentation using kinematic information | **ICRA 2019** | |
| Learning Where to Look While Tracking Instruments in Robot-assisted Surgery | **MICCAI 2019** | |
| Patch-based adaptive weighting with segmentation and scale (PAWSS) for visual tracking in surgical video | **MIA 2019** | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841519300593) |
| CATARACTS: Challenge on automatic tool annotation for cataRACT surgery | **MIA 2019** | |
| 2017 robotic instrument segmentation challenge | **Arxiv 2019** | |
| Multitask learning of temporal connectionism in convolutional networks using a joint distribution loss function to simultaneously identify tools and phase in surgical videos | **Arxiv 2019** | [Paper](https://arxiv.org/pdf/1905.08315.pdf) |
| U-NetPlus: A Modified Encoder-Decoder U-Net Architecture for Semantic and Instance Segmentation of Surgical Instruments from Laparoscopic Images | **EMBC 2019**| |
| RASNet: Segmentation for tracking surgical instruments in surgical videos using refined attention segmentation network | **EMBC 2019** | |
| CFCM: Segmentation via coarse to fine context memory | **MICCAI 2018** | |
| Monitoring tool usage in surgery videos using boosted convolutional and recurrent neural networks | **MIA 2018** | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841518302470) |
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
| Surgical Tool Tracking and Pose Estimation in Retinal Microsurgery | **MICCAI 2015** | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-24553-9_33) |
| Toward Detection and Localization of Instruments in Minimally Invasive Surgery | **TBE 2013** | [Paper](https://ieeexplore.ieee.org/document/6359786) |
| Unified detection and tracking of instruments during retinal microsurgery | **TPAMI 2013** | [Paper](https://ieeexplore.ieee.org/document/6319313) |
| Data-driven visual tracking in retinal microsurgery | **MICCAI 2012** | [Paper](https://link.springer.com/chapter/10.1007%2F978-3-642-33418-4_70) |

<!-- | Visual tracking of da vinci instruments for laparoscopic surgery | **MI 2014**| | -->
<!-- | Automated Surgical Instrument Detection from Laparoscopic Gastrectomy Video Images Using an Open Source Convolutional Neural Network Platform | **JACS 2020** | | -->
<!-- | Deep residual learning for instrument segmentation in robotic surgery | **MLMI 2019** | | -->

## <span id = "scene"> **Surgical Scene / Anatomy Recognition** </span>
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
| Video based activity recognition in trauma resuscitation | **FG 2013** | [Paper](https://ieeexplore.ieee.org/document/6553758) |


<!-- ## <span id = "navigation"> **Video-Based Navigation in Surgery** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Autonomously navigating a surgical tool inside the eye by learning from demonstration | **ICRA 2020** | [Paper](https://asco.lcsr.jhu.edu/wp-content/uploads/2020/09/eye_surgery_ICRA_2020_paris.pdf) |
| Evaluation and stability analysis of video-based navigation system for functional endoscopic sinus surgery on in vivo clinical data | **TMI 2018** | [Paper](https://www.cs.jhu.edu/~ayushis/pdfs/preprints/Leonard_TMI18_Nav.pdf) |
| Endoscopic navigation in the absence of CT imaging | **MICCAI 2018** | [Paper](http://www.cs.jhu.edu/~ayushis/pdfs/preprints/Sinha_MICCAI2018_EndNavwoCT.pdf) |
| Global rigid registration of CT to video in laparoscopic liver surgery | **IJCARS 2018** | [Paper](https://discovery.ucl.ac.uk/id/eprint/10048586/) |
| Intelligent viewpoint selection for efficient CT to video registration in laparoscopic liver surgery | **IJCARS 2017**| [Paper](https://discovery.ucl.ac.uk/id/eprint/1549761/) |
| Rendering-based video-CT registration with physical constraints for image-guided endoscopic sinus surgery | **ISOP 2015** | [Paper](https://www.cs.jhu.edu/~areiter/JHU/Publications_files/Otake_SPIE2015_Paper_VideoCT_9415-8.pdf) |
| A system for video-based navigation for endoscopic endonasal skull base surgery | **TMI 2011** | [Paper](http://www.cs.jhu.edu/~hwang/papers/TMI2011.pdf) |
 -->
## <span id = "others"> **Others** </span>

| Title | Venue | Links |
| :--------------------: | :-------------: | :-----: |
| Surgical Instruction Generation with Transformers | **MICCAI 2021** | [Paper](https://arxiv.org/pdf/2107.06964.pdf) |
| SurgeonAssist-Net: Towards Context-Aware Head-Mounted Display-Based Augmented Reality for Surgical Guidance | **MICCAI 2021** | [Paper](https://arxiv.org/pdf/2107.06397.pdf) |
| E-DSSR: Efficient Dynamic Surgical Scene Reconstruction with Transformer-based Stereoscopic Depth Perception | **MICCAI 2021** | [Paper](https://arxiv.org/pdf/2107.00229.pdf) |
| Learning Domain Adaptation with Model Calibration for Surgical Report Generation in Robotic Surgery | **ICRA 2021** | |
| A Computer Vision Platform to Automatically Locate Critical Events in Surgical Videos | **AnnS 2021** | [Paper](https://journals.lww.com/annalsofsurgery/Abstract/9000/A_Computer_Vision_Platform_to_Automatically_Locate.93788.aspx) |
| Ethical implications of AI in robotic surgical training: A Delphi consensus statement | **EUF 2021** | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S2405456921001127) |
| Future Frame Prediction for Robot-assisted Surgery | **IPMI 2021** | [Paper](https://arxiv.org/abs/2103.10308) | 
| Surgical Visual Domain Adaptation: Results from the MICCAI 2020 SurgVisDom Challenge | **Arxiv 2021** | [Paper](https://arxiv.org/abs/2102.13644) | 
| Offline identification of surgical deviations in laparoscopic rectopexy | **AIM 2020** | [Paper](https://www.sciencedirect.com/science/article/pii/S0933365719303185) |
| Recognition of Instrument-Tissue Interactions in Endoscopic Videos via Action Triplets | **MICCAI 2020** | [Paper](https://arxiv.org/abs/2007.05405) |
| Orientation Matters: 6-DoF Autonomous Camera Movement for Minimally Invasive Surgery | **Arxiv 2020** | [Paper](https://arxiv.org/pdf/2012.02836.pdf) |
| Future-State Predicting LSTM for Early Surgery Type Recognition | **TMI 2019** | [Paper](https://arxiv.org/abs/1811.11727) |
| Dense depth estimation in monocular endoscopy with self-supervised learning methods | **TMI 2019** |  |
| Real-time identification of blood regions for hemostasis support in laparoscopic surgery | **SIVP 2019** | [Paper](https://link.springer.com/article/10.1007%2Fs11760-018-1369-7) | 
| RSDNet: Learning to Predict Remaining Surgery Duration from Laparoscopic Videos Without Manual Annotations | **TMI 2018** | [Paper](https://ieeexplore.ieee.org/document/8509608) |
| Learning to see forces: Surgical force prediction with rgb-point cloud temporal convolutional networks | **MICCAIW 2018** | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-01201-4_14) |
| Deep Neural Networks Predict Remaining Surgery Duration from Cholecystectomy Videos | **MICCAI 2017** | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_66) |
| Projective biomechanical depth matching for soft tissue registration in laparoscopic surgery | **IJCARS 2017** | [Paper](https://link.springer.com/article/10.1007/s11548-017-1613-6) | 
| Surgical Soundtracks: Towards Automatic Musical Augmentation of Surgical Procedures | **MICCAI 2017** | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_76) |
| Distinguishing surgical behavior by sequential pattern discovery | **JBI 2017** | [Paper](https://www.sciencedirect.com/science/article/pii/S1532046417300229) |
| Finding discriminative and interpretable patterns in sequences of surgical activities | **AIM 2017** | [Paper](https://www.sciencedirect.com/science/article/pii/S0933365716305966?via%3Dihub) |
| Automatic matching of surgeries to predict surgeons’ next actions | **AIM 2017** | [Paper](https://www.sciencedirect.com/science/article/pii/S093336571730129X) | 
| Query-by-example surgical activity detection | **IJCARS 2016** | [Paper](https://link.springer.com/article/10.1007/s11548-016-1386-3) |
| Classification Approach for Automatic Laparoscopic Video Database Organization | **IJCARS 2015** | [Paper](https://link.springer.com/article/10.1007/s11548-015-1183-4) |
| Work domain constraints for modelling surgical performance | **IJCARS 2015** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-015-1166-5) |
| Smoke detection in endoscopic surgery videos: a first step towards retrieval of semantic events | **IJMRCAS 2015** | [Paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/rcs.1578) |
| Fisher Kernel Based Task Boundary Retrieval in Laparoscopic Database with Single Video Query | **MICCAI 2014** | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-10443-0_52) |
| Non-linear temporal scaling of surgical processes | **AIM 2014** | [Paper](https://www.sciencedirect.com/science/article/pii/S0933365714001225?via%3Dihu) |
| Surgical process modelling: a review | **IJCARS 2014** | [Paper](https://link.springer.com/article/10.1007/s11548-013-0940-5) |
| Real-time dense stereo reconstruction using convex optimisation with a cost-volume for image-guided robotic surgery | **MICCAI 2013** | [Paper](https://link.springer.com/chapter/10.1007/978-3-642-40811-3_6) |
| Vision-based proximity detection in retinal surgery | **TBE 2012** | [Paper](https://ieeexplore.ieee.org/document/6212340) |
| Automatic knowledge-based recognition of low-level tasks in ophthalmological procedures | **IJCARS 2012** | [Paper](https://link.springer.com/article/10.1007%2Fs11548-012-0685-6) |
| Similarity metrics for surgical process models | **AIM 2012** | [Paper](https://www.sciencedirect.com/science/article/pii/S0933365711001394?via%3Dihub) |















<!-- 
Out of scope

| A multi-camera, multi-view system for training and skill assessment for robot-assisted surgery | IJCARS 2020 | [Paper](https://link.springer.com/article/10.1007/s11548-020-02176-1) |

| Analysis of the structure of surgical activity for a suturing and knot-tying task | PloSOne 2016 | [Paper](https://www.semanticscholar.org/paper/Analysis-of-the-Structure-of-Surgical-Activity-for-Vedula-Malpani/5458ded654f048a86e3edde4b2c70678aaa7fb08) |

| A study of crowdsourced segment-level surgical skill assessment using pairwise rankings | IJCARS 2015 | [Paper]() |

| Unsupervised surgical task segmentation with milestone learning | ISRR 2015 | |

| Human-machine collaborative surgery using learned models | ICRA 2012 | [Paper](https://cra.org/ccc/wp-content/uploads/sites/2/2018/01/Human-machine-collaborative-surgery-using-learned-models.pdf) | -->


<!-- 
Lena Maier-Hein
Danail Stoyanov
Nicolas Padoy
Gregory D. Hager
 Pierre Jannin
 Stefanie Speidel
 Nassir Navab
 Swaroop S. Vedula -->
