# Augmented Space Linear Models
## Abstract
The linear model uses the space defined by the input to project the target or desired signal and find the optimal set of model parameters. When the problem is nonlinear, the adaptation requires nonlinear models for good performance, which becomes slower and more cumbersome. In this paper, we propose a nonlinear model in the full joint space of input and desired signal as the projection space, called Augmented Space Linear Model (ASLM). This new algorithm takes advantage of the linear solution, augmented with a table indexed by the current input vector containing the current error, which is available in the training phase. During testing stage, when there is no desired signal available, the model output is estimated by adding the current linear model output to the value of the error in the table indexed by the training inputs. This algorithm can solve nonlinear problems with the computational efficiency of linear methods extended with an error memory table, which can be regarded as a trade off between accuracy and computational complexity. Making full use of the training errors, the proposed augmented space model may provide a new way to improve many modeling tasks. We present the theory and show preliminary results to support the methodology. 
## Algorithms
In the demo, we demonstrate LS, KNN, ASLM and KLMS on Lorenz data set.
## Language
Matlab
## Cite
If you use this code, please cite the following paper:

@inproceedings{qin2018augmented,
  title={Augmented Space Linear Model},
  author={Qin, Zhengda and Chen, Badong and Zheng, Nanning and Principe, Jose C},
  booktitle={2018 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--6},
  year={2018},
  organization={IEEE}
}

