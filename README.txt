README

You should install the correspoding libraries improted at each file's beginning.

First, run Preprocess.py, you may need to manually change the path for training, validating and testing set respectively.

You can run DataVisualization.py to see the dataset visualization results.

Second, you can run DecisionTree.py, KNN.py, LogisticRegression.py, MLP.py, NaiveBayes.py, RandomForest.py and SVM.py respecively. If you encounter errors, you should check if the file name and path of the file is correct.

Then after yielding the prediction results for 7 modles, you may run Integration_AdaBoost.py Integration_MajorityVote.py, Integration_MajorityVoteWithWeight.py and Integration_MLP.py to obtain the remaining 4 results. In which the MLP.h5 is the model for Integration_MLP.py, you may load it with 'model.load('MLP.h5')'.

Test.csv is our final result (using ensemble learning with AdaBoost).
If you want to check the results for each different approach (!strongly recommended!), please check the directory 'Test_Results'. 
Thank you for your patience.
