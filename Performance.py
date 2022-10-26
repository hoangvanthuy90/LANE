import numpy as np
    
def Performance(Xtrain = None,Xtest = None,Ytrain = None,Ytest = None): 
    #Evaluate the performance of classification for both multi-class and multi-label Classification
#     [F1macro,F1micro] = Performance(Xtrain,Xtest,Ytrain,Ytest)
    
    #       Xtrain is the training data with row denotes instances, column denotes features
#       Xtest  is the test data with row denotes instances, column denotes features
#       Ytrain is the training labels with row denotes instances
#       Ytest  is the test labels
    
    #   Copyright 2017, Xiao Huang and Jundong Li.
#   $Revision: 1.0.0 $  $Date: 2017/10/18 00:00:00 $
    
    ## Multi class Classification
    if Ytrain.shape[2-1] == 1 and len(unique(Ytrain)) > 2:
        t = templateSVM('Standardize',True)
        model = fitcecoc(Xtrain,Ytrain,'Learners',t)
        pred_label = predict(model,Xtest)
        micro,macro = micro_macro_PR(pred_label,Ytest)
        F1macro = macro.fscore
        F1micro = micro.fscore
    else:
        ## For multi-label classification, computer micro and macro
        rng('default')
        NumLabel = Ytest.shape[2-1]
        macroTP = np.zeros((NumLabel,1))
        macroFP = np.zeros((NumLabel,1))
        macroFN = np.zeros((NumLabel,1))
        macroF = np.zeros((NumLabel,1))
        for i in np.arange(1,NumLabel+1).reshape(-1):
            model = fitcsvm(Xtrain,Ytrain(:,i),'Standardize',True,'KernelFunction','RBF','KernelScale','auto')
            pred_label = predict(model,Xtest)
            mat = confusionmat(Ytest(:,i),pred_label)
            if mat.shape[1-1] == 1:
                macroTP[i] = sum(pred_label)
                macroFP[i] = 0
                macroFN[i] = 0
                if macroTP(i) != 0:
                    macroF[i] = 1
            else:
                macroTP[i] = mat(2,2)
                macroFP[i] = mat(1,2)
                macroFN[i] = mat(2,1)
                macroF[i] = 2 * macroTP(i) / (2 * macroTP(i) + macroFP(i) + macroFN(i))
        F1macro = mean(macroF)
        F1micro = 2 * sum(macroTP) / (2 * sum(macroTP) + sum(macroFP) + sum(macroFN))
    
    return F1macro,F1micro
    
    
def micro_macro_PR(pred_label = None,orig_label = None): 
    # computer micro and macro: precision, recall and fscore
    mat = confusionmat(orig_label,pred_label)
    len_ = mat.shape[1-1]
    macroTP = np.zeros((len_,1))
    macroFP = np.zeros((len_,1))
    macroFN = np.zeros((len_,1))
    macroP = np.zeros((len_,1))
    macroR = np.zeros((len_,1))
    macroF = np.zeros((len_,1))
    for i in np.arange(1,len_+1).reshape(-1):
        macroTP[i] = mat(i,i)
        macroFP[i] = sum(mat(:,i)) - mat(i,i)
        macroFN[i] = sum(mat(i,:)) - mat(i,i)
        macroP[i] = macroTP(i) / (macroTP(i) + macroFP(i))
        macroR[i] = macroTP(i) / (macroTP(i) + macroFN(i))
        macroF[i] = 2 * macroP(i) * macroR(i) / (macroP(i) + macroR(i))
    
    #     macroP(isnan(macroP)) = 0;
#     macroR(isnan(macroR)) = 0;
    macroF[np.isnan[macroF]] = 0
    #     macro.precision = mean(macroP);
#     macro.recall = mean(macroR);
    macro.fscore = mean(macroF)
    micro.precision = sum(macroTP) / (sum(macroTP) + sum(macroFP))
    micro.recall = sum(macroTP) / (sum(macroTP) + sum(macroFN))
    micro.fscore = 2 * micro.precision * micro.recall / (micro.precision + micro.recall)
    return micro,macro
    
    return F1macro,F1micro