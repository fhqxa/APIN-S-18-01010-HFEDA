%% 
% Writed by Xinxin Liu.
% Modifed at date: 2018-05-16. 
    clear;clear;T_start = datestr(now);
%% Loading datasets process
    load Protein;
%     load VOC;
%     load Cifar4096d;
%     load CLEF;
%     load SAIAPR;  
    
data_array = double(data_array);    
data_array(:,1:end-1) = NormalizeFea(data_array(:,1:end-1),0); 
[M,~] = size(data_array);
indexRoot = tree_Root(tree);
internalNodes = tree_InternalNodes(tree);
n_internalNodes  = length(internalNodes);

nFolds = 10;
TrainTime = zeros(1,nFolds);
TestTime = zeros(1,nFolds);
accuracy_k = zeros(1,nFolds);
PH = zeros(1,nFolds);  RH = zeros(1,nFolds);  FH = zeros(1,nFolds);
P_LCA = zeros(1,nFolds);  R_LCA = zeros(1,nFolds); F_LCA = zeros(1,nFolds);
TIE =zeros(1,nFolds);

rand('seed',1);
indices = crossvalind('Kfold',M,nFolds);
testID = zeros(M,1);
for multi = 1:10     
    testID = (indices == multi);
    trainID = ~testID;
    train_array = data_array(trainID,:);        
    test_array  = data_array(testID,:);  

    tic;
    [X, Y] = creatSubTable(train_array, tree);    
    Param{indexRoot}.options = [];
    Param{indexRoot}.options.ReguAlpha = 0.1;
    Param{indexRoot} = SRDAtrainliu(X{indexRoot}, Y{indexRoot}, Param{indexRoot}.options);
    for i = 1:n_internalNodes
        i_node = internalNodes(i);
        if (isempty(X{i_node})==0)
            Param{i_node}.options = [];
            Param{i_node}.options.ReguAlpha = 0.1;
            Param{i_node} = SRDAtrainliu(X{i_node}, Y{i_node}, Param{i_node}.options);
        end
    end    
    TrainTime(1,multi) = toc;
    
    Param_svm{indexRoot} = svmtrain(Y{indexRoot},(X{indexRoot})*(Param{indexRoot}.projection),'-s 0 -c 1 -t 0 -q');
    for i = 1:n_internalNodes   %%-----（待优化）注意前面的中间结点，不一定在树结构的第2层
        i_node = internalNodes(i);
        if (isempty(X{i_node})==0)
            Param_svm{i_node} = svmtrain(Y{i_node},(X{i_node})*(Param{i_node}.projection),'-s 0 -c 1 -t 0 -q');
        end
    end
    
    test_data = test_array(:,1:end-1);
    test_label = test_array(:,end);
    [n_test,~] = size(test_array);
    outlabel = zeros(n_test,1);
    tic;
    for i=1:n_test
        [outlabel(i,1),~, ~] = svmpredict(test_label(i,1),test_data(i,:)*(Param{indexRoot}.projection),Param_svm{indexRoot});
        while ismember(outlabel(i,1),internalNodes)
            [outlabel(i,1),~, ~] = svmpredict(test_label(i,1),test_data(i,:)*(Param{outlabel(i,1)}.projection),Param_svm{outlabel(i,1)});
        end
    end
    TestTime(1,multi) = toc;
    
    [PH(1,multi), RH(1,multi), FH(1,multi)] = EvaHier_HierarchicalPrecisionAndRecall(outlabel,test_label',tree);
    [P_LCA(1,multi),R_LCA(1,multi),F_LCA(1,multi)] = EvaHier_HierarchicalLCAPrecisionAndRecall(test_label,outlabel',tree);
    TIE(1,multi) = EvaHier_TreeInducedError(test_label,outlabel',tree);
    accuracy_k(1,multi) = EvaHier_HierarchicalAccuracy(test_label,outlabel, tree);     
end
 accuracykMean = mean(accuracy_k); accuracykStd = std(accuracy_k);
 F_LCAMean=mean(F_LCA);
 FHMean=mean(FH);
 TIEmean=mean(TIE);
 TrainTimeMean = mean(TrainTime);
 TestTimeMean = mean(TestTime);
 
disp(['Completed!  HSRDA_SVM_node  开始：',T_start,'   结束：',datestr(now)]);
