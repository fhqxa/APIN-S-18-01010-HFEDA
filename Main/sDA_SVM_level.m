%% 
% Writed by Xinxin Liu.
% Modify at date: 2018-05-18.

clear;T_start = datestr(now);
%% Loading datasets process
	load Protein;
%     load VOC;
%     load CLEF;
%     load Cifar4096d;
%     load SAIAPR;

data_array = double(data_array);
data_array(:,1:end-1) = NormalizeFea(data_array(:,1:end-1),0);
N_level = max(tree(:,2))-min(tree(:,2));
[M,~] = size(data_array);   
Param = cell(N_level,1);
Param_svm = cell(N_level,1);
    
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
for multi = 1:nFolds         % (multi×10%)的数据来训练              
	testID =(indices == multi);            
	trainID = ~testID;
	train_array = data_array(trainID,:);        
	test_array  = data_array(testID,:); 
    
	tic;
	[X, Y]  = creatSubTable_level(train_array, tree);
	internalNodes = tree_InternalNodes(tree);
	for i=1:N_level
        Param{i}.options = [];
        Param{i}.options.ReguAlpha = 0.1;
        Param{i} = SRDAtrainliu(X{i}, Y{i}, Param{i}.options);
    end
    TrainTime(1,multi) = toc;	         
    
	for i=1:N_level    
        Param_svm{i} = svmtrain(Y{i},(X{i})*(Param{i}.projection),'-s 0 -c 1 -t 0 -m 6000 -q'); 
    end
        
    [n_test,~] = size(test_array);
    test_data = test_array(:,1:end-1);
    test_label = test_array(:,end);
	outlabel = zeros(n_test,1);
	for ii=1:n_test
        i_level=1;
        outlabel(ii) = svmpredict(test_label(ii),test_data(ii,:)*(Param{i_level}.projection),Param_svm{i_level});
         while ismember(outlabel(ii),internalNodes)
            i_level = i_level+1;
            [outlabel(ii),~,~] = svmpredict(test_label(ii),test_data(ii,:)*(Param{i_level}.projection),Param_svm{i_level});
        end
	end
    TestTime(1,multi) = toc;
    
    [PH(1,multi), RH(1,multi), FH(1,multi)] = EvaHier_HierarchicalPrecisionAndRecall(outlabel,test_label',tree);
    [P_LCA(1,multi),R_LCA(1,multi),F_LCA(1,multi)] = EvaHier_HierarchicalLCAPrecisionAndRecall(test_label,outlabel',tree);
    TIE(1,multi) = EvaHier_TreeInducedError(test_label,outlabel',tree);
    accuracy_k(1,multi) = EvaHier_HierarchicalAccuracy(test_label,outlabel, tree);%王煜     
    accuracy(1,multi) = length(find((test_label - outlabel)==0))/length(outlabel);%赵红
end
 accuracykMean = mean(accuracy_k); accuracykStd = std(accuracy_k);
 F_LCAMean=mean(F_LCA);
 FHMean=mean(FH);
 TIEmean=mean(TIE);
 TrainTimeMean = mean(TrainTime);
 TestTimeMean = mean(TestTime);

disp(['Completed!  sDA_SVM_level  开始：',T_start,'   结束：',datestr(now)]);