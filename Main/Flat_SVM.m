%% 
% Writed by Xinxin Liu.
% Modifed at date: 2018-05-18. 

clear;T_start = datestr(now);
%% Loading datasets process
%   load DDTrain;
% 	load ProteinTrain;
%     load ilsvrcTrain;
%     load CarTrain;
    
% data_array(:,1:end-1) = NormalizeFea(data_array(:,1:end-1),0); 

%% Training process and Testing process
% 	load DD;
%   load Protein;
%     load VOC;
% 	load Sun;
%     load Car;
%     load Cifar4096d;
%     load CLEF;
%     load ilsvrc;
%     load Car;
%     load AWAphog;
    load SAIAPR;
% str = {'DD';'Protein';'CLEF';'AWAphog';'VOC';'Car';'CompCar';'ilsvrc';'Cifar4096d';'Sun';'SAIAPR'};
% str = {'CompCar';'ilsvrc';'Cifar4096d'};
% n_dataset = length(str);
% for stri =1:n_dataset
% datafile = [str{stri} '.mat'];
% load  (datafile); 


    
    data_array = double(data_array);
    data_array(:,1:end-1) = NormalizeFea(data_array(:,1:end-1),0); 
    [M,~] = size(data_array);
    
    nFolds = 10;
    TestTime = zeros(1,nFolds);
    accuracy = zeros(1,nFolds);  
    accuracy_k = zeros(1,nFolds);
    PH = zeros(1,nFolds);  RH = zeros(1,nFolds);  FH = zeros(1,nFolds);
    P_LCA = zeros(1,nFolds);  R_LCA = zeros(1,nFolds); F_LCA = zeros(1,nFolds);
    TIE =zeros(1,nFolds);
    
    rand('seed',1);
    indices = crossvalind('Kfold',M,nFolds);%//进行随机分包 for k=1:10//交叉验证k=10，10个包轮流作为测试集

for k=1:nFolds
    testID = (indices == k); trainID = ~testID;
    train_array = data_array(trainID,:);        % svm train 前的数据集
    train_data = train_array(:,1:end-1);
    train_label = train_array(:,end);    
    
    test_array  = data_array(testID,:);         % svm predict前的数据集   
    test_data = test_array(:,1:end-1);
    test_label = test_array(:,end);  
    
    [n_test,~] = size(test_array);
    outlabel = zeros(n_test,1);
    
    model_svm= svmtrain(train_label,train_data,'-s 0 -c 1 -t 0 -m 6000 -q');
    
    tic;
    [outlabel,~, ~] = svmpredict(test_label,test_data,model_svm);
    TestTime(1,k) = toc;
    
    [PH(1,k), RH(1,k), FH(1,k)] = EvaHier_HierarchicalPrecisionAndRecall(outlabel,test_label',tree);
    [P_LCA(1,k),R_LCA(1,k),F_LCA(1,k)] = EvaHier_HierarchicalLCAPrecisionAndRecall(test_label,outlabel',tree);
    TIE(1,k) = EvaHier_TreeInducedError(test_label,outlabel',tree);
    accuracy_k(1,k) = EvaHier_HierarchicalAccuracy(test_label,outlabel, tree);%王煜     
    accuracy(1,k) = length(find((test_label - outlabel)==0))/length(outlabel);%赵红
end
 accuracyMean = mean(accuracy);  accuracyStd = std(accuracy);
 accuracykMean = mean(accuracy_k); accuracykStd = std(accuracy_k);
 F_LCAMean=mean(F_LCA);
 FHMean=mean(FH);
 TIEmean=mean(TIE);
 TestTimeMean = mean(TestTime);
 
% filename = ['Flat_SVM ' str{stri}];
filename = ['Flat_SVM ' 'SAIAPR'];
save(filename, 'accuracykMean', 'accuracykStd','accuracyMean', 'accuracyStd', 'F_LCAMean', 'FHMean', 'TIEmean',  'TestTimeMean');


% end
disp(['Completed! Flat_SVM_t2  开始：',T_start,'    结束：',datestr(now),'    平均FH = ',num2str(mean(FH))]);
