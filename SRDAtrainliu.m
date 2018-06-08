function [model] = SRDAtrainliu(fea, label, options)
% SRDAtrain: Training Spectral Regression Discriminant Analysis 
%
if ~isfield(options,'bCenter')
    options.bCenter = 1;
end
if ~isfield(options,'ReguType')
    options.ReguType = 'Ridge';
end
LARs = false;

nSmp = size(fea,1);
ClassLabel = unique(label);
model.ClassLabel = ClassLabel;
nClass = length(ClassLabel);

% Response Generation
rand('state',0);
Y = rand(nClass,nClass);
Z = zeros(nSmp,nClass);
for i=1:nClass
    idx = find(label==ClassLabel(i));
    Z(idx,:) = repmat(Y(i,:),length(idx),1);
end
Z(:,1) = ones(nSmp,1);
[Y,R] = qr(Z,0);
Y(:,1) = [];        % �������������𣬶����Ǽ򵥵�0001��0010��0100

feaOrig = fea;
if options.bCenter  % ��һ��ɾ���˺���Ч�������ˡ�
    sampleMean = mean(fea);
    fea = (fea - repmat(sampleMean,nSmp,1)); % ���еļ�����ֵ
end

[model.projection] = SRliu(options, Y, fea);

model.LARs = LARs;

Embed_Train = feaOrig*model.projection;    %  ͶӰ�������ĵ�
ClassCenter = zeros(nClass,size(Embed_Train,2));
for i = 1:nClass
	feaTmp = Embed_Train(label == ClassLabel(i),:);
	ClassCenter(i,:) = mean(feaTmp,1);
end
model.ClassCenter = ClassCenter;

model.TYPE = 'SRDA';
model.options = options;