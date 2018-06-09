%% creatSubTable_level
% Written by Xinxin Liu
% 2017-12-24
%% Creat subtable by level
function [DataMod,LabelMod]=creatSubTable_level(dataset, tree)
    [N_data,~] = size(dataset);
    N_level = max(tree(:,2))-min(tree(:,2));
    Data = dataset(:,1:end-1);
    Label =  dataset(:,end);
    
    DataMod = cell(1,N_level);
    LabelMod = cell(1,N_level);
%% ����  ÿһ�������μ����丸�ڵ���游������ڲ�ṹ�У�û�д���ͬ���ڵ��µ�����������data_array�е����λ�ã�
    for i=1:N_data
        i_label = Label(i);
        i_level = tree(Label(i),2);
        parent = tree(i_label,1);
        while i_level>0
            DataMod{i_level} = [DataMod{i_level};Data(i,:)];
            LabelMod{i_level} = [ LabelMod{i_level};i_label];
            i_label = parent;
            i_level = tree(parent,2);
            parent = tree(i_label,1);
        end
    end
%% ����  Ҷ�ӽ����ÿһ��������������μ��븸�ڵ��ṹ�У�ͬ����������У�
%     Leafs = tree_LeafNode(tree);
%     N_leaf = length(Leafs);
%     for i=1:N_leafs
%         tmpID = (Label==Leafs(i));
%         tmpdata = Data(tmpID,:);
%         tmplabel = Label(tmpID,1);
%         i_level = tree(Leafs(i),2);
%         parent = tree(Leafs(i),1);
%         while i_level>0
%             DataMod{i_level} = [DataMod{i_level};tmpdata];
%             LabelMod{i_level} = [LabelMod{i_level};tmplabel];
%             i_label = parent;
%             i_level = tree(parent,2);
%             parent = tree(i_label,1);
%             tmpID = (Label==i_label);
%             tmpdata = Data(tmpID,:);
%             tmplabel = Label(tmpID,1);
%         end
%     end
end