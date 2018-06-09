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
%% ——  每一样本依次加入其父节点和祖父结点所在层结构中（没有打乱同父节点下的样本数据在data_array中的相对位置）
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
%% ——  叶子结点中每一类的样本数据依次加入父节点层结构中（同类别样本集中）
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