%% �������Ƚ��
function [ A ] = tree_Ancestor( varargin)
% return the ancestor of node i
% every entry in tree points to its parent
% e.g. tree(i) = p, means node i's parent is p
% root is node 1, and root points to 0
% self�Ƿ�����Լ�������������Լ�����0
% 2016/5/3 �Ժ��޸Ĵ��룬��������������2����3��
   tree = varargin{1};
   tree=tree(:,1);
   i = varargin{2};
   if(length(varargin)==3)     
        self = varargin{3};
    else
        self = 0;
    end
    
A = [];
if (self)
    A(1) = i;
end

node = tree(i);
while (node~=0)
    A(end+1) = node;
    node = tree(node);
end

end

