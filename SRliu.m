function [eigvector] = SRliu(options, Responses, data)
% SR: Spectral Regression

[nSmp,mFea] = size(data);
if mFea > nSmp
    ddata = full(data*data');
    if options.ReguAlpha > 0
        for i=1:size(ddata,1)
            ddata(i,i) = ddata(i,i) + options.ReguAlpha;
        end
    end
    
    ddata = max(ddata,ddata');
    R = chol(ddata);
    eigvector = R\(R'\Responses);    
    eigvector = data'*eigvector;
else
    ddata = full(data'*data);    
    if options.ReguAlpha > 0
        for i=1:size(ddata,1)
            ddata(i,i) = ddata(i,i) + options.ReguAlpha;
        end
    end
    
    ddata = max(ddata,ddata');
    B = data'*Responses;
    
    R = chol(ddata);
    eigvector = R\(R'\B);
end
eigvector = eigvector./repmat(max(1e-10,sum(eigvector.^2,1).^.5),size(eigvector,1),1);
end