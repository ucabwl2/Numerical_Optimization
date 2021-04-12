function Y=Ker_Polynomial(X1,X2)
global poly_con
Y=zeros(size(X1,1),size(X2,1));%Gram Matrix
for i=1:size(X1,1)
    for j=1:size(X2,1)
        Y(i,j)=(1+dot(X1(i,:),X2(j,:))).^poly_con;
    end
end
return