
%% preparing datasetopts = delimitedTextImportOptions("NumVariables", 15);
feature = data_feature("Desktop/numerical project/abalone.data", [1, Inf]);
label = data_label("Desktop/numerical project/abalone.data", [1, Inf]);

y = grp2idx(label);



output1=[];%label
output2=[];
output3=[];

N=max(size(label));
%% M=1, F=I=-1
for i=1:N
    if strcmp(label(i),'M')
        output1=[output1;1];
    elseif strcmp(label(i),'F')
        output1=[output1;-1];
    elseif strcmp(label(i),'I')
        output1=[output1;-1];        
    end
end

%% F=1, M=I=-1
for i=1:N
    if strcmp(label(i),'F')
        output2=[output2;1];
    elseif strcmp(label(i),'M')
        output2=[output2;-1];
    elseif strcmp(label(i),'I')
        output2=[output2;-1];        
    end
end

%% I=1, F=M=1
for i=1:N
    if strcmp(label(i),'I')
        output3=[output3;1];
    elseif strcmp(label(i),'F')
        output3=[output3;-1];
    elseif strcmp(label(i),'M')
        output3=[output3;-1];        
    end
end

%%

X = feature;
y1 = output1;
y2 = output2;
y3 = output3;

rand_num = randperm(size(X,1));
X_train = X(rand_num(1:round(0.8*length(rand_num))),:);
y_train = y(rand_num(1:round(0.8*length(rand_num))),:);
y1_train = y1(rand_num(1:round(0.8*length(rand_num))),:);
y2_train = y2(rand_num(1:round(0.8*length(rand_num))),:);
y3_train = y3(rand_num(1:round(0.8*length(rand_num))),:);

X_test = X(rand_num(round(0.8*length(rand_num))+1:end),:);
y_test = y(rand_num(round(0.8*length(rand_num))+1:end),:);
y1_test = y1(rand_num(round(0.8*length(rand_num))+1:end),:);
y2_test = y2(rand_num(round(0.8*length(rand_num))+1:end),:);
y3_test = y3(rand_num(round(0.8*length(rand_num))+1:end),:);

