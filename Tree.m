clc
clear all
close all

%%%%%%%%% 1-  load data set
Data= xlsread('Data_airline3.xlsx');
Data=Data';
X=Data(1:end-1,:);
Y=Data(end,:);
DataNum=size(X,2);
InputNum=size(X,1);

%% 2- Normaliziation

MinX = min(X);
MaxX = max(X);

MinY = min(Y);
MaxY = max(Y);

for ii= 1: InputNum
    X(:,ii)= normalize_fcn(X(:,ii),MinX(ii),MaxX(ii));
end

% for ii = 1:OutputNum
%     Y(:,ii) = Normalize_Fcn(Y(:,ii),MinY(ii),MaxY(ii));
% end

%%%%%%%%% 3-  Create test and train data
TrPercent = 60;
TrNum = round(DataNum * TrPercent / 100);
TsNum = DataNum - TrNum;

% R = randperm(DataNum);
% trIndex = R(1 : TrNum);
% tsIndex = R(1+TrNum : end);

X=X';
Y=Y';
Xtr = X(1:end,:);
Ytr = Y(1:end,:);

Xts = X(TrNum+1:end,:);
Yts = Y(TrNum+1:end,:);

%%%%%%%%% 4-  Create the tree
tree = fitctree(Xtr,Ytr); 

%%%%%%%%% 5-  Test the network.
YtsNet = predict(tree,Xts);                             
% Yts=Yts*MaxYtr;
YtsNet=round(YtsNet);
%%%%%%%%% 6- result
figure(1)
plot(Yts,'-or');
hold on
plot(YtsNet,'-sb');
hold off
title('prediction')
legend('Target','Output')


T=Yts ;
O=YtsNet;
O=round(O);
Dif= T-O;
Ind = find(Dif==0);
Acc=(length(Ind))/(length(O))
