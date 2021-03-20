% LMST sin function data:
% y =sin(2*pi((n-1)*delt + 1)) for n=1:5000 and delt=0.01
clear;clc
% path(path,'D:\Polyspace\R2019b\toolbox\matlab\iofun\');
M = csvread('E:\VM_Share\Python_Dev\LSTM-Neural-Network-for-Time-Series-Prediction-master\data\sinewave.csv',1,0)
n = 1:5001;
delt = 0.01;
phi = 1;
y_formula= sin(2*pi*((n-1)*delt) + phi);
figure(1)
plot(M,'b')
title('csv original data')
axis([1,5002,-1,1])
figure(2)
plot(n,y_formula,'r+')
title('formula expression data')
axis([1,5002,-1,1])