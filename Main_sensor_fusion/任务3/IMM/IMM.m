clear all;
clc
close all
load Measure.mat%加载量测数据
load Real.mat%加载真实值数据
Z=TimeRPhi(2:3,:);%量测值
Xr=TXYdXdYAxAyTPhi;%真值
 
T=0.1;
pai=[0.8 0.2;0.1 0.9];  %定义一步转移概率矩阵
miu_CV=0;            %匀速运动模型在初始时刻正确的概率
miu_CA=1;            %匀加速运动模型在初始时刻正确概率

%UKF滤波器初始化
alf=0.001;
beta=2;
i=6;
a=((alf)^2-1)*i;
Wm0=a/(i+a);
Wc0=a/(i+a)+(1-(alf)^2+beta);
Wm=1/(2*(i+a));  %求一阶统计特性时的权系数
Wc=1/(2*(i+a));  %求二阶统计特性时的权系数
