clear all;clc;
H = [1 0 0 0;0 0 1 0];
R = [2500 0;0 2500];
load data.mat
MC=50;
ex1=zeros(MC,N);ex2=zeros(MC,N);
ey1=zeros(MC,N);ey2=zeros(MC,N);
%���ؿ��޷���
for mn=1:MC
    %ģ�ͳ�ʼ��
    Pi =[0.99,0.01;0.01,0.99];%ת�Ƹ���
    u1=1/2;   u2=1/2;%2��ģ�ͼ� ���ʴ�������
    U0 = [u1,u2];
    P0 =[ 100 0 0 0;0 1 0 0;0 0 100 0;0 0 0 1;];%��ʼЭ����
    X1_k_1=x0;X2_k_1=x0; %1-r(r=2)ÿ��ģ�͵�״̬��������
    P1=P0;P2=P0; %1-r(r=2)ÿ��ģ�͵�״̬��������
    %CVģ�Ϳ������˲�
    for i=1:400
        zA(:,i) = H*xA(:,i) + sqrt(R)*[randn,randn]';
        [P_kv,P_k_k_1v,X1_kv]=kalman(A1,G1,H,Q1,R,zA(:,i),x0,P0);
        X_kv(:,i)=X1_kv;
        x0=X1_kv;P0=P_kv;
    end
    %CTģ�Ϳ������˲�
    A2=CreatCTF(-pi/270,1);%�ı���ٶ�w
    x0 = [1000,10,1000,10]';
    P0 = [100 0 0 0;0 1 0 0 ;0 0 100 0;0 0 0 1];
    for i=1:400
        [P_kt,P_k_k_1t,X1_kt]=kalman(A2,G2,H,Q2,R,zA(:,i),x0,P0);
        X_kt(:,i)=X1_kt;
        x0=X1_kt;P0=P_kt;
    end
    ex1(mn,:)=X_kv(1,:)-xA(1,:);ey1(mn,:)=X_kv(3,:)-xA(3,:);
    ex2(mn,:)=X_kt(1,:)-xA(1,:);ey2(mn,:)=X_kt(3,:)-xA(3,:);
    %%%%%%%%%%IMM�˲���ʼ������%%%%%%% %%%%
    x0 = [1000,10,1000,10]';
    x1_k_1=x0;x2_k_1=x0; %1-r(r=2)ÿ��ģ�͵�״̬��������
    P1=P0;P2=P0; %1-r(r=2)ÿ��ģ�͵�״̬��������
    P0 = [100 0 0 0;
        0 1 0 0 ;
        0 0 100 0;
        0 0 0 1];
    for k = 1:400%1-400��
        %��ϸ���
        c1=Pi(1,1)*u1+Pi(2,1)*u2;
        c2=Pi(1,2)*u1+Pi(2,2)*u2;
        u11=Pi(1,1)*u1/c1;u12=Pi(1,2)*u1/c2;
        u21=Pi(2,1)*u2/c1;u22=Pi(2,2)*u2/c2;
        x1_m = x1_k_1*u11+x2_k_1*u21;
        x2_m = x1_k_1*u12+x2_k_1*u22;
   p1_k_1=(P1+(x1_k_1-x1_m)*(x1_k_1-x1_m)')*u11+(P2+(x2_k_1-x1_m)*(x2_k_1-x1_m)')*u21;
        p2_k_1=(P1+(x1_k_1-x2_m)*(x1_k_1-x2_m)')*u12+(P2+(x2_k_1-x2_m)*(x2_k_1-x2_m)')*u22;
        %״̬Ԥ��
        x1_pk1=A1*x1_m; x2_pk1=A2*x2_m;
        p1_k=A1*p1_k_1*A1'+G1*Q1*G1';
        p2_k=A2*p2_k_1*A2'+G2*Q2*G2';
        %Ԥ��вЭ�������
        zk=zA(:,k);
        v1=zk-H*x1_pk1; v2=zk-H*x2_pk1;
        Sv1=H*p1_k*H'+R;Sv2=H*p2_k*H'+R;
        like1=det(2*pi*Sv1)^(-0.5)*exp(-0.5*v1'*inv(Sv1)*v1);
        like2=det(2*pi*Sv2)^(-0.5)*exp(-0.5*v2'*inv(Sv2)*v2);
        %�˲�����
        K1=p1_k*H'*inv(Sv1); K2=p2_k*H'*inv(Sv2);
        xk1=x1_pk1+K1*v1;xk2=x2_pk1+K2*v2;
        P1=p1_k-K1*Sv1*K1';P2=p2_k-K2*Sv2*K2';
        %ģ�͸��ʸ���
        C=like1*c1+like2*c2;
        u1=like1*c1/C;u2=like2*c2/C;
        %�����ں�
        xk=xk1*u1+xk2*u2;
        %����
        x1_k_1=xk1;x2_k_1=xk2;
        X_imm(:,k)=xk;
        um1(k)=u1;um2(k)=u2;
    end
    ex3(mn,:)=X_imm(1,:)-xA(1,:);
    ey3(mn,:)=X_imm(3,:)-xA(3,:);
    UM1(mn,:)=um1;UM2(mn,:)=um2;
end
EX1=sqrt(sum(ex1.^2,1)/MC);EX2=sqrt(sum(ex2.^2,1)/MC);EX3=sqrt(sum(ex1.^2,1)/MC);
EY1=sqrt(sum(ey1.^2,1)/MC);EY2=sqrt(sum(ey2.^2,1)/MC);EY3=sqrt(sum(ey3.^2,1)/MC);
mex1=mean(ex1);mey1=mean(ey1);%CV
mex2=mean(ex2);mey2=mean(ey2);%CT
mex3=mean(ex3);mey3=mean(ey3);%IMM
Um1=mean(UM1);Um2=mean(UM2);
t=1:400;
figure(1)
plot(xA(1,t),xA(3,t),'k-' ,zA(1,t),zA(2,t),'g_',xA(1,t)+mex3(t),xA(3,t)+mey3(t),...
    'r.',xA(1,t)+mex1(t),xA(3,t)+mey1(t),'b*',xA(1,t)+mex2(t),xA(3,t)+mey2(t),'m:','MarkerSize',8,'LineWidth',4);
legend('��ʵֵ', '����ֵ', 'IMM�˲���', 'CVģ���˲���', 'CTģ���˲���',"FontSize",18);
xlabel("x","FontSize",18)
ylabel("y","FontSize",18)
set(gca,"FontSize",18)
figure(2)
subplot(2,1,1)
plot(t,mex1(t),'b:',t,mex2(t),'m.',t,mex3(t),'r*' ,"LineWidth",4);
title('X�����˲����',"FontSize",18)
xlabel("t(s)","FontName","Times New Roman","FontSize",18)
ylabel('λ�����(m)',"FontName","Times New Roman","FontSize",18)
legend('CVģ���˲�','CTģ���˲�','IMM�˲���',"FontSize",18);
set(gca,"FontSize",18)
subplot(2,1,2)
plot(t,mey1(t),'b:',t,mey2(t),'m.',t,mey3(t),'r*',"LineWidth",4);
legend('CVģ���˲�','CTģ���˲�','IMM�˲���',"FontSize",18);
xlabel("t(s)","FontName","Times New Roman","FontSize",18)
ylabel('λ�����(m)',"FontName","Times New Roman","FontSize",18)
title('Y�����˲����',"FontSize",18)
xlabel('t(s)'),ylabel('λ�����(m)',"FontSize",18);
set(gca,"FontSize",18)
figure(3)
subplot(2,1,1)
plot( t,EX1(t),'b:',t,EX2(t),'m.',t,EX3(t),'r*',"LineWidth",4);
title('X����RMSE',"FontSize",18)
xlabel("t(s)","FontName","Times New Roman","FontSize",18)
ylabel('λ�����(m)',"FontName","Times New Roman","FontSize",18)
legend('CVģ���˲�','CTģ���˲�','IMM�˲���',"FontSize",18);
set(gca,"FontSize",18)
subplot(2,1,2)
plot(t,EY1(t),'b:',t,EY2(t),'m.',t,EY3(t),'r*', "LineWidth",4);
legend('CVģ���˲�','CTģ���˲�','IMM�˲���',"FontSize",18);
title('Y����RMSE')
xlabel('t(s)'),ylabel('λ�����(m)',"FontSize",18);
set(gca,"FontSize",18)
figure(4)
plot(t,Um1(t),'b-',t,Um2(t),'m:',"LineWidth",4);
legend('CVģ���˲�','CTģ���˲�',"FontSize",18);
title('CV��CTģ�͸��ʱ仯')  
xlabel("x","FontName","Times New Roman","FontSize",18)
ylabel("y","FontName","Times New Roman","FontSize",18)
set(gca,"FontName","Times New Roman","FontSize",18)

