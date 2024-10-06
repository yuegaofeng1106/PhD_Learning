
%思路：产生一组信号，用两个传感器去测量，之后分别通过kalaman滤波得到滤波后的数据，将两组数据进行简单凸组合融合，对比结果并分析。

clc;clear;
T=1;
N=80/T;

X=zeros(4,N);
X1=zeros(4,N);
X2=zeros(4,N);
X1(:,1)=[-100,2,200,20];
X2(:,1)=[-100,2,200,20];
Z=zeros(2,N);
Z1=zeros(2,N);
Z1(:,1)=[X(1,1),X(3,1)];
Z2=zeros(2,N);
Z2(:,1)=[X(1,1),X(3,1)];
delta_w=1e-2;
Q1=delta_w*diag([0.5,1,0.5,1]) ;
R1=100*eye(2);
%这里认为两个传感器的过程噪声是一样的，测量噪声不同；
% Q2=delta_w*diag([0.5,1,0.5,1]);
R2=80*eye(2);

F=[1,T,0,0;0,1,0,0;0,0,1,T;0,0,0,1];
H=[1,0,0,0;0,0,1,0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for t=2:N
%     X(:,t)=F*X(:,t-1);
%     Z(:,t)=H*X(:,t);%无噪声时的滤波轨迹
% end
m=5000;
for j=1:m
for t=2:N
    X1(:,t)=F*X1(:,t-1)+sqrtm(Q1)*randn(4,1);
    Z1(:,t)=H*X1(:,t)+sqrtm(R1)*randn(2,1); %带噪声的轨迹
end
Xkf1=zeros(4,N);
Xkf1(:,1)=X1(:,1);
P01=eye(4);
for i=2:N
    Xn=F*Xkf1(:,i-1);
    P1=F*P01*F'+Q1;
    K=P1*H'*inv(H*P1*H'+R1);
    Xkf1(:,i)=Xn+K*(Z1(:,i)-H*Xn);
    P01=(eye(4)-K*H)*P1;
end
%P0是协方差矩阵;Xkf是滤波后的数据。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=2:N
%     X2(:,t)=F*X2(:,t-1)+sqrtm(Q1)*randn(4,1);
    Z2(:,t)=H*X1(:,t)+sqrtm(R2)*randn(2,1);  %
end
Xkf2=zeros(4,N);
Xkf2(:,1)=X1(:,1);
% P02=0.5*eye(4);%这里注意两者的误差矩阵是否相关？
P02=diag([0.5,1,0.5,1]);
for i=2:N
    Xn=F*Xkf2(:,i-1);
    P1=F*P02*F'+Q1;
    K=P1*H'*inv(H*P1*H'+R2);
    Xkf2(:,i)=Xn+K*(Z2(:,i)-H*Xn);
    P02=(eye(4)-K*H)*P1;
end

 for i=2:N,
%     P03=inv(P01)+inv(P02);
%     X3
      X3(:,i)=inv([inv(P01)+inv(P02)])*[inv(P01)]*Xkf1(:,i)+inv([inv(P01)+inv(P02)])*[inv(P02)]*Xkf2(:,i);
      P03=inv(P01)+inv(P02);
 end
end

  q1=0;
  q2=0;
  q3=0;
for j=1:m,
 for i=2:N,
     error_kalman1(i)=DIST(X(:,i),Z1(:,i));
     error_kalman2(i)=DIST(X(:,i),Z2(:,i));
     error_fusion(i)=DIST(X(:,i),X3(:,i));
 end
      q1=error_kalman1(i)+q1;
      q2=error_kalman2(i)+q2;
      q3=error_fusion(i)+q3;
end
error_kalman_1=sqrt(q1/(2*m))
error_kalman_2=sqrt(q2/(2*m))
error_fusion_1=sqrt(q3/(2*m))
figure
hold on;box on;
 plot(X1(1,:),X1(3,:),'-k*','LineWidth',2);
 plot(X3(1,2:80),X3(3,2:80),'-.b.','LineWidth',4);
 plot(Xkf1(1,:),Xkf1(3,:),':r+','LineWidth',4);
 plot(Xkf2(1,:),Xkf2(3,:),'--g*','LineWidth',4);
legend('Real trajectory','Fused trajectory','Filter1 trajectory','Filter2 trajectory',"FontSize",18)
xlabel("x","FontName","Times New Roman","FontSize",18)
ylabel("y","FontName","Times New Roman","FontSize",18)

figure
hold on; box on;
plot(error_kalman1,'g--','LineWidth',4)
plot(error_kalman2,'b-+','LineWidth',4)
plot(error_fusion,'r-.','LineWidth',4)
xlabel("x","FontName","Times New Roman","FontSize",18)
ylabel("y","FontName","Times New Roman","FontSize",18)
legend('Kalman1 filtering error','Kalman2 filtering error','Fusion error',"FontSize",18)
set(gca,"FontSize",18)

% 画椭圆
figure
hold on;
box on;
    tuoyuan([P01(1,1),P01(1,2);P01(2,1),P01(2,1)],'-')
    tuoyuan([P02(1,1),P02(1,2);P02(2,1),P02(2,1)],'--')
    tuoyuan([P1(1,1),P1(1,2);P1(2,1),P1(2,1)],'-.')




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 function dist=DIST(X1,X2);
 if length(X2)<=2
     dist=(X1(1)-X2(1))^2 + (X1(3)-X2(2))^2 ;
 else
     dist= (X1(1)-X2(1))^2 + (X1(3)-X2(3))^2 ;
 end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 end