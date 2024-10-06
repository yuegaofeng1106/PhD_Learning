%% CV模型
% xk = xk-1 + vxk * delta_T + 0.5*ax*delta_T^2
% vxk = vxk-1 + ax*delta_T
% yk = yk-1 + vyk * delta_T + 0.5*ay*delta_T^2
% vyk = vyk-1 + ay*delta_T
% X = [x;vx;y;vy];

close
clear all
cv_M = 30;
cv_X = zeros(4,cv_M);
cv_delta_t = 1;
cv_noise_sigma = [2  0;
                  0  1.5]; %加速度噪声 可自行调节，有不同的变化
cv_noise_sigma2 = cv_noise_sigma*cv_noise_sigma';       %方差      
cv_a_noise = cv_noise_sigma2 * randn(2,1);
cv_F = [1, cv_delta_t, 0, 0;
        0, 1, 0,          0;
        0, 0, 1, cv_delta_t;
        0, 0, 0,          1];
cv_G = [0.5*cv_delta_t^2, 0;
        cv_delta_t,       0;
        0, 0.5*cv_delta_t^2;
        0,       cv_delta_t];
cv_x0 = [50; 2; 70; 3];

subplot(1,3,1);
for i=1:cv_M
    cv_X(:,i) = cv_F*cv_x0 + cv_G*cv_a_noise;
    cv_x0 = cv_X(:,i);
    plot(cv_X(1,i),cv_X(3,i),'g*','MarkerSize',16);  
    text(cv_X(1,i),cv_X(3,i),num2str(i));
    hold on
end
xlabel("x","FontName","Times New Roman","FontSize",24)
ylabel("y","FontName","Times New Roman","FontSize",24)
set(gca,"FontName","Times New Roman","FontSize",24)
title('CV模型',"FontSize",24);
%% CA模型
%会涉及到变加速直线运动的公式
% a = a0 + r*t
% v = v0 + a0*t + 0.5*r*t^2
% s = v0*t + 0.5*a0*t^2 + (1/6)*r*t^3
% 状态值X = [x;vx;ax;y;vy;ay],6个值
% xk+1  = xk + vxk*t + 0.5*axk*t^2 + (1/6)*rx*t^3  rx符合正态分布
% vxk+1 = vxk + axk*t + 0.5*rx*t^2
% axk+1 = axk + rx*t
% yk+1  = yk + vyk*t + 0.5*ayk*t^2 + (1/6)*ry*t^3  ry符合正态分布
% vyk+1 = vyk + ayk*t + 0.5*ry*t^2
% ayk+1 = ayk + ry*t   
ca_delta_t = 1;
ca_M = 20;
ca_X = zeros(6,ca_M);
ca_x0 = [60;8;1;55;3;2]; %初值
ca_noise_sigma = [0.2   0;
                  0 0.3]; %噪声的标准差
ca_noise_sigma2 = ca_noise_sigma*ca_noise_sigma';%噪声的方差
ca_a_noise = ca_noise_sigma2*randn(2,1);
ca_F = [1, ca_delta_t, 0.5*ca_delta_t^2, 0, 0, 0;
     0,          1,       ca_delta_t, 0, 0, 0;
     0,          0,                1, 0, 0, 0;
     0, 0, 0, 1, ca_delta_t, 0.5*ca_delta_t^2;
     0,          0,       0, 0, 1, ca_delta_t;
     0,          0,                0, 0, 0, 1]; %系数矩阵
ca_G = [ (1/6)*ca_delta_t^3,  0;
          0.5*ca_delta_t^2,   0;
          ca_delta_t,         0;
          0,  (1/6)*ca_delta_t^3;
          0,   0.5*ca_delta_t^2;
          0,         ca_delta_t];
subplot(1,3,2); 
for i = 1:ca_M
    ca_X(:,i) = ca_F*ca_x0 + ca_G*ca_a_noise;
    ca_x0 = ca_X(:,i);
    plot(ca_X(1,i),ca_X(4,i),'b*','MarkerSize',16); 
    text(ca_X(1,i),ca_X(4,i),num2str(i));
    hold on
end
title('CA模型',"FontSize",24);
xlabel("x","FontName","Times New Roman","FontSize",24);
ylabel("y","FontName","Times New Roman","FontSize",24);
set(gca,"FontName","Times New Roman","FontSize",24);
delta_t = 1;
noise_sigma = [1 0 0; %x方向加速度的标准差值
               0 1 0; %y方向加速度的标准差值
               0 0 0.05];%  %角速度的标准差值。  标准差，只涉及到三个噪声而非五个
 %如果是5个噪声的话，分别是：x的加速度噪声，vx的加速度噪声，y的加速度噪声，vy的加速度的噪声，omega的噪声
 %如果是3个噪声的话，表明x和vx的噪声是同一个噪声，y和vy的噪声是同一个，omega的噪声是一个
 %因此，如果采用的是5个噪声，那么噪声系数必然是5列的；但是这里采用了3个噪声，因此噪声系数就是3列了
 %而我们的状态值选取的是[x ; vx ; y ; vy ; omega]是5行的，因此噪声系数是 5行3列的，与3行1列的噪声V相乘后得到5行1列
 
 % CT运动模型的公式或者说方程：恒定速度和恒定角速度。略去公式推导了
 % xk+1  = xk + (sinwt)/w * vxk - (1-coswt)/w * vyk + 0.5*ax*t^2  而ax符合N(0,noise_sigma_x)
 % vxk+1 = (coswt) * vxk - (sinwt) * vyk + ax * t
 % yk+1  = (1-coswt)/w * vxk + yk + (sinwt)/w * vyk + 0.5*ay*t^2  而ay符合N(0,noise_sigma_y)
 % vyk+1 = (sinwt) * vxk + (coswt) * vyk + ay * t
 % wk = wk+1 + aw*t%恒定角速度
 %写成矩阵形式就可以得到 运动系数F 和 噪声系数G
 % Xk+1 = F * Xk + G * a_noise
G = [ 0.5*delta_t^2, 0, 0;
      delta_t,       0, 0;
      0, 0.5*delta_t^2, 0;
      0,       delta_t, 0;
      0,       0, delta_t]; %为常系数，而F并不是常系数（如果w有变化的话）！？
noise_sigma2 = noise_sigma .* noise_sigma' ; %方差值
a_noise = noise_sigma2 * randn(3,1);  % 结果是得到3行1列的矩阵值,代表加速度值

%运动方程必须初始化初值即最开始的速度、位置、角速度值需要自己给定或者通过其他方式要获得，有初值才能递推
M = 20; %跟踪时长
X=zeros(5,M);%目标状态值 所有时刻下的
x0 = [100; 10; 100; 5; 0.02]; %初始坐标时(100,100),初始速度是(10,5),初始角速度是0.02 这是单个目标的
%x=x0(1,1),vx=x0(2,1),y=x0(3,1),vy=x0(4,1),w=x0(5,1)
% x0 = [100 20; 0 0; 100 50; 0 0; 0.01 0.02]; %这是两个目标的即跟踪两个目标

subplot(1,3,3); %单个目标下的CT运动模型
for i=1:M %遍历每个时刻即计算每个时刻的状态值
    sinwt = sin(x0(5,1)*delta_t);
    coswt = cos(x0(5,1)*delta_t);
    X(1,i) = x0(1,1) + sinwt*x0(2,1)/x0(5,1) - (1-coswt)*x0(4,1)/x0(5,1);
    X(2,i) = (coswt)*x0(2,1) -  sinwt*x0(4,1);
    X(3,i) = (1-coswt)*x0(2,1)/x0(5,1) + x0(3,1) + sinwt*x0(4,1)/x0(5,1);
    X(4,i) = sinwt*x0(2,1) + (coswt)*x0(4,1);
    X(5,i) = x0(5,1); %这是还未加入噪声时的
%     x0 = X(:,i); %实现迭代 
%     plot(X(1,i),X(3,i),'k.'); pause(0.5); hold on
    
    %加入噪声
    a_noise = noise_sigma2 * randn(3,1);
    X(:,i) = X(:,i) + G*a_noise;
    x0 = X(:,i); %实现迭代 
    plot(X(1,i),X(3,i),'r*','MarkerSize',16);
    text(X(1,i),X(3,i),num2str(i));
    hold on
end  
title('CT模型',"FontSize",24);
xlabel("x","FontName","Times New Roman","FontSize",24);
ylabel("y","FontName","Times New Roman","FontSize",24);
set(gca,"FontName","Times New Roman","FontSize",24);
%如果是多个目标，那么每个目标的noise_sigma2和x0都应该是不同的






