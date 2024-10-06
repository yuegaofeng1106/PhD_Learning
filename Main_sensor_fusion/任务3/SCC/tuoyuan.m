function tuoyuan(C,S)
Mu=[0 0];
[V1,D1] = eig(C);
alpha=0.97;
r=sqrt(chi2inv(alpha,2));
 f_plot_ellipse(r,V1,D1,Mu,S);
end


function f_plot_ellipse(r,V,D,Mu,S)


y = linspace(-sqrt(r^2*D(2,2)),sqrt(r^2*D(2,2)),60);
% compute x
x(1,:) = sqrt((r^2-y(:).^2/D(2,2))*D(1,1));
x(1,:) = real(x(1,:));
%这只产生了半个椭圆，还要产生另一半（注意两条曲线的坐标旋转方向要一致），然后旋转，平移，画图：
Ellip = [x,-x(1,:)]; % x
Ellip(2,:) = [y,fliplr(y)]; %y
Ellip = Ellip'*inv(V); % rotate
Ellip(:,1) = Ellip(:,1)+Mu(1); %shift
Ellip(:,2) = Ellip(:,2)+Mu(2);

hold on;
plot(Ellip(:,1),Ellip(:,2),'LineStyle',S,'LineWidth',4);
legend('Kalman1滤波误差','Kalman2滤波误差','我们的滤波误差',"FontSize",18)
set(gca,"FontSize",18)
grid on;

%plot(Mu(1),Mu(2),'+'); %Plot center
end
