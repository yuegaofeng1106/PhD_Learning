function F=CreatCTF(w,t)
%CTģ�͵�״̬ת�ƾ���
f1=1;
f2=sin(w*t)/w;
f3=(1-cos(w*t))/w;
f4=cos(w*t);
f5=sin(w*t);
F=[ f1 f2 0 -f3 ;
    0  f4 0 -f5 ;
    0 f3 f1  f2 ;
    0 f5 0 f4;];
end
