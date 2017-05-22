function [BestGen,BestFitness,TrainIData,TrainFitness,TestOData,TestFitness]=SGHS_AFCMAC(In_train,Out_train,in_test,out_test)

%以上[最优解的根，最优解结果，训练数据的最后测试结果，训练数据的最优解，测试数据的最后测试结果，测试数据的最优解]
%=HarmonySearch_CMAC（训练数据的自变量，训练数据的结果，测试数据的自变量，测试数据的结果）
%数据集都是以列代表一个样本
%%%%%%%%%%%%%%%%%%%%%%%HS里的参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NVAR NG NH MaxItr HMS HMCR PAR PARmin PARmax bwmin bwmax pfErr;

global HM NCHV fitness PVB BW HMCRm PARm HMCRv PARv HMCRsum PARsum;                                           

global BestIndex WorstIndex BestFit WorstFit currentIteration LP lp;      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global m nb block_num mns v_std wght u_exp Fi M uc xmin xmax Gu Gv input_n output_n Wnum;    %%CMAC参数
input_n = 2; %输入的维数
output_n = 1; %输出的维数
m = 4;  %CMAC的层数
nb = 2;  %每层的块数
block_num = m*nb;
M=(m * (nb - 1)+1);  %输入量化的块数（范围）
xmin = -1;      %输入的最小最大值，用于量化计算
xmax = 1;       %
% border = xmax*m/M;
border = 1;
mns = zeros(input_n,block_num);
v_std = zeros(input_n,block_num);
wght = zeros(output_n,block_num);
u_exp = zeros(input_n,block_num);
Fi = zeros(1,block_num);

Wnum = m*nb^input_n;  %权值矩阵的大小
uc = 0.1;      %CMAC中每块边界值在高斯函数中的值

Gumin = xmin-border;
Gumax = xmax+border;
Gvmin = 0;
Gvmax = 1;
Wmin = -2;
Wmax = 2;

% Gu = zeros(m,nb,input_n);   %CMAC每块高斯函数期望值
% Gv = ones(m,nb,input_n).*m/(2*sqrt(-log(uc)));  %标准差。根据公式计算
% cmac_w = zeros(output_n,Wnum);  %CMAC权值矩阵。这里似乎没用到，因为我把Gu、Gv、W都放入了和声变量里
PVB = [];  %和声变量的取值范围
pvbGv = [];
for rm = 1:1:m
   for rn = 1:1:nb
       PVB = [PVB;Gumin Gumax];%%%2.0->0.
       pvbGv = [pvbGv;Gvmin Gvmax];
   end
end
PVB = [PVB;pvbGv];
for rinput = 2:1:input_n
    PVB = [PVB;PVB];
end
for rcmac_w = 1:1:(block_num*output_n)
    PVB = [PVB;Wmin Wmax];
end
%以上是设置和声变量的取值范围PVB，顺序是PVB[Gu1;Gv1;Gu2;Gv2;cmac_w]
%这个范围参考了原CMAC的参数训练结果

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 

NVAR= m*nb*input_n*2+block_num*output_n;         %number of variables 变量数

NG=6;           %number of ineguality constraints 不等式约束？

NH=0;           %number of eguality constraints 等式约束

% MaxItr=40000;    % maximum number of iterations 迭代次数
MaxItr=4000;

HMS=40;          % harmony memory size 记忆库大小

% HMCR=0.9;       % harmony consideration rate  0< HMCR <1
% HMCR=0.9;

PARmin=0.4;      % minumum pitch adjusting rate 扰动概率

% PARmax=0.9;      % maximum pitch adjusting rate
PARmax=0.9;

% bwmin=0.0001;    % minumum bandwidth 带宽
bwmin=0.0001;

bwmax=1.0;      % maxiumum bandwidth

LP = 100;
HMCRm = 0.9;
PARm = 0.9;
HMCRv = 0.01;
PARv = 0.05;
BestIndex =1;

% PVB=[0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi];   % range of variables 变量取值范围


% /**** Initiate Matrix ****/

HM=zeros(HMS,NVAR); %和声库

NCHV=zeros(1,NVAR);  %临时生成新的HS变量

BestGen=zeros(1,NVAR);  %最优解

fitness=zeros(1,HMS);  %和声 解向量

BW=zeros(1,NVAR);  %真正带宽


% warning off MATLAB:m_warning_end_without_block


MainHarmony;

for gi = 1:1:input_n
    for gj = 1:1:block_num
        Gu(gi,gj) = BestGen(2*(gi-1)*block_num+gj);
        Gv(gi,gj) = BestGen((2*(gi-1)+1)*block_num+gj);
    end
end

for go = 1:output_n
    for gj = 1:block_num 
        wght(go,gj) = BestGen(block_num*input_n*2+(go-1)*block_num+gj);
    end
end

save SGHS_Gu Gu;
save SGHS_Gv Gv;
save SGHS_wght wght;

%%%%%%%%%%%%%%%%%%%其他进化算法可参照此CMAC%%%%%%%%%%%%%%%%%%%%%%
% /*********************CMAC网络*************************/
    function valcmac = H_CMAC(sol,data_in)
        valcmac = [];       
        
        for k1 = 1:1:size(data_in,2) %遍历每个样本
            u = data_in(:,k1); 
            for i = 1:1:input_n   %遍历输入的每维
                for j = 1:block_num    
                    mns(i,j) = sol(2*(i-1)*block_num+j);
                    v_std(i,j) = sol((2*(i-1)+1)*block_num+j);
                    u_exp(i,j) = exp(-(u(i)-mns(i,j)).^2/(v_std(i,j).^2));
                end
            end
            for o = 1:output_n
                for j = 1:block_num 
                    wght(o,j) = sol(block_num*input_n*2+(o-1)*block_num+j);
                end
            end
            Fi = prod(u_exp); 
            output = Fi*wght';

            valcmac = [valcmac output]; %最终的训练样本的输出矩阵
            
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function sumf =Fitness(sol)

        Val = H_CMAC(sol,In_train);
      sum1 = (Val-Out_train).^2;   %每个样本的平方误差
%         sum = 0.6224*sol(1)*sol(3)*sol(4)+1.7781*sol(2)*sol(3)^2+3.1661*sol(1)^2*sol(4)+19.84*sol(1)^2*sol(3)+ eg(sol);  %F(x) = f(x) + penalty 
        sumf = sqrt(mean(sum1(:)));  %以平均方差作为函数值
       

    end


    function initialize                         %初始化

        % randomly initialize the HM

        for i=1:HMS

            for j=1:NVAR

                HM(i,j)=randval(PVB(j,1),PVB(j,2));  %生成随机变量

            end
%             hm = HM;
            fitness(i) = Fitness(HM(i,:));  %%F(x) = f(x) + penalty 

        end

    end


%/*******************************************/


    function MainHarmony

        % global NVAR NG NH MaxItr HMS HMCR PARmin PARmax bwmin bwmax;

        % global HM NCHV fitness PVB BW gx currentIteration;

       pfErr = zeros(size(MaxItr));%%看收敛

        initialize;             %初始化

        currentIteration  = 0;  %迭代次数
        lp = 1;
        HMCRsum = [];
        PARsum = [];
       
        while(currentIteration < MaxItr)%迭代次数

            HMCR = normrnd(HMCRm,HMCRv);
            PAR = normrnd(PARm,PARv);
%             PAR=(PARmax-PARmin)/(MaxItr)*currentIteration+PARmin; %扰动概率生成，随着迭代次数增加，PAR变大

%             coef=log(bwmin/bwmax)/MaxItr;
%             for pp =1:NVAR
% 
%                 BW(pp)=bwmax*exp(coef*currentIteration); %带宽
% 
%             end
            if currentIteration<(MaxItr/2)
                for pp =1:NVAR
                    bwmax = (PVB(pp,2)-PVB(pp,1))/10;
                    bwmin = (PVB(pp,2)-PVB(pp,1))/10000;
                    BW(pp) = bwmax - (bwmax-bwmin)*2*currentIteration/MaxItr;
%                     BW(pp)=bwmax*exp(coef*currentIteration); %带宽

                end
            else
                for pp =1:NVAR
                    BW(pp) = bwmin;
                end
            end

            % improvise a new harmony vector 临时生成一个和声向量

            for i =1:NVAR

                ran = rand(1);

                if( ran < HMCR ) % memory consideration

                    index = randint(1,HMS);  %在HMS中随机挑选一个变量
                                        
                    NCHV(i) = HM(index,i);
                    
                    ra = rand(1);
                    resulta = NCHV(i);
                    if( ra < 0.5)
                        resulta =resulta+  rand(1) * BW(i);  %BW为真正带宽
                        if( resulta < PVB(i,2))
                            NCHV(i) = resulta;
                        end
                    else
                        resulta =resulta- rand(1) * BW(i);
                        if( resulta > PVB(i,1))
                            NCHV(i) = resulta;
                        end
                    end

                    pvbRan = rand(1);

                    if( pvbRan < PAR) % pitch adjusting

                        NCHV(i) = HM(BestIndex,i);

                    end  %if( pvbRan < PAR) 扰动

%                     ra = rand(1);
%                     resulta = NCHV(i);
%                     if( ra < 0.5)
%                         resulta =resulta+  rand(1) * BW(i);  %BW为真正带宽
%                         if( resulta < PVB(i,2))
%                             NCHV(i) = resulta;
%                         end
%                     else
%                         resulta =resulta- rand(1) * BW(i);
%                         if( resulta > PVB(i,1))
%                             NCHV(i) = resulta;
%                         end
%                     end
                    
                else

                    NCHV(i) = randval( PVB(i,1), PVB(i,2) ); % random selection 在允许范围内生成变量

                end  %if( ran < HMCR )

            end  %for i =1:NVAR

            newFitness = Fitness(NCHV);  %%F(x) = f(x) + penalty 

            UpdateHM( newFitness );  %更新和声库

           
            pfErr(currentIteration+1) = mean(fitness(:));  %记录结果，用于画收敛图
%             pfErr(currentIteration+1) = BestFit;

            if lp == LP
                HMCRm = mean(HMCRsum(:));
                PARm = mean(PARsum(:));
                HMCRsum = [];
                PARsum = [];
                lp = 1;
            else
                lp = lp+1;
            end
            currentIteration=currentIteration+1;
%             if currentIteration==3000
%                 stopp=0;
%             end

        end % end while
        
        epochs = linspace(1,MaxItr,MaxItr);%步数
        figure(4);
        plot(epochs,pfErr);  %画收敛图
        BestFitness = min(fitness);
        %%%用最终得到的CMAC测试训练数据集
        TrainIData = H_CMAC(BestGen,In_train);
        trsum = (TrainIData - Out_train).^2;
        TrainFitness = sqrt(mean(trsum(:)));
        %%%用最终得到的CMAC测试测试数据集
        TestOData = H_CMAC(BestGen,in_test);
        tsum = (TestOData - out_test).^2;
        TestFitness = sqrt(mean(tsum(:)));
        
        
    end

% /*****************************************/

%%%%更新和声库
    function UpdateHM( NewFit )

        % global NVAR MaxItr HMS ;

        % global HM NCHV BestGen fitness ;

        % global BestIndex WorstIndex BestFit WorstFit currentIteration;

       

        if(currentIteration==0)

            BestFit=fitness(1);
            BestIndex =1;

            for i = 1:HMS   %找到解向量中的最优解

                if( fitness(i) < BestFit )

                    BestFit = fitness(i);

                    BestIndex =i;

                end

            end

           

            WorstFit=fitness(1);
            
            WorstIndex =1;
            
            for i = 1:HMS    %找到解向量中的最差解

                if( fitness(i) > WorstFit )

                    WorstFit = fitness(i);

                    WorstIndex =i;

                end

            end

        end   %if(currentIteration==0)

        if (NewFit< WorstFit)

           

            if( NewFit < BestFit )  %新向量比原来和声向量中的最优还优

                HM(WorstIndex,:)=NCHV;

%                 BestGen=NCHV;

                fitness(WorstIndex)=NewFit;
                
                BestFit = NewFit;

                BestIndex=WorstIndex;   %因为新向量是最优的且代替了最差向量

            else

                HM(WorstIndex,:)=NCHV;

                fitness(WorstIndex)=NewFit;

            end

           

           

            WorstFit=fitness(1);

            WorstIndex =1;

            for i = 1:HMS  %找到解向量中的最差解

                if( fitness(i) > WorstFit )

                    WorstFit = fitness(i);

                    WorstIndex =i;

                end

            end
            HMCRsum = [HMCRsum HMCR];
            PARsum = [PARsum PAR];
           

        end
        BestGen=HM(BestIndex,:);
    end % main if / function UpdateHM

end %function HarmonySearch


% /*****************************************/

function val1=randval(Maxv,Minv)

    val1=rand(1)*(Maxv-Minv)+Minv;

end


function val2=randint(Maxv,Minv)

    val2=round(rand(1)*(Maxv-Minv)+Minv);

end

