function [BestGen,BestFitness,TrainIData,TrainFitness,TestOData,TestFitness]=SGHS_AFCMAC(In_train,Out_train,in_test,out_test)

%����[���Ž�ĸ������Ž�����ѵ�����ݵ������Խ����ѵ�����ݵ����Ž⣬�������ݵ������Խ�����������ݵ����Ž�]
%=HarmonySearch_CMAC��ѵ�����ݵ��Ա�����ѵ�����ݵĽ�����������ݵ��Ա������������ݵĽ����
%���ݼ��������д���һ������
%%%%%%%%%%%%%%%%%%%%%%%HS��Ĳ���%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global NVAR NG NH MaxItr HMS HMCR PAR PARmin PARmax bwmin bwmax pfErr;

global HM NCHV fitness PVB BW HMCRm PARm HMCRv PARv HMCRsum PARsum;                                           

global BestIndex WorstIndex BestFit WorstFit currentIteration LP lp;      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global m nb block_num mns v_std wght u_exp Fi M uc xmin xmax Gu Gv input_n output_n Wnum;    %%CMAC����
input_n = 2; %�����ά��
output_n = 1; %�����ά��
m = 4;  %CMAC�Ĳ���
nb = 2;  %ÿ��Ŀ���
block_num = m*nb;
M=(m * (nb - 1)+1);  %���������Ŀ�������Χ��
xmin = -1;      %�������С���ֵ��������������
xmax = 1;       %
% border = xmax*m/M;
border = 1;
mns = zeros(input_n,block_num);
v_std = zeros(input_n,block_num);
wght = zeros(output_n,block_num);
u_exp = zeros(input_n,block_num);
Fi = zeros(1,block_num);

Wnum = m*nb^input_n;  %Ȩֵ����Ĵ�С
uc = 0.1;      %CMAC��ÿ��߽�ֵ�ڸ�˹�����е�ֵ

Gumin = xmin-border;
Gumax = xmax+border;
Gvmin = 0;
Gvmax = 1;
Wmin = -2;
Wmax = 2;

% Gu = zeros(m,nb,input_n);   %CMACÿ���˹��������ֵ
% Gv = ones(m,nb,input_n).*m/(2*sqrt(-log(uc)));  %��׼����ݹ�ʽ����
% cmac_w = zeros(output_n,Wnum);  %CMACȨֵ���������ƺ�û�õ�����Ϊ�Ұ�Gu��Gv��W�������˺���������
PVB = [];  %����������ȡֵ��Χ
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
%���������ú���������ȡֵ��ΧPVB��˳����PVB[Gu1;Gv1;Gu2;Gv2;cmac_w]
%�����Χ�ο���ԭCMAC�Ĳ���ѵ�����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 

NVAR= m*nb*input_n*2+block_num*output_n;         %number of variables ������

NG=6;           %number of ineguality constraints ����ʽԼ����

NH=0;           %number of eguality constraints ��ʽԼ��

% MaxItr=40000;    % maximum number of iterations ��������
MaxItr=4000;

HMS=40;          % harmony memory size ������С

% HMCR=0.9;       % harmony consideration rate  0< HMCR <1
% HMCR=0.9;

PARmin=0.4;      % minumum pitch adjusting rate �Ŷ�����

% PARmax=0.9;      % maximum pitch adjusting rate
PARmax=0.9;

% bwmin=0.0001;    % minumum bandwidth ����
bwmin=0.0001;

bwmax=1.0;      % maxiumum bandwidth

LP = 100;
HMCRm = 0.9;
PARm = 0.9;
HMCRv = 0.01;
PARv = 0.05;
BestIndex =1;

% PVB=[0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi;0 2*pi];   % range of variables ����ȡֵ��Χ


% /**** Initiate Matrix ****/

HM=zeros(HMS,NVAR); %������

NCHV=zeros(1,NVAR);  %��ʱ�����µ�HS����

BestGen=zeros(1,NVAR);  %���Ž�

fitness=zeros(1,HMS);  %���� ������

BW=zeros(1,NVAR);  %��������


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

%%%%%%%%%%%%%%%%%%%���������㷨�ɲ��մ�CMAC%%%%%%%%%%%%%%%%%%%%%%
% /*********************CMAC����*************************/
    function valcmac = H_CMAC(sol,data_in)
        valcmac = [];       
        
        for k1 = 1:1:size(data_in,2) %����ÿ������
            u = data_in(:,k1); 
            for i = 1:1:input_n   %���������ÿά
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

            valcmac = [valcmac output]; %���յ�ѵ���������������
            
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function sumf =Fitness(sol)

        Val = H_CMAC(sol,In_train);
      sum1 = (Val-Out_train).^2;   %ÿ��������ƽ�����
%         sum = 0.6224*sol(1)*sol(3)*sol(4)+1.7781*sol(2)*sol(3)^2+3.1661*sol(1)^2*sol(4)+19.84*sol(1)^2*sol(3)+ eg(sol);  %F(x) = f(x) + penalty 
        sumf = sqrt(mean(sum1(:)));  %��ƽ��������Ϊ����ֵ
       

    end


    function initialize                         %��ʼ��

        % randomly initialize the HM

        for i=1:HMS

            for j=1:NVAR

                HM(i,j)=randval(PVB(j,1),PVB(j,2));  %�����������

            end
%             hm = HM;
            fitness(i) = Fitness(HM(i,:));  %%F(x) = f(x) + penalty 

        end

    end


%/*******************************************/


    function MainHarmony

        % global NVAR NG NH MaxItr HMS HMCR PARmin PARmax bwmin bwmax;

        % global HM NCHV fitness PVB BW gx currentIteration;

       pfErr = zeros(size(MaxItr));%%������

        initialize;             %��ʼ��

        currentIteration  = 0;  %��������
        lp = 1;
        HMCRsum = [];
        PARsum = [];
       
        while(currentIteration < MaxItr)%��������

            HMCR = normrnd(HMCRm,HMCRv);
            PAR = normrnd(PARm,PARv);
%             PAR=(PARmax-PARmin)/(MaxItr)*currentIteration+PARmin; %�Ŷ��������ɣ����ŵ����������ӣ�PAR���

%             coef=log(bwmin/bwmax)/MaxItr;
%             for pp =1:NVAR
% 
%                 BW(pp)=bwmax*exp(coef*currentIteration); %����
% 
%             end
            if currentIteration<(MaxItr/2)
                for pp =1:NVAR
                    bwmax = (PVB(pp,2)-PVB(pp,1))/10;
                    bwmin = (PVB(pp,2)-PVB(pp,1))/10000;
                    BW(pp) = bwmax - (bwmax-bwmin)*2*currentIteration/MaxItr;
%                     BW(pp)=bwmax*exp(coef*currentIteration); %����

                end
            else
                for pp =1:NVAR
                    BW(pp) = bwmin;
                end
            end

            % improvise a new harmony vector ��ʱ����һ����������

            for i =1:NVAR

                ran = rand(1);

                if( ran < HMCR ) % memory consideration

                    index = randint(1,HMS);  %��HMS�������ѡһ������
                                        
                    NCHV(i) = HM(index,i);
                    
                    ra = rand(1);
                    resulta = NCHV(i);
                    if( ra < 0.5)
                        resulta =resulta+  rand(1) * BW(i);  %BWΪ��������
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

                    end  %if( pvbRan < PAR) �Ŷ�

%                     ra = rand(1);
%                     resulta = NCHV(i);
%                     if( ra < 0.5)
%                         resulta =resulta+  rand(1) * BW(i);  %BWΪ��������
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

                    NCHV(i) = randval( PVB(i,1), PVB(i,2) ); % random selection ������Χ�����ɱ���

                end  %if( ran < HMCR )

            end  %for i =1:NVAR

            newFitness = Fitness(NCHV);  %%F(x) = f(x) + penalty 

            UpdateHM( newFitness );  %���º�����

           
            pfErr(currentIteration+1) = mean(fitness(:));  %��¼��������ڻ�����ͼ
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
        
        epochs = linspace(1,MaxItr,MaxItr);%����
        figure(4);
        plot(epochs,pfErr);  %������ͼ
        BestFitness = min(fitness);
        %%%�����յõ���CMAC����ѵ�����ݼ�
        TrainIData = H_CMAC(BestGen,In_train);
        trsum = (TrainIData - Out_train).^2;
        TrainFitness = sqrt(mean(trsum(:)));
        %%%�����յõ���CMAC���Բ������ݼ�
        TestOData = H_CMAC(BestGen,in_test);
        tsum = (TestOData - out_test).^2;
        TestFitness = sqrt(mean(tsum(:)));
        
        
    end

% /*****************************************/

%%%%���º�����
    function UpdateHM( NewFit )

        % global NVAR MaxItr HMS ;

        % global HM NCHV BestGen fitness ;

        % global BestIndex WorstIndex BestFit WorstFit currentIteration;

       

        if(currentIteration==0)

            BestFit=fitness(1);
            BestIndex =1;

            for i = 1:HMS   %�ҵ��������е����Ž�

                if( fitness(i) < BestFit )

                    BestFit = fitness(i);

                    BestIndex =i;

                end

            end

           

            WorstFit=fitness(1);
            
            WorstIndex =1;
            
            for i = 1:HMS    %�ҵ��������е�����

                if( fitness(i) > WorstFit )

                    WorstFit = fitness(i);

                    WorstIndex =i;

                end

            end

        end   %if(currentIteration==0)

        if (NewFit< WorstFit)

           

            if( NewFit < BestFit )  %��������ԭ�����������е����Ż���

                HM(WorstIndex,:)=NCHV;

%                 BestGen=NCHV;

                fitness(WorstIndex)=NewFit;
                
                BestFit = NewFit;

                BestIndex=WorstIndex;   %��Ϊ�����������ŵ��Ҵ������������

            else

                HM(WorstIndex,:)=NCHV;

                fitness(WorstIndex)=NewFit;

            end

           

           

            WorstFit=fitness(1);

            WorstIndex =1;

            for i = 1:HMS  %�ҵ��������е�����

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

