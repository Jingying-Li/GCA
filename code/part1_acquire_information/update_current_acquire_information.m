clc;
clear;
%��ȡÿ��Ӱ���ϸ�������basic unit����Ϣ������ȡ�������֮��Ŀռ��ϵ(���ڣ���Χ������Χ)
%% configs
imagepath='D:\A_userfile\lly\data\gid5\image_rename\';
labelpath='D:\A_userfile\lly\data\gid5\label\';
datapath='D:\A_userfile\generate_image\data_information\';
labmicrDir=dir([labelpath '*.png']);
savepath='D:\A_userfile\generate_image\data_information_update\';
gray_value=1:5; %��ǰ���ݼ���Ӧ���ĻҶ�ֵ
ignored_value=0;
s=512; %ѵ�����ݵĴ�С

%% initialization
All_Intrainfor=[];%�洢basic unit information form all raw observed image
% Difclass_pixelsnumall��¼����ԭʼӰ���Ӧ�������ظ�������Ϊ����3���ַ���struct��
%  (1)Image_index:��Ӧ���ݼ�������ֵ
%  (2)Current_class:�������
%  (3)Pixelsnum:����Ӧ����������
Difclass_pixelsnumall=[]; 
%��ȡ���ݼ��еĸ��������ڽ���Ϣ
adjcant_relationship=zeros(length(gray_value));%����
contain_relationship=zeros(length(gray_value));%��Χ
containwith_relationship=zeros(length(gray_value));%����Χ
pro=0.0005; %��ֵ

%% ��ȡ���ݼ��ϸ��໥���ཻ��������Ϣ(regionprops)�Լ������Ϣ����
% ������(geo_relationship):��ȡÿ��Ӱ���ϸ��໥���ཻ��������Ϣ�Լ������Ϣ
for k=1:length(labmicrDir)
    image_name_micro=labmicrDir(k).name;
    label=imread([labelpath image_name_micro]);
    image=imread([imagepath image_name_micro]);
    %��ȡӰ���basic unit��Ϣ�Լ��������֮���λ�ù�ϵ��Ϣ
    [Intrainfor,Difclass_pixelsnum,AreaRelationship]=...
        geo_relationship(gray_value,image,label,ignored_value,pro,k);
    %��ÿ��ԭʼ���ݻ�ȡ������ϢSubregion_intrainfor��Subregion_interinfor�ϲ���Allsubregion_intrainfor��Allsubregion_interinfor
     All_Intrainfor=[All_Intrainfor Intrainfor];
     Difclass_pixelsnumall=[Difclass_pixelsnumall Difclass_pixelsnum];
     adjcant_relationship=adjcant_relationship+AreaRelationship.adjcant_relationship;
     contain_relationship=contain_relationship+AreaRelationship.contain_relationship;
     containwith_relationship=containwith_relationship+AreaRelationship.containwith_relationship; 
end

%% extracting_information������Ϣ�������������ݼ�����ֵ����Ľṹ����Ϣ����Ϊ�������ݼ������Ϣ����Ľṹ��
[basicunit_infor,Rawimage_pixelsnum]=extracting_information(All_Intrainfor,Difclass_pixelsnumall,gray_value);
% output:
%  basicunit_infor(����10���ֶΣ���СΪ1*length(gray_value)�Ľṹ�壬����Ϊ���ֵ)����¼ÿ����������������Ϣ
%  Rawimage_pixelsnum(���������ֶ�raw_imageid��pixels_num,��СΪ1*length(gray_value)�Ľṹ�壬����Ϊ���ֵ)����¼ÿ������Ӧ����������
upbasicunit_infor=basicunit_infor;
% �Ƴ�ָ���ֶ�
upbasicunit_infor = rmfield(upbasicunit_infor, 'allcurclass_centroid');
upbasicunit_infor = rmfield(upbasicunit_infor, 'allcurclass_SameclassDis');
upbasicunit_infor = rmfield(upbasicunit_infor, 'allcurclass_Boundaryjud');
upbasicunit_infor = rmfield(upbasicunit_infor, 'allcurclass_mindis');
upbasicunit_infor = rmfield(upbasicunit_infor, 'allcurclass_minbox');

%% upbasicunit_infor����¼ÿ������Ӧ��basic unit���� 
% ���ã���basicunit_infor�������С����s*s��С�Ľ��ж��δ�����֤���е�basic unit��s*s��
%       ������ȡ�����µ�basic unit��Ϣ���ϵ�upbasicunit_infor�ṹ����
minbox_diffclass=cell(length(gray_value),1);
for k1=1:length(gray_value)
    % Step 1: Get minimum size information for the basic unit of the current class
    allboundbox = basicunit_infor(k1).allcurclass_BoundingBox;
    % ���һ��Լ��������Ϊ��֤basic unit����Ч���Լ�Ч����С�ߴ粻��С��2^5
    allboundbox_limit = find(allboundbox(:,3)>=2^5 & allboundbox(:,4)>=2^5);
    allboundboxlimit = allboundbox(allboundbox_limit,3:4);
    allboundboxlimit_s = allboundboxlimit(:,1).*allboundboxlimit(:,2);
    basic_windows=allboundboxlimit(allboundboxlimit_s==min(allboundboxlimit_s),:);
    minbox_diffclass{k1,1}=basic_windows; %��¼���ṹ��minbox_diffclass 
    
    % Step 2: Get all basic unit sizes that exceed s*s
    original_allid=basicunit_infor(k1).alloriginal_id; %��ȡbasic unit��Ӧԭʼ���ݵ���������
    original_allsparity=basicunit_infor(k1).allcurclass_sparity; %��ȡbasic unit��Ӧ��ϡ�����Ϣ
    allbasic_boxinfor=[basicunit_infor(k1).allcurclass_BoundingBox]; %��ȡbasic unit��Ӧ��box��Ϣ
    statsbasic_id = find(allbasic_boxinfor(:,3)>s | allbasic_boxinfor(:,4)>s); % ɸѡbasic unit�����гߴ糬��s����������
    
    % Step 3: Generate optimized size combinations
    row_allpos=unique([(1:fix(s/ basic_windows(1))).*basic_windows(1) s]); % m_p���д�С
    col_allpos=unique([(1:fix(s/ basic_windows(2))).*basic_windows(2) s]); % n_p�����д�С
    combinations = combvec(row_allpos, col_allpos)';  % Generate all combinations
    products = prod(combinations, 2);% Calculate the product of each combination
    differences = abs(combinations(:, 1) - combinations(:, 2));
    [~, sorting_index] = sortrows([products, -differences]);
    combinations_sorted = combinations(sorting_index, :);% Apply the sorting index to combinations and products
    
    % Initialize variables to store the updated basic unit information
    updata_basicoriginalid=[];
    updata_basicarea=[];
    updata_basicboundingbox=zeros(0, 4);% Preallocate with correct column size
    updata_basicsparity=[];
    updata_basiccountshistogram=zeros(0, 256);
    
    % Inner loop for statsbasic_id elements
    for k2=1:length(statsbasic_id)
        tic;
        originalid = original_allid(statsbasic_id(k2)); %��ȡ��ǰbasic unit��Ӧԭʼ���ݵ�����ֵ
        inibasic_box = ceil(allbasic_boxinfor(statsbasic_id(k2),:)-[0 0 1 1]);
        curimage=imread([imagepath labmicrDir(originalid).name]);
        curlabel = imread([labelpath labmicrDir(originalid).name]);
        curbasicimage = imcrop(curimage,inibasic_box);
        curbasiclabel = imcrop(curlabel,inibasic_box);
        curbasiclabel(curbasiclabel~=k1)=0;
        curbasiclabel_copy=curbasiclabel;
        [row_p,col_p]=size(curbasiclabel);
        inibasic_sparity = sum((curbasiclabel(:)==k1))/(row_p*col_p); %��ȡ��ǰbasic unit��ϡ���
        
        % Filter combinations based on current dimensions
        filtered_combinations=combinations_sorted;
        if row_p <= s && col_p > s
           stats_comsort=find(combinations_sorted(:,1)<=row_p);
           filtered_combinations=combinations_sorted(stats_comsort,:);
        end
        if col_p <= s && row_p > s
           stats_comsort=find(combinations_sorted(:,2)<=col_p);
           filtered_combinations=combinations_sorted(stats_comsort,:);
        end
        
        % Evaluate each combination
        for k3=1:size(filtered_combinations,1)
            k_com1=size(filtered_combinations,1)+1-k3;%����
            basic_row=filtered_combinations(k_com1,1);%���ȵĴ�СΪ(basic_row,basic_column��
            basic_column=filtered_combinations(k_com1,2);
            n1=basic_row-1;
            n2=basic_column-1;                        
            h_row=round(basic_row/8*7);%����ʱ���в���
            h_column=round(basic_column/8*7);%����ʱ���в���
            temp_k=k1*ones(basic_row,basic_column);
            h_rowallpos=unique([1:h_row:row_p-n1 max(row_p-s+1,1)]);
            h_colallpos=unique([1:h_column:col_p-n2 max(col_p-s+1,1)]);
            
            % Nested loops for position
            for i_row=1:length(h_rowallpos)
                for j_col=1:length(h_colallpos)
                    i=h_rowallpos(i_row);
                    j=h_colallpos(j_col);
                    diff=abs(double(curbasiclabel(i:i+n1,j:j+n2))-temp_k);
                    
                    % Check if this region satisfies sparsity criteria
                    if sum(diff(:)==0)>=basic_row*basic_column * inibasic_sparity
                        points_tooriginal=inibasic_box(1:2)+[j-1 i-1];
                        curbasic_boundingbox=[points_tooriginal n2+1 n1+1];
                        curbasic_sparity=sum(diff(:)==0)/(basic_row*basic_column);
                        curbasic_orignalid=originalid;
                        curbasic_area=sum(diff(:)==0);
                        %����ֲ�
                        currenregion_image=imcrop(curbasicimage,[j i n2+1 n1+1]);
                        currenregion_label=imcrop(curbasiclabel_copy,[j i n2+1 n1+1]);
                        currenregion_label_logits=currenregion_label==k1;
                        current_onlyoneclassimage=currenregion_image(currenregion_label_logits);
                        [countshistogram,~] = histcounts(current_onlyoneclassimage, 256);
                        curbasiclabel(i:i+n1/8*7,j:j+n2/8*7)=0;
                    end
                end
            end
        end %���������ȴ�С�ı���
        toc
    end
    
    %% Consolidate updated info back to upbasicunit_infor
    upid=upbasicunit_infor(k1).alloriginal_id;
    uparea=upbasicunit_infor(k1).allcurclass_area;
    upBoundingBox=upbasicunit_infor(k1).allcurclass_BoundingBox;
    upsparity=upbasicunit_infor(k1).allcurclass_sparity;
    upcountshistogram=upbasicunit_infor(k1).allcurclass_countshistogram;
    
    % Remove outdated information
    upid(statsbasic_id)=[];
    uparea(statsbasic_id)=[];
    upBoundingBox(statsbasic_id,:)=[];
    upsparity(statsbasic_id)=[];
    upcountshistogram(statsbasic_id)=[];
    
    % Append optimized information
    upbasicunit_infor(k1).alloriginal_id=[upid updata_basicoriginalid];
    upbasicunit_infor(k1).allcurclass_area=[uparea updata_basicarea];
    upbasicunit_infor(k1).allcurclass_BoundingBox=[upBoundingBox;updata_basicboundingbox];
    upbasicunit_infor(k1).allcurclass_sparity=[upsparity updata_basicsparity];
    upbasicunit_infor(k1).allcurclass_countshistogram=[upcountshistogram updata_basiccountshistogram];
end

%% Obtain the set of possible category combinations within an s��s region of the original training dataset 
  
Relationship.adjcant_relationship=adjcant_relationship;
Relationship.contain_relationship=contain_relationship;
Relationship.containwith_relationship=containwith_relationship;

%% ��Ϣ����
save([savepath 'simi_matrix.mat'], 'simi_matrix', '-v7.3');
save([savepath 'Difclass_pixelsnumall.mat'],'Difclass_pixelsnumall')
save([savepath 'All_Intrainfor.mat'],'All_Intrainfor')
save([savepath 'basicunit_infor.mat'],'basicunit_infor')
save([savepath 'upbasicunit_infor.mat'],'upbasicunit_infor')
save([savepath 'Rawimage_pixelsnum.mat'],'Rawimage_pixelsnum')
save([savepath 'Relationship.mat'],'Relationship')
save([savepath 'minbox_diffclass.mat'],'minbox_diffclass1')