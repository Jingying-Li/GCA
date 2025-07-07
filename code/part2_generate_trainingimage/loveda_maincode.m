% clc;
% clear;
% %Ŀ�ģ�ȷ�������ռ��ȷ����module R��ʵ��(1.�����ȡbasic unit 2.����relationship��ϡ��ȣ��Լ�ѡȡ��basic unit����training image)
% %% config
% % imagepath='D:\ljy\gid_data\Land-use dataset��GID\gid5_change\image_rename\';%139
% % labelpath='D:\ljy\gid_data\Land-use dataset��GID\gid5_change\label\';
% imagepath='F:\lly\data_allinfor\loveda\Train\images_png\';
% labelpath='F:\lly\data_allinfor\loveda\Train\masks_png\';
% imageDir=dir([imagepath '*.png']);
% labelDir=dir([labelpath '*.png']);
% data_infor_path='E:\2025.228\data\generate_image_training_loveda��\code\new_augmis\data_information\';
% % data_infor_path='D:\ljy\generate_image\data_information\';
% save_similar_infor='D:\A_userfile\lly\data\lovada\code\uvaid_code\data_information\similar_infor\';
% save_micro_path='D:\A_userfile\lly\data\lovada\code\uvaid_code\generate\label\';
% save_image_path='D:\A_userfile\lly\data\lovada\code\uvaid_code\generate\image\';
% training_set_number=10088;%ѵ����������
% gray_value=1:7; %���ݼ���Ӧ�����Ҷ�ֵ
% s=512;  % ѵ�����ݵĴ�С512*512
% irr_class=[1,3,4];
% road_class=3; % the value of road
% car_class=[]; % the information of car
% clutter=1; %background

clc;
clear;
%Ŀ�ģ�ȷ�������ռ��ȷ����module R��ʵ��(1.�����ȡbasic unit 2.����relationship��ϡ��ȣ��Լ�ѡȡ��basic unit����training image)
%% config
% imagepath='D:\ljy\gid_data\Land-use dataset��GID\gid5_change\image_rename\';%139
% labelpath='D:\ljy\gid_data\Land-use dataset��GID\gid5_change\label\';
imagepath='F:\lly\data_allinfor\gid\image_rename\';
labelpath='F:\lly\data_allinfor\gid\label_gid15\';
imageDir=dir([imagepath '*.png']);
labelDir=dir([labelpath '*.png']);
data_infor_path='E:\2025.228\code_alldata\gid15_alltraining\gid15\data_information\';
% data_infor_path='D:\ljy\generate_image\data_information\';
save_similar_infor='E:\2025.228\code_alldata\gid15_alltraining\gid15\data_information\similar_infor\';

save_micro_path='F:\lly\compare_data_instance\gid15\DRAW_LABEL\';
save_image_path='F:\lly\compare_data_instance\gid15\DRAW_IMAGE\';
mkdir(save_micro_path)
mkdir(save_image_path)
training_set_number=3000;%ѵ����������
gray_value=1:15; %���ݼ���Ӧ�����Ҷ�ֵ
s=512;  % ѵ�����ݵĴ�С512*512
irr_class=[12,13];
road_class=12; % the value of road
car_class=[]; % the information of car
clutter=[]; %background

% gray_value=1:15; %���ݼ���Ӧ�����Ҷ�ֵ
% s=512;  % ѵ�����ݵĴ�С512*512
% irr_class=[1];
% road_class=1; % the value of road
% car_class=[]; % the information of car
% clutter=[]; %background

Category_counts=zeros(1,length(gray_value)+1);
for k=1:length(labelDir)
    image_name_micro=labelDir(k).name;
    label=imread([labelpath image_name_micro]);
    category_counts = histcounts(label(:), 0:max(gray_value)+1); % ��1���������� 0 ��ʼ
    Category_counts=Category_counts+category_counts;
end

%% ��Ϣ����
load([data_infor_path 'Rawimage_pixelsnum.mat'])
load([data_infor_path 'basicunit_infor.mat'])
load([data_infor_path 'upbasicunit_infor.mat'])
load([data_infor_path 'Relationship.mat'])
load([data_infor_path 'Prior_matrix.mat'])
load([data_infor_path 'minbox_diffclass.mat'])
load([data_infor_path 'Curclass_infor.mat'])
Classcol_infor=Curclass_infor;
empty_positions = find(cellfun(@(x) isempty(x), {Classcol_infor.classcol}));
Classcol_infor(empty_positions)=[];
%������������Ϣ�洢���ṹ����
Given_infor.imagepath=imagepath;
Given_infor.labelpath=labelpath;
Given_infor.training_set_number=training_set_number;
Given_infor.gray_value=gray_value;
%Given_infor.training_set_perclass_num=training_set_perclass_num;
Given_infor.basicunit_infor=basicunit_infor;
Given_infor.upbasicunit_infor=upbasicunit_infor;
Given_infor.Rawimage_pixelsnum=Rawimage_pixelsnum;
Given_infor.Prior_matrix=Prior_matrix;
Given_infor.s = s;
Given_infor.minbox_diffclass=minbox_diffclass;
Given_infor.adjcant_relationship=Relationship.adjcant_relationship;

%% ����������
allclass_containre=sum(Relationship.contain_relationship,2);
allclass_objectnum=cell2mat(cellfun(@(x) length(x),{basicunit_infor.alloriginal_id},...
    'UniformOutput', false));
contain_toclass=gray_value(allclass_containre./allclass_objectnum'>0.5);

%% ���ڳ�������ǿ�������ֶԵ���������
class_weight=sum(Category_counts(2:end))./Category_counts(2:end);
Prior_matrix=roundn(Prior_matrix,-3).*repmat(class_weight,[length(gray_value) 1]);%����3λС��
% Prior_matrix=roundn(Prior_matrix,-3);%����3λС��
Prior_matrix_all=Prior_matrix+Prior_matrix';
Prior_easy=triu(Prior_matrix_all)-diag(diag(Prior_matrix_all));
emisclassification_thres=roundn(sum(Prior_easy(:))/(sum(Prior_easy(:)~=0)),-2);
Prior_easy_thres=Prior_easy.*(Prior_easy>emisclassification_thres);
[em_stats_x,em_stats_y]=find(Prior_easy_thres~=0);
% ����XȡֵΪ1��length(gray_value)���п��ܵĸ���ֵ
%��1������ϵͳ�����õ���������п��ܹ�������������
classall_pro=[Classcol_infor.classcol_totalnum]./sum([Classcol_infor.classcol_totalnum]);
classall_pro_idx=find(classall_pro<0.0001);
classall_pro(classall_pro_idx)=0.0001;
%��2����ȡ���г�����𼯺϶�Ӧ����ǿȨ��
classcol_allpossible={Classcol_infor.classcol};
all_misclassified_pairs = [em_stats_x em_stats_y];%���y���Ϊ���x
all_misclassified_pairspro=Prior_easy_thres(Prior_easy_thres~=0);
all_misclassified_pairspro=all_misclassified_pairspro./sum(all_misclassified_pairspro);
augweight=zeros(length(Classcol_infor),1);
empair_num=zeros(length(Classcol_infor),1);
empair_allexits=cell(length(Classcol_infor),1);
empair_topro=cell(length(Classcol_infor),1);
for k_p=1:length(Classcol_infor)
    classcol_cur=Classcol_infor(k_p).classcol;
    empair_in_classcol_logits = sum(ismember(all_misclassified_pairs,classcol_cur),2)==2;
    if sum(empair_in_classcol_logits)==0
        augweight(k_p)=1;
        empair_num(k_p)=0;
        empair_topro{k_p}=0;
    else
        empair_in_classcol_topro=empair_in_classcol_logits.*all_misclassified_pairspro;
        empair_in_classcol_topro(empair_in_classcol_topro==0)=[];
        augweight(k_p)=sum(exp(empair_in_classcol_topro)');
        empair_num(k_p)=sum(empair_in_classcol_logits);
        empair_allexits{k_p}=all_misclassified_pairs(empair_in_classcol_logits,:);
        empair_topro{k_p}=exp(empair_in_classcol_topro)./sum(exp(empair_in_classcol_topro));
    end
end
classall_pro_weight=classall_pro.*augweight'./sum(classall_pro.*augweight');

%% ��������Ϊtraining_set_number��ѵ�����ݼ���
%% step1:���ɺ�����������Ե�ѵ�����ݣ�����¼δ������������Ե������Ϣ
%initialization
choose_basic_index=cell(length(gray_value),1);%��¼��ȡ��basic unit��Ϣ
choose_basicidx=cell(length(gray_value),1);%��¼��ȡ��basic unit��Ӧraw image��idx��Ϣ
record_class_infor=cell(training_set_number,1);
gray_pixelsnum=zeros(1,length(gray_value));%��¼����Ӱ��ĸ���������ظ���֮��
record_oneclass=zeros(length(gray_value),1);%��¼����Ӱ������ֻ����һ���Ӧ�ĸ���
record_curclasscol=cell(training_set_number,1);%��¼����Ӱ�����𼯺�
record_curclass_pro_normal=cell(training_set_number,1);%��¼����Ӱ��������ռ��������
record_curclass_areanum=cell(training_set_number,1);%��¼����Ӱ��ĸ����������������
record_lastclass=cell(training_set_number,1);
record_curclass_id=[];
withoeasy_classinfor=cell(training_set_number,1);
easyclass_infor=cell(training_set_number,1);
easypair_index=cell(length(gray_value));
% �����жϺ󣬼���������Ϣ��������
% load([save_image_path 'choose_basic_index.mat'])
% load([save_image_path 'record_class_infor.mat'])
% load([save_image_path 'choose_basicidx.mat'])
% load([save_image_path 'gray_pixelsnum.mat'])
% load([save_image_path 'record_curclasscol.mat'])
% load([save_image_path 'record_oneclass.mat'])
% load([save_image_path 'record_curclass_pro_normal.mat'])
% load([save_image_path 'record_curclass_areanum.mat'])
% load([save_image_path 'record_curclass_id.mat'])
% load([save_image_path 'withoeasy_classinfor.mat'])
% load([save_image_path 'easyclass_infor.mat'])
% load([save_image_path 'record_lastclass.mat'])
% load([save_image_path 'easypair_index.mat']) 
Rand_infor={}; 
for k=80:training_set_number
    %1.1 �������Ȩ�س�ȡ��ǰ��𼯺�
    seedrand=rand();
    choose_iniclass = find(seedrand <= cumsum(classall_pro_weight), 1, 'first');
    % ��ֶ��жϣ�����ȡ��ֶ���Ϣ��
    if empair_num(choose_iniclass)~=0 %������ֶ�
        empair_topro_cur=cell2mat(empair_topro(choose_iniclass));
        seedrand1=rand();%�ж��Ƿ�ǿ���������
        if seedrand1>0.5
            seedrand2=rand();
            chooseempairs_id = find(seedrand2 <= cumsum(empair_topro_cur), 1, 'first');
            exits_empairs=empair_allexits{choose_iniclass};
            chooseempairs=exits_empairs(chooseempairs_id,:);
            choose_class=chooseempairs(2);
            em_toclass=chooseempairs(1);%the class of  choose_class is misclassified as the class of  em_toclass.
            easyclass_infor{k}=chooseempairs;
        else
            choose_classcol=Classcol_infor(choose_iniclass).classcol;
            withoeasy_classinfor{k}=choose_classcol;
            choose_class=[];
            em_toclass=[];
            chooseempairs=[];
            if isempty(choose_class)
                continue;
            end
        end
        %choose_class_idx = find(sum(simi_easymisinfor>0.8,2)>1);%��ѡ����basic unit��������Ϣ
    else
        %��������ֶԣ������������ѡȡһ����Ӱ����Ϊѵ������
        choose_classcol=Classcol_infor(choose_iniclass).classcol;
        withoeasy_classinfor{k}=choose_classcol;
        choose_class=[];
        em_toclass=[];
        chooseempairs=[];
        continue;
    end
    % ���ݰ�����ϵ���������Ϣ˳��
    choose_classcol_initial = Classcol_infor(choose_iniclass).classcol;
    Curclasscol_infor = updatasort_accordingconrelationship(choose_classcol_initial,chooseempairs,Relationship,basicunit_infor);
    choose_classcol = Curclasscol_infor.classcol;
    em_toclass = Curclasscol_infor.em_toclass;
    choose_class = Curclasscol_infor.choose_class;
    choosepair_easy=[choose_class em_toclass];
    strr1=strcat('similarity_matrix_',num2str(choose_class),'_',num2str(em_toclass),'.mat');
    load([save_similar_infor strr1]);
    simi_matrix=similarity_matrix;
    choose_classcolareanum=Classcol_infor(choose_iniclass).classnum_todistribution;
    choose_classcolpro=Classcol_infor(choose_iniclass).classpro_todistribution;
    % 1.2 ���ݳ�ȡ��𼯺϶�Ӧ��������ֲ�ȷ����ǰ�������Լ���Ӧ�����ռ��
    choose_classcol_areanum=zeros(length(choose_classcol),1);
    choose_classcol_pro=zeros(length(choose_classcol),1);
    for kk=1:length(choose_classcol)
        %1.2.1 ���ݷֲ����ɵ�ǰ����Ӧ�������
        curclass_toareanum=choose_classcolareanum(kk,:);
        curclass_toareanum_logits= curclass_toareanum~=0;
        curclass_toareanum_validindex = find(curclass_toareanum_logits == 1,1, 'last');
        curclass_toareanum_valid = curclass_toareanum(1:curclass_toareanum_validindex);
        seedrand2=rand();
        choose_classnum = find(seedrand2 <= cumsum(curclass_toareanum_valid), 1, 'first');
        %1.2.2 ���ݷֲ����ɵ�ǰ����Ӧ��ռ��
        curclass_numtopro=choose_classcolpro{kk};
        curclass_topro=curclass_numtopro(choose_classnum,:);
        curclass_topro_logits= curclass_topro~=0;
        curclass_topro_validindex = find(curclass_topro_logits == 1,1, 'last');
        curclass_topro_valid = curclass_topro(1:curclass_topro_validindex);
        seedrand3=rand();
        choose_proidx = find(seedrand3 <= cumsum(curclass_topro_valid), 1, 'first');
        choose_pro=choose_proidx*0.1;
        choose_classcol_areanum(kk)=choose_classnum;
        choose_classcol_pro(kk)=choose_pro;
    end
    % update area number
    curclass_pro_normal=choose_classcol_pro./sum(choose_classcol_pro);
    %�������˳�򣬱�֤�����Ϊ����������������ȡbasic unit����
    choosepick_classcol=[intersect(choose_classcol,contain_toclass) choose_classcol(~ismember(choose_classcol,contain_toclass))];
    [~, new_indices] = ismember(choosepick_classcol, choose_classcol_initial);

    % ������˳������������������ռ������
    curclass_pro_normalchange = curclass_pro_normal(new_indices);
    choose_classcol_areanumchange= choose_classcol_areanum(new_indices);
    
    
    classcol_infor=update_area_number(Relationship,choosepick_classcol,curclass_pro_normalchange,choose_classcol_areanumchange,irr_class);  
    lastclass=classcol_infor.lastclass;
    curclass_pro_normal = classcol_infor.curclass_pro_normal;
    choose_classcol_areanum = classcol_infor.choose_classcol_areanum;
    record_lastclass{k}=lastclass;
    record_curclasscol{k}=choosepick_classcol;
    record_curclass_pro_normal{k}=curclass_pro_normal;
    record_curclass_areanum{k}=choose_classcol_areanum;
    
    % 1.3 pick basic unit��easy misclassification)
    classcol_adjacent=Relationship.adjcant_relationship(choosepick_classcol,choosepick_classcol); %��ǰ��𼯺ϵ����ڹ�ϵ��Ϣ����
    classcol_contain=Relationship.contain_relationship(choosepick_classcol,choosepick_classcol); %��ǰ��𼯺ϵİ�����ϵ��Ϣ����
    classcol_containwith=Relationship.containwith_relationship(choosepick_classcol,choosepick_classcol);
    classcol_georelationship=classcol_adjacent+classcol_contain+classcol_containwith;
 
    curclasswithotherclass_allgeorelation=zeros(3,length(choosepick_classcol));
    Curbasic_unit_easy=[];Withcontain_classinfor_all=[];choosebasic_index=[];
    if ismember(1,choosepair_easy)
        [stats_x,stats_y]=find(simi_matrix>=0.2);
    else
        [stats_x,stats_y]=find(simi_matrix>=0.5);
    end
    cur_misclass_pair_idx=[stats_x stats_y];
    delete_idx=easypair_index{choose_class,em_toclass};
    if isempty(cur_misclass_pair_idx) || length(stats_x)<=length(delete_idx)
        thres=0.5;
        while isempty(stats_x) || length(stats_x)<=length(delete_idx)
            thres=thres-0.05;
            [stats_x,stats_y]=find(simi_matrix>=thres);
        end
        cur_misclass_pair_idx=[stats_x stats_y];
    end
       
    %% choose simi basic unit
    stats_idx=zeros(length(stats_x),2);
    stats_areainfor=zeros(1,2);
    rand_infor=cell(2,2);
    for kk=1:2
       curfill_class=choosepair_easy(kk);
       curchooseclass_allarea = upbasicunit_infor(curfill_class).allcurclass_area;
       cur_chooseclass_area=curchooseclass_allarea(cur_misclass_pair_idx(:,kk));
       chooseclass_idx=choosepick_classcol==curfill_class;
       chooseclass_pro = curclass_pro_normal(chooseclass_idx);
       chooseclass_areanum= choose_classcol_areanum(chooseclass_idx);
       chooseclass_meanarea=round(s*s*chooseclass_pro/chooseclass_areanum);
       stats_areainfor(kk)=s*s*chooseclass_pro;
       stats_condi= cur_chooseclass_area>=chooseclass_meanarea-chooseclass_meanarea*0.3 &...
           cur_chooseclass_area<=s*s*chooseclass_pro+chooseclass_meanarea*0.3;
       stats_idx(:,kk)=stats_condi;
    end
    % ����basicunit����������Ҫ��
    exits_easypair_all=find(sum(stats_idx,2)==2);
    exits_easypair=setdiff(exits_easypair_all,delete_idx);
    %ֻ��һ��basicunit��������Ҫ��
    exits_easypaironly_all=find(sum(stats_idx,2)==1);
    exits_easypaironly=setdiff(exits_easypaironly_all,delete_idx);
    if ~isempty(exits_easypair)
        choose_pairidx=exits_easypair(randperm(length(exits_easypair),1));
        choose_pairindex=cur_misclass_pair_idx(choose_pairidx,:);
        easypair_index{choose_class,em_toclass}=[easypair_index{choose_class,em_toclass} choose_pairidx];
        for ke=1:length(choose_pairindex)
            curfill_class=choosepair_easy(ke);
            k1_idx=find(curfill_class==choosepick_classcol);
            %�жϵ�ǰֵcurfill_class���������λ����Ϣ                
            otherclass_indexcol = ~ismember(choosepick_classcol,curfill_class);%0��ʾ����
            curclasswithotherclass_allgeorelation(1,:) =classcol_adjacent(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));%��ȡ��ǰ��������λ�ø���
            curclasswithotherclass_allgeorelation(2,:) =classcol_contain(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));%��ȡ��ǰ��������λ�ø���
            curclasswithotherclass_allgeorelation(3,:) = classcol_containwith(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));
            curclass_withotherclass_georelation = curclasswithotherclass_allgeorelation(:,otherclass_indexcol==1);

            curclass_area=curclass_pro_normal(choosepick_classcol==curfill_class)*s*s;
            curclass_area=curclass_pro_normal(choosepick_classcol==curfill_class)*s*s;
            curclassbasic_allarea = upbasicunit_infor(curfill_class).allcurclass_area; %��ȡ���㵱ǰ����basic unit��area����
            curclass_alloriginalid = upbasicunit_infor(curfill_class).alloriginal_id;
            curclass_allBoundingBox = upbasicunit_infor(curfill_class).allcurclass_BoundingBox;
            curselected_id=choose_pairindex(ke);%�������������������basic unitչ�����������Ϣ 
            choose_basic_index{curfill_class}=[choose_basic_index{curfill_class} curselected_id];
            %��ȡ��ǰѡȡ��basic unit��Ϣ
            sampledpoints_id=curclass_alloriginalid(curselected_id);
            statscurclass_boundingbox=curclass_allBoundingBox(curselected_id,:);%[�� �� �� ��]
            rand_infor{ke,1}=sampledpoints_id;
            rand_infor{ke,2}=statscurclass_boundingbox;
            % ��ȡ����ȡ��basic unit��Ϣ
            [basic_image,basic_label,sampledp]=randcrop_image(Given_infor,sampledpoints_id,statscurclass_boundingbox,curfill_class,curclass_area);
            [curbasic_row,curbasic_col,~]=size(basic_image);
            % ��ȡ�������а�������Ӧ�ĸ�����Ϣ(�����·�ϵĸ�����Ϣ)
            % Withcontain_classinfor���Ǹ����а�����Щ�����Ϣ���Լ���С��Ϣ
            Curbasic_unitidx=size(Curbasic_unit_easy,1)+1;
            [Withcontain_classinfor,basic_label_update,curbasic_boxb]=acquire_conclassinfor(basic_label,curfill_class,contain_toclass,car_class,clutter);
            if ~isempty(Withcontain_classinfor)
                Withcontain_classinfor = arrayfun(@(x, y) setfield(x, 'containclassidx', y),...
                                         Withcontain_classinfor, repmat(Curbasic_unitidx,[size(Withcontain_classinfor,1) 1]));
            end
            Withcontain_classinfor_all=[Withcontain_classinfor_all;Withcontain_classinfor];
            Cursameclass_easy=cell(1,10);
            Cursameclass_easy{1,1}=basic_label_update;
            Cursameclass_easy{1,2}=basic_image;
            Cursameclass_easy{1,3}=curfill_class;
            Cursameclass_easy{1,4}=(sum(basic_label_update(:)~=0))/(curbasic_row*curbasic_col);
            Cursameclass_easy{1,5}=curbasic_row*curbasic_col;
            Cursameclass_easy{1,6}=choosepick_classcol(curclasswithotherclass_allgeorelation(1,:)~=0); %�洢���������������е�����λ�ù�ϵ
            Cursameclass_easy{1,7}=curclasswithotherclass_allgeorelation(:,curclasswithotherclass_allgeorelation(1,:)~=0); %�洢���������������е�����λ�ù�ϵ�ĸ��ʼ���
             binary_mask=basic_label_update~=0;
            props = regionprops(binary_mask, 'Centroid');
            Cursameclass_easy{1,8}=props.Centroid;
            if choose_class == curfill_class
                choosebasic_index=[choosebasic_index;curselected_id];
                Cursameclass_easy{1,9}=1;
            else
                Cursameclass_easy{1,9}=0;
            end 
            Cursameclass_easy{1,10}=0;
            Cursameclass_easy{1,11}=curbasic_boxb;%������boxboundings
            Curbasic_unit_easy=[Curbasic_unit_easy;Cursameclass_easy];
        end
    elseif ~isempty(exits_easypaironly) && isempty(exits_easypair)
        % �����������һ��choose_class
        if sum(stats_idx(exits_easypaironly,1))~=0
            cur_areainfor=stats_areainfor(2);
            firstcolumn_idx=exits_easypaironly(stats_idx(exits_easypaironly,1)==1);
            firstcolumn_toallidx=stats_y(firstcolumn_idx);
            curfill_class=choosepair_easy(2);
            curchooseclass_allarea = upbasicunit_infor(curfill_class).allcurclass_area;
            cur_chooseclass_area=curchooseclass_allarea(firstcolumn_toallidx);
            [minfc_value,minfc_idx]=min(abs(cur_chooseclass_area-cur_areainfor));
            minendf_idx=firstcolumn_idx(minfc_idx(1));
        end
        if sum(stats_idx(exits_easypaironly,2))~=0
            cur_areainfor=stats_areainfor(1);
            secondcolumn_idx=exits_easypaironly(stats_idx(exits_easypaironly,2)==1);
            secondcolumn_toallidx=stats_x(secondcolumn_idx);
            curfill_class=choosepair_easy(1);
            curchooseclass_allarea = upbasicunit_infor(curfill_class).allcurclass_area;
            cur_chooseclass_area=curchooseclass_allarea(secondcolumn_toallidx);
            [minsc_value,minsc_idx]=min(abs(cur_chooseclass_area-cur_areainfor));
            minends_idx=secondcolumn_idx(minsc_idx(1));
        end
        if sum(stats_idx(exits_easypaironly,1))~=0 && sum(stats_idx(exits_easypaironly,2))~=0
            if min(minfc_value,minsc_value)==minsc_value
                choose_pairidx=minends_idx;
            else
                choose_pairidx=minendf_idx;
            end
        elseif sum(stats_idx(exits_easypaironly,1))~=0 && sum(stats_idx(exits_easypaironly,2))==0
            choose_pairidx=minendf_idx;
        else
            choose_pairidx=minends_idx;
        end
        choose_pairindex=cur_misclass_pair_idx(choose_pairidx,:);
        easypair_index{choose_class,em_toclass}=[easypair_index{choose_class,em_toclass} choose_pairidx];
      
        for ke=1:length(choose_pairindex)
            curfill_class=choosepair_easy(ke);
            k1_idx=find(curfill_class==choosepick_classcol);
            %�жϵ�ǰֵcurfill_class���������λ����Ϣ                
            otherclass_indexcol = ~ismember(choosepick_classcol,curfill_class);%0��ʾ����
            curclasswithotherclass_allgeorelation(1,:) =classcol_adjacent(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));%��ȡ��ǰ��������λ�ø���
            curclasswithotherclass_allgeorelation(2,:) =classcol_contain(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));%��ȡ��ǰ��������λ�ø���
            curclasswithotherclass_allgeorelation(3,:) = classcol_containwith(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));
            curclass_withotherclass_georelation = curclasswithotherclass_allgeorelation(:,otherclass_indexcol==1);

            curclass_area=curclass_pro_normal(choosepick_classcol==curfill_class)*s*s;
            curclassbasic_allarea = upbasicunit_infor(curfill_class).allcurclass_area; %��ȡ���㵱ǰ����basic unit��area����
            curclass_alloriginalid = upbasicunit_infor(curfill_class).alloriginal_id;
            curclass_allBoundingBox = upbasicunit_infor(curfill_class).allcurclass_BoundingBox;
            curselected_id=choose_pairindex(ke);%�������������������basic unitչ�����������Ϣ 
            choose_basic_index{curfill_class}=[choose_basic_index{curfill_class} curselected_id];
            %��ȡ��ǰѡȡ��basic unit��Ϣ
            sampledpoints_id=curclass_alloriginalid(curselected_id);
            statscurclass_boundingbox=curclass_allBoundingBox(curselected_id,:);%[�� �� �� ��]
            rand_infor{ke,1}=sampledpoints_id;
            rand_infor{ke,2}=statscurclass_boundingbox;
            % ��ȡ����ȡ��basic unit��Ϣ
            [basic_image,basic_label,sampledp]=randcrop_image(Given_infor,sampledpoints_id,statscurclass_boundingbox,curfill_class,curclass_area);
            [curbasic_row,curbasic_col,~]=size(basic_image);
            % ��ȡ�������а�������Ӧ�ĸ�����Ϣ(�����·�ϵĸ�����Ϣ)
            % Withcontain_classinfor���Ǹ����а�����Щ�����Ϣ���Լ���С��Ϣ
            Curbasic_unitidx=size(Curbasic_unit_easy,1)+1;
            [Withcontain_classinfor,basic_label_update,curbasic_boxb]=acquire_conclassinfor(basic_label,curfill_class,contain_toclass,car_class,clutter);
            if ~isempty(Withcontain_classinfor)
                Withcontain_classinfor = arrayfun(@(x, y) setfield(x, 'containclassidx', y),...
                                         Withcontain_classinfor, repmat(Curbasic_unitidx,[size(Withcontain_classinfor,1) 1]));
            end
            Withcontain_classinfor_all=[Withcontain_classinfor_all;Withcontain_classinfor];
            Cursameclass_easy=cell(1,10);
            Cursameclass_easy{1,1}=basic_label_update;
            Cursameclass_easy{1,2}=basic_image;
            Cursameclass_easy{1,3}=curfill_class;
            Cursameclass_easy{1,4}=(sum(basic_label_update(:)~=0))/(curbasic_row*curbasic_col);
            Cursameclass_easy{1,5}=curbasic_row*curbasic_col;
            Cursameclass_easy{1,6}=choosepick_classcol(curclasswithotherclass_allgeorelation(1,:)~=0); %�洢���������������е�����λ�ù�ϵ
            Cursameclass_easy{1,7}=curclasswithotherclass_allgeorelation(:,curclasswithotherclass_allgeorelation(1,:)~=0); %�洢���������������е�����λ�ù�ϵ�ĸ��ʼ���
            binary_mask=basic_label_update~=0;
            props = regionprops(binary_mask, 'Centroid');
            Cursameclass_easy{1,8}=props.Centroid;
            if choose_class == curfill_class
                choosebasic_index=[choosebasic_index;curselected_id];
                Cursameclass_easy{1,9}=1;
            else
                Cursameclass_easy{1,9}=0;
            end 
            Cursameclass_easy{1,10}=0;
            Cursameclass_easy{1,11}=curbasic_boxb;%������boxboundings
            Curbasic_unit_easy=[Curbasic_unit_easy;Cursameclass_easy];
        end
    else
        %���������ȡһ��
        remainpair_idx=setdiff([1:length(cur_misclass_pair_idx)],delete_idx);
        remainpair_infor=cur_misclass_pair_idx(remainpair_idx,:),
        choose_pairidx=randperm(length(remainpair_infor),1);
        choose_pairindex=cur_misclass_pair_idx(choose_pairidx,:);
        easypair_index{choose_class,em_toclass}=[easypair_index{choose_class,em_toclass} choose_pairidx];
        for ke=1:length(choose_pairindex)
            curfill_class=choosepair_easy(ke);
            k1_idx=find(curfill_class==choosepick_classcol);
            %�жϵ�ǰֵcurfill_class���������λ����Ϣ                
            otherclass_indexcol = ~ismember(choosepick_classcol,curfill_class);%0��ʾ����
            curclasswithotherclass_allgeorelation(1,:) =classcol_adjacent(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));%��ȡ��ǰ��������λ�ø���
            curclasswithotherclass_allgeorelation(2,:) =classcol_contain(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));%��ȡ��ǰ��������λ�ø���
            curclasswithotherclass_allgeorelation(3,:) = classcol_containwith(k1_idx,:)./sum(classcol_georelationship(k1_idx,:));
            curclass_withotherclass_georelation = curclasswithotherclass_allgeorelation(:,otherclass_indexcol==1);

            curclass_area=curclass_pro_normal(choosepick_classcol==curfill_class)*s*s;
            curclassbasic_allarea = upbasicunit_infor(curfill_class).allcurclass_area; %��ȡ���㵱ǰ����basic unit��area����
            curclass_alloriginalid = upbasicunit_infor(curfill_class).alloriginal_id;
            curclass_allBoundingBox = upbasicunit_infor(curfill_class).allcurclass_BoundingBox;
            curselected_id=choose_pairindex(ke);%�������������������basic unitչ�����������Ϣ 
            choose_basic_index{curfill_class}=[choose_basic_index{curfill_class} curselected_id];
            %��ȡ��ǰѡȡ��basic unit��Ϣ
            sampledpoints_id=curclass_alloriginalid(curselected_id);
            statscurclass_boundingbox=curclass_allBoundingBox(curselected_id,:);%[�� �� �� ��]
            rand_infor{ke,1}=sampledpoints_id;
            rand_infor{ke,2}=statscurclass_boundingbox;

            % ��ȡ����ȡ��basic unit��Ϣ
            [basic_image,basic_label,sampledp]=randcrop_image(Given_infor,sampledpoints_id,statscurclass_boundingbox,curfill_class,curclass_area);
            [curbasic_row,curbasic_col,~]=size(basic_image);
            % ��ȡ�������а�������Ӧ�ĸ�����Ϣ(�����·�ϵĸ�����Ϣ)
            % Withcontain_classinfor���Ǹ����а�����Щ�����Ϣ���Լ���С��Ϣ
            Curbasic_unitidx=size(Curbasic_unit_easy,1)+1;
            [Withcontain_classinfor,basic_label_update,curbasic_boxb]=acquire_conclassinfor(basic_label,curfill_class,contain_toclass,car_class,clutter);
            if ~isempty(Withcontain_classinfor)
                Withcontain_classinfor = arrayfun(@(x, y) setfield(x, 'containclassidx', y),...
                                         Withcontain_classinfor, repmat(Curbasic_unitidx,[size(Withcontain_classinfor,1) 1]));
            end
            Withcontain_classinfor_all=[Withcontain_classinfor_all;Withcontain_classinfor];
            Cursameclass_easy=cell(1,10);
            Cursameclass_easy{1,1}=basic_label_update;
            Cursameclass_easy{1,2}=basic_image;
            Cursameclass_easy{1,3}=curfill_class;
            Cursameclass_easy{1,4}=(sum(basic_label_update(:)~=0))/(curbasic_row*curbasic_col);
            Cursameclass_easy{1,5}=curbasic_row*curbasic_col;
            Cursameclass_easy{1,6}=choosepick_classcol(curclasswithotherclass_allgeorelation(1,:)~=0); %�洢���������������е�����λ�ù�ϵ
            Cursameclass_easy{1,7}=curclasswithotherclass_allgeorelation(:,curclasswithotherclass_allgeorelation(1,:)~=0); %�洢���������������е�����λ�ù�ϵ�ĸ��ʼ���
            binary_mask=basic_label_update~=0;
            props = regionprops(binary_mask, 'Centroid');
            Cursameclass_easy{1,8}=props.Centroid;
            if choose_class == curfill_class
                choosebasic_index=[choosebasic_index;curselected_id];
                Cursameclass_easy{1,9}=1;
            else
                Cursameclass_easy{1,9}=0;
            end 
            Cursameclass_easy{1,10}=0;
            Cursameclass_easy{1,11}=curbasic_boxb;%������boxboundings
            Curbasic_unit_easy=[Curbasic_unit_easy;Cursameclass_easy];
        end
    end 
    Rand_infor{k}=rand_infor;
    
    %keyboard
    for k_rand=1:2
        sampledpoints_id=rand_infor{k_rand,1};
        statscurclass_boundingbox=rand_infor{k_rand,2};
        statscurclass_boundingbox(3)=511;
        statscurclass_boundingbox(4)=511;
        [basic_image,basic_label]=randcrop_image(Given_infor,sampledpoints_id,statscurclass_boundingbox,curfill_class,curclass_area);
        if size(basic_label,1)~=512
            statscurclass_boundingbox(2)=6800-512+1;
            [basic_image,basic_label]=randcrop_image(Given_infor,sampledpoints_id,statscurclass_boundingbox,curfill_class,curclass_area);
        elseif  size(basic_label,2)~=512
            statscurclass_boundingbox(1)=7200-512+1;
            [basic_image,basic_label]=randcrop_image(Given_infor,sampledpoints_id,statscurclass_boundingbox,curfill_class,curclass_area);
        end
        strr=strcat('rand_',num2str(k),num2str(k_rand),'.png');
        imwrite(uint8(basic_image),['F:\lly\compare_data_instance\gid15\rand\image\' strr])
        imwrite(uint8(basic_label),['F:\lly\compare_data_instance\gid15\rand\label\' strr])
    end

    %% ���·���ֶԵ�basic unit����Ϣ
    for kc=1:size(Curbasic_unit_easy,1)
        curchooseclass=Curbasic_unit_easy{kc,3};
        curchooseclass_idx=find(choosepick_classcol==curchooseclass);
        curchooseclass_area=Curbasic_unit_easy{kc,4}*Curbasic_unit_easy{kc,5};
        choose_classcol_areanum(curchooseclass_idx)=choose_classcol_areanum(curchooseclass_idx)-1;
        curclass_pro_normal(curchooseclass_idx)=max(0,curclass_pro_normal(curchooseclass_idx)-curchooseclass_area/(s*s));
    end
    
    %% 1.4����ȡ��ǰ���֮��Ĺ�ϵ����
    classcol_adjacent=Relationship.adjcant_relationship(choosepick_classcol,choosepick_classcol); %��ǰ��𼯺ϵ����ڹ�ϵ��Ϣ����
    classcol_contain=Relationship.contain_relationship(choosepick_classcol,choosepick_classcol); %��ǰ��𼯺ϵİ�����ϵ��Ϣ����
    classcol_containwith=Relationship.containwith_relationship(choosepick_classcol,choosepick_classcol);
    classcol_georelationship=classcol_adjacent+classcol_contain+classcol_containwith;
    % 1.5��pick basic unit 
    curclasswithotherclass_allgeorelation=zeros(3,length(choosepick_classcol));%��һ�б�ʾ���ڹ�ϵ���ڶ��б�ʾ������ϵ
    Curbasic_unit_all=[]; choosebasic_index=[]; chooseclass_his=[];  Withcontain_classinfor_all=[];
    for k1=1:length(choosepick_classcol)
        curfill_class=choosepick_classcol(k1);%��ȡ��ǰ�����ֵ
        curclass_area=curclass_pro_normal(k1)*(s^2);
        curfill_areanum=choose_classcol_areanum(k1);
        if curclass_area>0 && curfill_areanum<=0 
            curfill_areanum=1;
        end
        if curclass_area==0
            continue;
        end
        %�жϵ�ǰֵcurfill_class���������λ����Ϣ                
        otherclass_indexcol = ~ismember(choosepick_classcol,curfill_class);%0��ʾ����
        curclasswithotherclass_allgeorelation(1,:) =classcol_adjacent(k1,:)./sum(classcol_georelationship(k1,:));%��ȡ��ǰ��������λ�ø���
        curclasswithotherclass_allgeorelation(2,:) =classcol_contain(k1,:)./sum(classcol_georelationship(k1,:));%��ȡ��ǰ��������λ�ø���
        curclasswithotherclass_allgeorelation(3,:) = classcol_containwith(k1,:)./sum(classcol_georelationship(k1,:));
        curclass_withotherclass_georelation = curclasswithotherclass_allgeorelation(:,otherclass_indexcol==1);
        % ��ǰ����ȡ�����Ϣ����
        curclass_infor.curfill_class=curfill_class;
        curclass_infor.curclass_area=curclass_area;
        curclass_infor.curfill_areanum=curfill_areanum;
        curclass_infor.choose_classcol=choosepick_classcol;
        curclass_infor.choose_class=choose_class;
        curclass_infor.em_toclass=em_toclass;
        curclass_infor.curclasswithotherclass_allgeorelation=curclasswithotherclass_allgeorelation;
        
        if ismember(curfill_class,irr_class) 
            % ѡȡ���ʵ�basic unitʱarea�Լ������Ϣѡȡ
            curclassbasic_allarea = basicunit_infor(curfill_class).allcurclass_area; %��ȡ���㵱ǰ����basic unit��area����
            curclass_alloriginalid = basicunit_infor(curfill_class).alloriginal_id;
            curclass_allBoundingBox = basicunit_infor(curfill_class).allcurclass_BoundingBox;
            curclass_allcentroid = basicunit_infor(curfill_class).allcurclass_centroid;
            
            statsneed_classindex_ori=1:length(curclassbasic_allarea);
            delete_chooseindex=ismember(statsneed_classindex_ori, [choose_basic_index{curfill_class}]); 
            statsneed_classindex_ori(delete_chooseindex)=[];  %��ȡ��ǰ�������ϣ����Żس���
            %2. �ֲ��ȡ��ǰ����basic unit��Ϣ
            statsnees_classidx = curclass_alloriginalid(statsneed_classindex_ori);
            choose_id_already=unique([choose_basicidx{curfill_class}]);
            if length(choose_id_already)==length(unique(curclass_alloriginalid))
                choose_basicidx{curfill_class}=[];
            end
            statsneed_classindex=statsneed_classindex_ori(~ismember(statsnees_classidx,[choose_basicidx{curfill_class}]));

            %��basic unitȫ����ȡ����������³�ȡ
            if isempty(statsneed_classindex)
                statsneed_classindex=statsneed_classindex_ori;
            end
            if isempty(statsneed_classindex)
                choose_basic_index{curfill_class}=[];
                statsneed_classindex=1:length(curclassbasic_allarea);
            end
            
            statsneed_classtoarea=curclassbasic_allarea(statsneed_classindex);
            [~,nearest_idx]=min(abs(statsneed_classtoarea-curclass_area));
            curselected_id=statsneed_classindex(nearest_idx);
            % ��ȡ��ȡ��������Ϣ
            sampledpoints_id=curclass_alloriginalid(curselected_id);%�������������������basic unitչ�����������Ϣ 
            if curselected_id>length(curclassbasic_allarea)
                keyboard
            end
            choose_basic_index{curfill_class}=[choose_basic_index{curfill_class} curselected_id];
            %��ȡ��ǰѡȡ��basic unit��Ϣ
            statscurclass_boundingbox=curclass_allBoundingBox(curselected_id,:);%[�� �� �� ��]
            statscurclass_centroid=curclass_allcentroid(curselected_id,:);
            % ��ȡ����ȡ��basic unit��Ϣ
            [basic_image,basic_label,sampledp]=randcrop_image(Given_infor,sampledpoints_id,statscurclass_boundingbox,curfill_class,curclass_area);
            [curbasic_row,curbasic_col,~]=size(basic_image);
            % ��ȡ�������а�������Ӧ�ĸ�����Ϣ(�����·�ϵĸ�����Ϣ)
            % Withcontain_classinfor���Ǹ����а�����Щ�����Ϣ���Լ���С��Ϣ
            Curbasic_unitidx=size(Curbasic_unit_all,1)+1;
            [Withcontain_classinfor,basic_label_update,curbasic_boxb]=acquire_conclassinfor(basic_label,curfill_class,contain_toclass,car_class,clutter);
            if ~isempty(Withcontain_classinfor)
                Withcontain_classinfor = arrayfun(@(x, y) setfield(x, 'containclassidx', y),...
                                         Withcontain_classinfor, repmat(Curbasic_unitidx,[size(Withcontain_classinfor,1) 1]));
            end
            Withcontain_classinfor_all=[Withcontain_classinfor_all;Withcontain_classinfor];
            Cursameclass=cell(1,10);
            Cursameclass{1,1}=basic_label_update;
            Cursameclass{1,2}=basic_image;
            Cursameclass{1,3}=curfill_class;
            Cursameclass{1,4}=(sum(basic_label_update(:)~=0))/(curbasic_row*curbasic_col);
            Cursameclass{1,5}=curbasic_row*curbasic_col;
            Cursameclass{1,6}=choosepick_classcol(curclasswithotherclass_allgeorelation(1,:)~=0); %�洢���������������е�����λ�ù�ϵ
            Cursameclass{1,7}=curclasswithotherclass_allgeorelation(:,curclasswithotherclass_allgeorelation(1,:)~=0); %�洢���������������е�����λ�ù�ϵ�ĸ��ʼ���
            Cursameclass{1,8}=statscurclass_centroid;
            if choose_class == curfill_class
                choosebasic_index=[choosebasic_index;curselected_id];
                Cursameclass{1,9}=1;
            else
                Cursameclass{1,9}=0;
            end 
            Cursameclass{1,10}=0;
            Cursameclass{1,11}=curbasic_boxb;%������boxboundings
            Curbasic_unit_all=[Curbasic_unit_all;Cursameclass];
        else
            tic
            [Cursameclass,Curclass_areavalue,choose_basic_index,choose_basicidx]=pick_basic_unit(Given_infor,...
                curclass_infor,choose_basic_index,choose_basicidx,Withcontain_classinfor_all);
            elapsed_time2=toc;
            Curbasic_unit_all=[Curbasic_unit_all;Cursameclass];
            disp(['��ȡbasic unit��', num2str(elapsed_time2)])
        end
    end
    Curbasic_unit_all=[Curbasic_unit_all;Curbasic_unit_easy];
    %����Curbasic_unit
    Curbasic_unit=update_Curbasicunit(Curbasic_unit_all);
    
    %1.6 combinating basic units into training image
    % 1.6.1 ��Χ��ϵ�����
    Curbasic_unitupdate=containorwithfill(Curbasic_unit,s);
    % 1.6.2 ����ϡ����Լ���С����basic unit�����
    basicunit_fillsort=sort_basicunit(Curbasic_unitupdate,lastclass,s);
    % 1.6.3 ���
    label_micro=zeros(s,s);
    image_micro=zeros(s,s,3);
    basic_cen=cell(length(basicunit_fillsort),2);
    for k2=1:length(basicunit_fillsort)
        willfill_basicid=basicunit_fillsort(k2);
        fill_factor=Curbasic_unitupdate{willfill_basicid,8};
        willfill_image=Curbasic_unitupdate{willfill_basicid,2}; % ����basic unit image
        willfill_label=Curbasic_unitupdate{willfill_basicid,1}; % ����basic unit label
        %%
%         figure,imshow(uint8(willfill_image))
        if isempty(willfill_image)
            continue;
        end
        [curbasic_row,curbasic_col]=size(willfill_label);
        willfill_class=Curbasic_unitupdate{willfill_basicid,3};%���basic unit�����
        willfill_adclass=Curbasic_unitupdate{willfill_basicid,6};%����basicunit���ڽ����
        if sum(label_micro(:))==0%��һ�������label_micro��ȫδ���
            row_inipoints=1;
            col_inipoints=1;
            angle_rand=rand();
            angle_value=randi(360,1);
            if angle_rand>0.5
                willfill_image=imrotate(willfill_image,angle_value); % ����basic unit image
                willfill_label=imrotate(willfill_label,angle_value);
                [curbasic_row,curbasic_col]=size(willfill_label);
                if curbasic_row>s || curbasic_col>s
                    willfill_image=Curbasic_unitupdate{willfill_basicid,2}; % ����basic unit image
                    willfill_label=Curbasic_unitupdate{willfill_basicid,1}; % ����basic unit label
                    [curbasic_row,curbasic_col]=size(willfill_label);
                end
            end
            %����������䵽Ӱ����
            willfill_image_new=uint8(willfill_image).*uint8(repmat(willfill_label~=0,[1, 1, 3]));
            label_micro(row_inipoints:row_inipoints+curbasic_row-1,col_inipoints:col_inipoints+curbasic_col-1)=willfill_label;
            image_micro(row_inipoints:row_inipoints+curbasic_row-1,col_inipoints:col_inipoints+curbasic_col-1,:)=willfill_image_new;
            basic_cen{1,1}=willfill_class;
            basic_cen{1,2}=Curbasic_unit{willfill_basicid,8};
        else
            label_micro_middle=zeros(s,s);image_micro_middle=zeros(s,s,3);
            [fill_pixelpoints,theta]=compare_choosepoints(label_micro,Curbasic_unitupdate, willfill_basicid,minbox_diffclass,basic_cen);
            basic_cen{1}=willfill_class;
            basic_cen{2}=Curbasic_unitupdate{willfill_basicid,8}+fill_pixelpoints-1;
            willfill_image=imrotate(willfill_image,theta);
            willfill_label=imrotate(willfill_label,theta);
            willfill_image(all(willfill_label==0,2),:,:)=[];
            willfill_image(:,all(willfill_label==0,1),:)=[];
            
            willfill_label(all(willfill_label==0,2),:)=[];
            willfill_label(:,all(willfill_label==0,1))=[];
            
            willfill_image_new=uint8(willfill_image).*uint8(repmat(willfill_label~=0,[1, 1, 3]));
            if size(willfill_image_new,1)>s
                willfill_image_new=imcrop(willfill_image_new,[1 1 size(willfill_image_new,1) s-1]);
                willfill_label=imcrop(willfill_label,[1 1 size(willfill_image_new,1) s-1]);
            elseif size(willfill_image_new,2)>s
                willfill_image_new=imcrop(willfill_image_new,[1 1 s-1 size(willfill_image_new,2)]);
                willfill_label=imcrop(willfill_label,[1 1 s-1 size(willfill_image_new,2)]);
            end
            [curbasic_row,curbasic_col]=size(willfill_label);

%             % ȥ��ϸ��
%             mask_label_grid=mask_grid(label_micro,32);
%             mask_label=matrixexpand(mask_label_grid,32);
%             mask_label_micro=mask_label~=0 |  label_micro~=0;
            
            image_micro_middle(fill_pixelpoints(1):fill_pixelpoints(1)+curbasic_row-1,fill_pixelpoints(2):fill_pixelpoints(2)+curbasic_col-1,:)=willfill_image_new;
            label_micro_middle(fill_pixelpoints(1):fill_pixelpoints(1)+curbasic_row-1,fill_pixelpoints(2):fill_pixelpoints(2)+curbasic_col-1)=willfill_label;
            imagemicro_1=image_micro(:,:,1);imagemicro_2=image_micro(:,:,2);imagemicro_3=image_micro(:,:,3);
            choose_image1=image_micro_middle(:,:,1);  choose_image2=image_micro_middle(:,:,2);choose_image3=image_micro_middle(:,:,3);             
            imagemicro_1(label_micro==0)=choose_image1(label_micro==0);
            imagemicro_2(label_micro==0)=choose_image2(label_micro==0);
            imagemicro_3(label_micro==0)=choose_image3(label_micro==0);
            image_micro(:,:,1)=imagemicro_1;image_micro(:,:,2)=imagemicro_2;image_micro(:,:,3)=imagemicro_3;
            label_micro(label_micro==0)=label_micro_middle(label_micro==0);
        end
    end
%     image_micro1=image_micro=image_micro1;
%     label_micro1=label_micro=label_micro1;
    figure,imshow(uint8(image_micro))
    last_fillcalss=lastclass;

    flag2=1;flag3=1;
    noreapeat_idx=[];
    %% ������䣺
    %��ȡ��ǰ����basic unit����Ϣ��
    willbasic_windows=[s,s];
    [basic_image,basic_label,fill_rotate,choose_basic_index]=fill_lastclass_maxone(label_micro,imagepath,labelpath,...
    upbasicunit_infor,choose_basic_index,last_fillcalss,noreapeat_idx);
    label_micro=imrotate(label_micro,fill_rotate);
    image_micro=imrotate(image_micro,fill_rotate);
    [curbasicrow,curbasiccol]=size(basic_label);
    willfillimage_new=basic_image.*uint8(repmat(basic_label~=0,[1, 1, 3]));
    image_micro_middle(1:1+curbasicrow-1,1:1+curbasiccol-1,:)=willfillimage_new;
    label_micro_middle(1:1+curbasicrow-1,1:1+curbasiccol-1)=basic_label;
    imagemicro_1=image_micro(:,:,1);imagemicro_2=image_micro(:,:,2);imagemicro_3=image_micro(:,:,3);
    choose_image1=image_micro_middle(:,:,1);  choose_image2=image_micro_middle(:,:,2);choose_image3=image_micro_middle(:,:,3);             
    imagemicro_1(label_micro==0)=choose_image1(label_micro==0);
    imagemicro_2(label_micro==0)=choose_image2(label_micro==0);
    imagemicro_3(label_micro==0)=choose_image3(label_micro==0);
    image_micro(:,:,1)=imagemicro_1;image_micro(:,:,2)=imagemicro_2;image_micro(:,:,3)=imagemicro_3;
    label_micro(label_micro==0)=label_micro_middle(label_micro==0);
    % ��������С��0.4����������       
    while sum(label_micro(:)==0)>s*s*0.4
        %��ȡ��ǰȱʧ������Ϣ
        % Ѱ��һ�����ķ��ϵ�ǰ�����basic unit������� 
        pro=0.008;
        label_binary=label_micro==0;
        State=regionprops(label_binary);
        toosmall_dilateindex=find([State.Area]<s*s*pro);
        State(toosmall_dilateindex)=[];
        %����State��ȡbasic unit�������
        for k7=1:length(State)
            absent_size=ceil(State(k7).BoundingBox);%��ǰȱʧ��С��Ϣ
            %��ȡ��ǰ����basic unit����Ϣ��
            curclassbasic_allarea = upbasicunit_infor(last_fillcalss).allcurclass_area; %��ȡ���㵱ǰ����basic unit��area����
            curclass_originalid = upbasicunit_infor(last_fillcalss).alloriginal_id;
            curclass_BoundingBox = upbasicunit_infor(last_fillcalss).allcurclass_BoundingBox;
            statsneed_classindex=1:length(curclassbasic_allarea); %���Żس���
            delete_chooseindex=unique([choose_basic_index{last_fillcalss}]);           
            if length(delete_chooseindex)==length(statsneed_classindex)%��־�ŵ�ǰ������е�basic unit���ѳ�ȡ���������¶��γ�ȡ
%                 keyboard;
                choose_basic_index{last_fillcalss}=[];
                delete_chooseindex=unique([choose_basic_index{last_fillcalss}]);
            end
            statsneed_classindex(delete_chooseindex)=[];%�޳���ȡ����basic unit��������Ϣ 
            Allsubregion_indices= find((curclass_BoundingBox(statsneed_classindex,3)>= absent_size(3) & curclass_BoundingBox(statsneed_classindex,4)>= absent_size(4))==1);%��ȡstatsneed_classindex��������������������ֵ
            if isempty(Allsubregion_indices)
                Allsubregion_indices= find((curclass_BoundingBox(:,3)>= absent_size(3) & curclass_BoundingBox(:,4)>= absent_size(4))==1);
                Allsubregion_indices_nprempat=setdiff(Allsubregion_indices,noreapeat_idx);
                thres=1;
                while isempty(Allsubregion_indices_nprempat)
                    thres=thres-0.01;
                    Allsubregion_indices= find((curclass_BoundingBox(:,3)>= absent_size(3)*thres & curclass_BoundingBox(:,4)>= absent_size(4)*thres)==1);
                    Allsubregion_indices_nprempat=setdiff(Allsubregion_indices,noreapeat_idx);
                end
                selectarea_allinfor=curclassbasic_allarea(Allsubregion_indices_nprempat);
                Allsubregion_indices=Allsubregion_indices_nprempat(find(selectarea_allinfor>absent_size(3)*absent_size(4)*0.7));
                if isempty(Allsubregion_indices)
                    Allsubregion_indices=Allsubregion_indices_nprempat(randperm(length(Allsubregion_indices_nprempat),1));
                end
            end
            chooseimage_index=Allsubregion_indices(randperm(length(Allsubregion_indices),1));
            noreapeat_idx=[noreapeat_idx,chooseimage_index];
            choose_basic_index{last_fillcalss}=[choose_basic_index{last_fillcalss} chooseimage_index];%��¼���г�ȡ����basic unit������Ϣ
            original_id=curclass_originalid(chooseimage_index);
            basic_image=imread([imagepath labelDir(original_id).name]);
            raw_label=imread([labelpath labelDir(original_id).name]);
            %��ȡ��Ӧ��basic unit��Ϣ(512*512��С)
            basic_boundingbox=curclass_BoundingBox(chooseimage_index,:)-[0 0 1 1];
            basic_image=imcrop(basic_image,basic_boundingbox);%imcrop��ʾ[������ ������ ��ֵ ��ֵ]��
            basic_label=imcrop(raw_label,basic_boundingbox);
            %�������ڵ�ǰֵ�����ȫ��ע��Ϊ0
            basic_label(basic_label~=last_fillcalss)=0;
            [curbasic_row,curbasic_col]=size(basic_label);
            if curbasic_row>s || curbasic_col>s
                crop_row=min(curbasic_row,s);
                crop_col=min(curbasic_col,s);
                [fill_cpixelpoints]=pick_suitarea(basic_label,[crop_row crop_col]);
                basic_label_c=imcrop(basic_label,[fill_cpixelpoints(2) fill_cpixelpoints(1)...
                               crop_col-1 crop_row-1]);
                basic_image_c=imcrop(basic_image,[fill_cpixelpoints(2) fill_cpixelpoints(1)...
                               crop_col-1 crop_row-1]);
                basic_image=basic_image_c;
                basic_label=basic_label_c;
                [curbasic_row,curbasic_col]=size(basic_label);
            end  
            basiclabel_classpnly3=repmat(uint8(basic_label~=0), [1, 1, 3]);
            basic_image=basic_image.*basiclabel_classpnly3;
            [fill_pixelpoints]=pick_submatrix_max(label_micro,basic_label);
            image_micro_middle=zeros(s,s,3);label_micro_middle=zeros(s,s);
            image_micro_middle(fill_pixelpoints(1):fill_pixelpoints(1)+curbasic_row-1,fill_pixelpoints(2):fill_pixelpoints(2)+curbasic_col-1,:)=basic_image;
            label_micro_middle(fill_pixelpoints(1):fill_pixelpoints(1)+curbasic_row-1,fill_pixelpoints(2):fill_pixelpoints(2)+curbasic_col-1)=basic_label;
            imagemicro_1=image_micro(:,:,1);imagemicro_2=image_micro(:,:,2);imagemicro_3=image_micro(:,:,3);
            choose_image1=image_micro_middle(:,:,1);  choose_image2=image_micro_middle(:,:,2);choose_image3=image_micro_middle(:,:,3);             
            imagemicro_1(label_micro==0)=choose_image1(label_micro==0);
            imagemicro_2(label_micro==0)=choose_image2(label_micro==0);
            imagemicro_3(label_micro==0)=choose_image3(label_micro==0);
            image_micro(:,:,1)=imagemicro_1;image_micro(:,:,2)=imagemicro_2;image_micro(:,:,3)=imagemicro_3;
            label_micro(label_micro==0)=label_micro_middle(label_micro==0); 
            flag3=flag3+1;
         end
    end
    %��¼ÿ�����ɵ�Ӱ������������ظ���
    image_curclass=setdiff(unique(label_micro),0);
    for kgray=1:length(image_curclass)
        curclassinfor=image_curclass(kgray);
        gray_pixelsnum(curclassinfor)=gray_pixelsnum(curclassinfor)+sum(label_micro(:)==curclassinfor);
    end   
    strr=strcat('image_',num2str(k),'.png');
    save([save_image_path 'record_oneclass.mat'],'record_oneclass')
    save([save_image_path 'record_curclasscol.mat'],'record_curclasscol')
    imwrite(uint8(image_micro),([save_image_path strr]))
    imwrite(uint8(label_micro),([save_micro_path strr]))
    save([save_image_path 'choose_basic_index.mat'],'choose_basic_index')
    save([save_image_path 'record_class_infor.mat'],'record_class_infor')
    save([save_image_path 'gray_pixelsnum.mat'],'gray_pixelsnum')
    save([save_image_path 'record_curclasscol.mat'],'record_curclasscol')
    save([save_image_path 'record_oneclass.mat'],'record_oneclass')
    save([save_image_path 'record_curclass_pro_normal.mat'],'record_curclass_pro_normal')
    save([save_image_path 'record_curclass_id.mat'],'record_curclass_id')
    save([save_image_path 'record_curclass_areanum.mat'],'record_curclass_areanum')
    save([save_image_path 'withoeasy_classinfor.mat'],'withoeasy_classinfor')
    save([save_image_path 'easyclass_infor.mat'],'easyclass_infor')
    save([save_image_path 'record_lastclass.mat'],'record_lastclass')
    save([save_image_path 'choose_basicidx.mat'],'choose_basicidx')
    save([save_image_path 'easypair_index.mat'],'easypair_index')
end


%% step2��������𼯺ϴ��Ӽ����г�ȡ����Ҫ�������
savepathimage='D:\A_userfile\lly\data\lovada\code\uvaid_code\generate\imageend\';
savepathlabel='D:\A_userfile\lly\data\lovada\code\uvaid_code\generate\labelend\';
image_idx=find(cell2mat(cellfun(@(x) isempty(x),withoeasy_classinfor, 'UniformOutput', false)));
imagedirr=dir([save_image_path '*.png']);
parfor kkk=1:length(image_idx)
    k_choose=image_idx(kkk);
    strr1=strcat('image_',num2str(k_choose),'.png');
    image=imread([save_image_path strr1]);
    label=imread([save_micro_path strr1]);
    imwrite(uint8(image),[savepathimage strr1])
    imwrite(uint8(label),[savepathlabel strr1])
end

%1����ȡԭʼ�����д�СΪs*s��Ӱ�񼯺��е�������������unique_arrays�Լ����Ӧ��λ����Ϣ
imagepath='D:\A_userfile\lly\data\lovada\Train\all_image\';
labelpath='D:\A_userfile\lly\data\lovada\Train\all_label\';
save_image_path='D:\A_userfile\lly\data\lovada\code\uvaid_code\generate\randimage\';
save_micro_path='D:\A_userfile\lly\data\lovada\code\uvaid_code\generate\randlabel\';
imagedir=dir([labelpath '*.png']);
croppedData =cropimage_bystep(labelpath,[s s]);%��ԭʼ���ݰ���s*s�Ĵ��ڴ�С���вü�����ȡ��𼯺�
classinfor_extra = {croppedData(:).classes};
tuple_strings = cellfun(@(x) mat2str(x), classinfor_extra, 'UniformOutput', false);
[allclassinfor_arrays, ~, idx] = unique(tuple_strings);
position_info = arrayfun(@(v) find(idx == v), 1:length(allclassinfor_arrays), 'UniformOutput', false);

%2����ȡ��ǰѵ�����ݼ���Ӱ�����𼯺�training data(td)
%withoeasy_classinfor_id=withoeasy_classinfor(1:1000);
td_statsrand_allclasss=cellfun(@(x) mat2str(x),withoeasy_classinfor, 'UniformOutput', false);
[allclassinfor_td, ~, td_idx] = unique(td_statsrand_allclasss);
%�������ֶԵ��ܸ���
all_withoeasy=0;
diffennrt_classcolnum=zeros(length(allclassinfor_td)-1,1);
for k5=1:length(allclassinfor_td)-1
%    curclass_td=str2num(allclassinfor_td{k4});
    samplenumbers_td = sum(td_idx==k5);
    diffennrt_classcolnum(k5)=samplenumbers_td;
    all_withoeasy=all_withoeasy+samplenumbers_td;
end
if ~isempty(gcp('nocreate'))  % ������г��Ѿ�����
    delete(gcp('nocreate'));  % �ر����еĲ��г�
end
willnum=0;
Samplenumbers_td=zeros(1,length(allclassinfor_td)-1);
for k4=1:length(allclassinfor_td)-1
    curclass_td=str2num(allclassinfor_td{k4});
    samplenumbers_td = sum(td_idx==k4)+willnum;
    statsclasscol_toiniimageidx_c = position_info(strcmp(allclassinfor_arrays,allclassinfor_td(k4)));
    statsclasscol_toiniimageidx = statsclasscol_toiniimageidx_c{:};
    allstatsclasscol_index = [croppedData(statsclasscol_toiniimageidx).index];
    allstatsclasscol_boxbounding = {croppedData(statsclasscol_toiniimageidx).position};
    boundingCounts_toindex = cellfun(@(x) size(x, 1), allstatsclasscol_boxbounding);  % ÿ�� bounding ����Ŀ��
    if sum(boundingCounts_toindex) < samplenumbers_td
        willnum=samplenumbers_td-sum(boundingCounts_toindex);
        Samplenumbers_td(k4)=sum(boundingCounts_toindex);
    else
        willnum=0;
        Samplenumbers_td(k4)=samplenumbers_td;
    end
end

parfor k5=1:length(allclassinfor_td)-1
    samplenumbers_td = Samplenumbers_td(k5);
    statsclasscol_toiniimageidx_c = position_info(strcmp(allclassinfor_arrays,allclassinfor_td(k5)));
    statsclasscol_toiniimageidx = statsclasscol_toiniimageidx_c{:};
    allstatsclasscol_index = [croppedData(statsclasscol_toiniimageidx).index];
    allstatsclasscol_boxbounding = {croppedData(statsclasscol_toiniimageidx).position};
    boundingCounts_toindex = cellfun(@(x) size(x, 1), allstatsclasscol_boxbounding);  % ÿ�� bounding ����Ŀ��
    %% ���ݵ�ǰ��idxȷ������λ�ó�ȡ��Ӱ��ĸ���extractPerEntry
    if numel(boundingCounts_toindex)>samplenumbers_td
        randsample_idx=randperm(numel(boundingCounts_toindex), samplenumbers_td);
        allstatsclasscol_index = allstatsclasscol_index(randsample_idx);
        allstatsclasscol_boxbounding={allstatsclasscol_boxbounding{randsample_idx}};
        extractPerEntry = ones(1, samplenumbers_td);
        numEntries = numel(extractPerEntry);  % ���������� idx ������
    else
        numEntries = numel(boundingCounts_toindex);  % ���������� idx ������
        extractPerEntry = ones(1, numEntries);  % ÿ�� idx ���ٳ�ȡ 1 �� bounding

        % ʣ����Ҫ��ȡ�� bounding ����
        remainingToExtract = samplenumbers_td - numEntries;  % ���������ʣ����Ҫ��ȡ������

        if remainingToExtract > 0
            % �����ʼ�ķ��������ȷ������������� bounding ��
            proportions = (boundingCounts_toindex-extractPerEntry) / sum((boundingCounts_toindex-extractPerEntry));
            extractIncrement = min(round(proportions * remainingToExtract), boundingCounts_toindex - extractPerEntry);
            extractPerEntry = extractPerEntry + extractIncrement;

            % ����ʣ��ĳ�ȡ����
            remainingToExtract = samplenumbers_td - sum(extractPerEntry);

            % ��������������ʣ��� bounding ������ֱ�� remainingToExtract Ϊ 0
            while remainingToExtract > 0
                % ʹ���߼�����������Ƶ������ find
                availableIdx = find(boundingCounts_toindex > extractPerEntry);  % �����Գ�ȡ�� idx
                availableBoundingCounts = boundingCounts_toindex(availableIdx);
                if ~any(availableIdx)  % ������е� idx �������أ�����
                    break;
                end
                if length(availableIdx) < remainingToExtract
                    proportions = (boundingCounts_toindex-extractPerEntry) / sum((boundingCounts_toindex-extractPerEntry));
                    extractIncrement = min(round(proportions * remainingToExtract), boundingCounts_toindex - extractPerEntry);
                    extractPerEntry = extractPerEntry + extractIncrement;

                    % ����ʣ��ĳ�ȡ����
                    remainingToExtract = samplenumbers_td - sum(extractPerEntry);
                else
                    randomIndices = availableIdx(randperm(numel(availableIdx), remainingToExtract));
                    extractPerEntry(randomIndices) = extractPerEntry(randomIndices) + 1;
                    remainingToExtract = samplenumbers_td - sum(extractPerEntry);
                end
            end
        end
    end
 %% ����ÿ��idx��Ӧ������extractPerEntry���г�ȡ��Ӱ��
    for k6 = 1:numEntries
        iniimage_idx = allstatsclasscol_index(k6);
        numToExtract = extractPerEntry(k6);
        numAvailable = allstatsclasscol_boxbounding{k6};

        % �ӵ�ǰ idx �в��Ż������ȡ bounding
        selectedIdx = randperm(size(numAvailable,1), numToExtract);
        for k7 = 1:length(selectedIdx)
            sample_index=selectedIdx(k7);
            extractedBounding = numAvailable(sample_index,:);
            image=imread([imagepath imagedir(iniimage_idx).name]);
            label=imread([labelpath  imagedir(iniimage_idx).name]);
            image_c=imcrop(image,extractedBounding);
            label_c=imcrop(label,extractedBounding);
            if size(label_c,1)~=s || size(label_c,2)~=s
                keyboard;
            end
            strr=strcat('image',num2str(k5),'_',num2str(iniimage_idx),'_',num2str(k5),'_',num2str(k7),'.png');
            imwrite(uint8(image_c),([save_image_path strr]))
            imwrite(uint8(label_c),([save_micro_path strr]))
        end
    end
end