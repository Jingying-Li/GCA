function [Cursameclass,Curclass_areavalue,choose_basic_index,choose_basicidx]=pick_basic_unit(Given_infor,...
                curclass_infor,choose_basic_index,choose_basicidx,Withcontain_classinfor_all)
%% ����Ŀ�ģ�
%  �ӵ�ǰ���curfill_class��Ӧ��basic unit�����г�ȡarea֮����[curfill_areanum,1.1curclass_area]֮���basic
%  unit,�ҳ�ȡ��basic unit�ĸ���Ϊcurfill_areanum,��ȷ����ǰ���basic unit��otherclass_indexcol��λ�ù�ϵ
% Input��
%   Given_infor:����һЩ�̶���Ϣ(path information,basic unit,s)
%   curclass_infor:�ṹ��,������ǰ����ȡ���Ļ�����Ϣ
%       curfill_areanum:��ǰ����basic unit����
%       curfill_class:��ǰ���ֵ
%       choose_basic_index:��¼���г�ȡ��basic unit������Ϣ
%       curclass_area:��ǰ����area�ܺ�
%       choose_classcol:��ǰѵ�����ݵ���𼯺�
%       em_toclass: ������
% Output:
%   Cursameclass(��СΪcurfill_areanum*7��Ԫ��):��ȡ��basic unit�ĸ�����Ϣ,��ÿһ�зֱ��ʾbasic
%                                    unit��label,image,curfill_class,sparity,georelation,size
%   Curclass_areavalue:��ȡ��basicunit�ĸ���area��Ϣ
%   choose_basic_index:���º��choose_basic_index
%   chooseclass_histogram����ȡ�������basic unit��Ӧ�ķֲ���Ϣ

%% ������Ϣ����
imagepath=Given_infor.imagepath;
labelpath=Given_infor.labelpath;
imageDir=dir([labelpath '*.png']);
upbasicunit_infor = Given_infor.upbasicunit_infor;

curfill_class=curclass_infor.curfill_class;
curclass_area=curclass_infor.curclass_area;
curfill_areanum=curclass_infor.curfill_areanum;
choose_classcol=curclass_infor.choose_classcol;
curclasswithotherclass_allgeorelation=curclass_infor.curclasswithotherclass_allgeorelation;     

% ȷ����ȡ��Ϣ��basic unit,������ȡ
if ~isempty(Withcontain_classinfor_all)
    allcontainclasscol = unique(cat(1,Withcontain_classinfor_all.classinfor));
    if ismember(curfill_class,allcontainclasscol)
        class_equalidx=find(cat(1,Withcontain_classinfor_all.classinfor)==curfill_class);
        Clearclass_areanum=0; Clearclass_areasum=0; Clearclass_area=[];withotherclassidx=[];
        for k=1:numel(class_equalidx)
            curclassidx=class_equalidx(k);
            withotherclassidx=[withotherclassidx repmat(Withcontain_classinfor_all(curclassidx).containclassidx,[1 Withcontain_classinfor_all(curclassidx).Area_num])];
            Clearclass=Withcontain_classinfor_all(curclassidx).classinfor;
            Clearclass_areanum=Clearclass_areanum+Withcontain_classinfor_all(curclassidx).Area_num;
            Clearclass_areasum=Clearclass_areasum+Withcontain_classinfor_all(curclassidx).Areasum;
            Clearclass_area=[Clearclass_area Withcontain_classinfor_all(curclassidx).Area];
        end
        curclass_area=curclass_area-Clearclass_areasum;
        if curclass_area<=0
            curclass_area=0;
            curfill_areanum=0;
        end
        curfill_areanum=curfill_areanum-Clearclass_areanum;
        if curfill_areanum<=0 && curclass_area>0
            curfill_areanum=1;
        elseif curfill_areanum<=0 && curclass_area==0
            curfill_areanum=0;
        end
    else
        Clearclass=[];
        Clearclass_areanum=0;
        withotherclassidx=0;
    end
else
    Clearclass=[];
    Clearclass_areanum=0;
    withotherclassidx=0;
end
    

%��ȡ��ǰ����basic unit����Ϣ��area,��ʼ��(��ԭʼӰ����), ������Ϣ(��s*s��Ӱ����), ������������������������, ��������
curclassbasic_allarea = upbasicunit_infor(curfill_class).allcurclass_area; %��ȡ���㵱ǰ����basic unit��area����
curclass_alloriginalid = upbasicunit_infor(curfill_class).alloriginal_id;
curclass_allBoundingBox = upbasicunit_infor(curfill_class).allcurclass_BoundingBox;
statsneed_allclassindex=1:length(curclassbasic_allarea); %���Żس���
%% �жϵ�ǰ����Ƿ���������������������������Ӧ�������ƶ�>0.8���ϵ�basic unit���г�ȡ
if curfill_areanum~=0
    statsneed_classindex_ori=statsneed_allclassindex;
    % �޳�����ȡ��basic unit��Ϣ
    delete_chooseindex=ismember(statsneed_classindex_ori, [choose_basic_index{curfill_class}]); 
    statsneed_classindex_ori(delete_chooseindex)=[];  %��ȡ��ǰ�������ϣ����Żس���
    % �ֲ��ȡbasic unit����Ϊ����Ӱ����Ϣ��
    statsnees_classidx = curclass_alloriginalid(statsneed_classindex_ori);
    choose_id_already=unique([choose_basicidx{curfill_class}]);
    if length(choose_id_already)==length(unique(curclass_alloriginalid))
        choose_basicidx{curfill_class}=[];
    end
    statsneed_classindex=statsneed_classindex_ori(~ismember(statsnees_classidx,[choose_basicidx{curfill_class}]));
    %��basic unitȫ����ȡ����������³�ȡ
    if isempty(statsneed_classindex) || length(statsneed_classindex)<curfill_areanum 
        statsneed_classindex=statsneed_classindex_ori;
    end
    if isempty(statsneed_classindex)
        choose_basic_index{curfill_class}=[];
        statsneed_classindex=1:length(curclassbasic_allarea);
    end

    %% �����ϲ���ȡ��������Ϣ�Լ���Ӧ��������ֵ���г�ȡ
    recurclassbasic_allarea = curclassbasic_allarea(statsneed_classindex); % �޳��ѳ�ȡbasic unit��area��Ϣ(����һ����Ϣ���Ӧ)
    [selected_area, selected_indices] = selectbasicareaWithIndices(recurclassbasic_allarea, curclass_area, curfill_areanum); %��ȡ�����ݳ�ȡ��ȷ����ȡbasic unit����Ϣ
    selected_idcol=statsneed_classindex(selected_indices); %��¼��ȡbasic unit��������������
    Curclass_areavalue=selected_area; %��¼��ȡbasic unit������area��Ϣ����
    choose_basic_index{curfill_class}=[choose_basic_index{curfill_class} selected_idcol];%��¼���г�ȡ����basic unit������Ϣ
else
    selected_idcol=[];
    Curclass_areavalue=Clearclass_area;
end

%% step2����ȡָ�������С��basic unit��Ϣ
if ~isempty(Clearclass)
    Statsclearclass_idx=zeros(1,length(Clearclass_area));
    for k2=1:length(Clearclass_area)
        curarea_classclear=Clearclass_area(k2);
        statsneed_classclearindex_ori=statsneed_allclassindex;
        % �޳�����ȡ��basic unit��Ϣ
        delete_chooseindex=ismember(statsneed_classclearindex_ori, [choose_basic_index{curfill_class} Statsclearclass_idx]); 
        statsneed_classclearindex_ori(delete_chooseindex)=[]; 
        curchoose_allarea=curclassbasic_allarea(statsneed_classclearindex_ori);
        [~,statschoose_mididx]=min(abs(curchoose_allarea-curarea_classclear));
        statschoose_idx=statsneed_classclearindex_ori(statschoose_mididx);
        Statsclearclass_idx(k2)=statschoose_idx;
    end
    selected_idcol=[selected_idcol Statsclearclass_idx];
    choose_basic_index{curfill_class}=[choose_basic_index{curfill_class} Statsclearclass_idx];%��¼���г�ȡ����basic unit������Ϣ
end
  
%% step3:�洢basic unit��Ϣ
%��ʼ����ֵ
Cursameclass=cell(length(selected_idcol),9);%��һ��Ϊ��ͬ����¸�������ı߽���Ϣ���ڶ���Ϊ����������(s,s)�ϵ�Ӱ����Ϣ
for k1=1:length(selected_idcol)%curfill_areanum
    curselected_id=selected_idcol(k1);  %��ѭ���µ�basic unit��Ӧ������ֵ(��Ӧ��������Ϊstatsneed_allclassindex)
    chooseoriginalimage_index=curclass_alloriginalid(curselected_id);%�������������������basic unitչ�����������Ϣ 
    choose_basicidx{curfill_class}=[choose_basicidx{curfill_class} chooseoriginalimage_index];
    current_basicimage=imread([imagepath imageDir(chooseoriginalimage_index).name]); %��ȡԭʼ������Ϣ
    current_basiclabel=imread([labelpath imageDir(chooseoriginalimage_index).name]);
    
    %��ȡ��ǰѡȡ��basic unit��Ϣ
    statscurclass_boundingbox=curclass_allBoundingBox(curselected_id,:)-[0 0 1 1];%[�� �� �� ��]
    current_basicimage=imcrop(current_basicimage,statscurclass_boundingbox);
    current_basiclabel=imcrop(current_basiclabel,statscurclass_boundingbox);
    %�������ڵ�ǰֵ�����ȫ��ע��Ϊ0
    current_basiclabel(current_basiclabel~=curfill_class)=0;
    current_basicimage=current_basicimage.*uint8(repmat((current_basiclabel~=0),[1,1,3]));
    current_basicimage(all(current_basiclabel==0,2),:,:)=[];
    current_basicimage(:,all(current_basiclabel==0,1),:)=[];
    current_basiclabel(all(current_basiclabel==0,2),:)=[];
    current_basiclabel(:,all(current_basiclabel==0,1))=[];
    [curbasic_row,curbasic_col]=size(current_basiclabel); 
    if isempty(current_basiclabel)
        continue;
    end
    Cursameclass{k1,1} = current_basiclabel;
    Cursameclass{k1,2} = current_basicimage;
    Cursameclass{k1,3} = curfill_class;
    Cursameclass{k1,4} = (sum(current_basiclabel(:)~=0))/(curbasic_row*curbasic_col);
    Cursameclass{k1,5} = curbasic_row*curbasic_col;
    %���
    %����������ӵ�,��ȡ���ϸ���������λ�ù�ϵ
    classwithothergeorelation_seed=rand;
    %keyboard
    %�Ȼ�ȡ��ǰ�������һ�����λ�ù�ϵ
    curclass_withotherclass_allgeorelation=sum(curclasswithotherclass_allgeorelation);
    selected_withotherclass_index = find(classwithothergeorelation_seed <= cumsum(curclass_withotherclass_allgeorelation), 1, 'first');
    selected_withotherclass_value = choose_classcol(selected_withotherclass_index); %��ȡ����λ�ù�ϵ�����ֵ
    selected_withotherclass_georelation = curclasswithotherclass_allgeorelation(:,selected_withotherclass_index); %��ȡ����ڵĿ�����[���� ��Χ]
    Cursameclass{k1,6} = selected_withotherclass_value;
    Cursameclass{k1,7} = selected_withotherclass_georelation; %��¼��selected_withotherclass_value�Ĺ�ϵ����[���� ��Χ]
    binary_mask=current_basiclabel~=0;
    props = regionprops(binary_mask, 'Centroid');
    Cursameclass{k1,8}=props.Centroid;
    Cursameclass{k1,9}=0;
    %�жϱ�����basic��Ϣ 
    if length(selected_idcol)-k1<Clearclass_areanum
       % keyboard
        Cursameclass{k1,10}=withotherclassidx(k1-(length(selected_idcol)-length(withotherclassidx)));
    else
        Cursameclass{k1,10}=0;
    end
    Cursameclass{k1,11}=[];
end

  