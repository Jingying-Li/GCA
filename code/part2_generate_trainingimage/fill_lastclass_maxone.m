function [basic_image,basic_label,fill_rotate,choose_basic_index]=fill_lastclass_maxone(label_micro,imagepath,labelpath,...
    upbasicunit_infor,choose_basic_index,last_fillcalss,noreapeat_idx)

willbasic_windows=size(label_micro);
curclassbasic_allarea = upbasicunit_infor(last_fillcalss).allcurclass_area; %��ȡ���㵱ǰ����basic unit��area����
curclass_originalid = upbasicunit_infor(last_fillcalss).alloriginal_id;
curclass_BoundingBox = upbasicunit_infor(last_fillcalss).allcurclass_BoundingBox;

statsneed_classindex=1:length(curclassbasic_allarea); %���Żس���
delete_chooseindex=unique([choose_basic_index{last_fillcalss}]);           
if length(delete_chooseindex)==length(statsneed_classindex)%��־�ŵ�ǰ������е�basic unit���ѳ�ȡ���������¶��γ�ȡ
    choose_basic_index{last_fillcalss}=[];
    delete_chooseindex=unique([choose_basic_index{last_fillcalss}]);
end
statsneed_classindex(delete_chooseindex)=[];%�޳���ȡ����basic unit��������Ϣ 
if sum(curclass_BoundingBox(statsneed_classindex,3)>= willbasic_windows(1) & curclass_BoundingBox(statsneed_classindex,4)>= willbasic_windows(2))~=0
    Allsubregion_indices= find((curclass_BoundingBox(statsneed_classindex,3)>= willbasic_windows(1) & curclass_BoundingBox(statsneed_classindex,4)>= willbasic_windows(2))==1);%��ȡstatsneed_classindex��������������������ֵ
    if isempty(Allsubregion_indices)
        Allsubregion_indices= find((curclass_BoundingBox(:,3)>= willbasic_windows(1) & curclass_BoundingBox(:,4)>= willbasic_windows(2))==1);
        Allsubregion_indices_nprempat=setdiff(Allsubregion_indices,noreapeat_idx);
        thres=1;
        while isempty(Allsubregion_indices_nprempat)
            thres=thres-0.01;
            Allsubregion_indices= find((curclass_BoundingBox(:,3)>= willbasic_windows(1)*thres & curclass_BoundingBox(:,4)>=willbasic_windows(2)*thres)==1);
            Allsubregion_indices_nprempat=setdiff(Allsubregion_indices,noreapeat_idx);
        end
        selectarea_allinfor=curclassbasic_allarea(Allsubregion_indices_nprempat);
        Allsubregion_indices=Allsubregion_indices_nprempat(selectarea_allinfor>willbasic_windows(1)*willbasic_windows(2)*0.7);
    end
else
    all_area=curclass_BoundingBox(statsneed_classindex,3).*curclass_BoundingBox(statsneed_classindex,4);
    max_indices= all_area==max(all_area);
    Allsubregion_indices=statsneed_classindex(max_indices);
end

chooseimage_index=Allsubregion_indices(randperm(length(Allsubregion_indices),1));
if chooseimage_index>length(curclassbasic_allarea)
    keyboard
end
noreapeat_idx=[noreapeat_idx,chooseimage_index];
choose_basic_index{last_fillcalss}=[choose_basic_index{last_fillcalss} chooseimage_index];%��¼���г�ȡ����basic unit������Ϣ
original_id=curclass_originalid(chooseimage_index);
labelDir=dir([labelpath '*.png']);
basic_image=imread([imagepath labelDir(original_id).name]);
basic_label=imread([labelpath labelDir(original_id).name]);
%�������ڵ�ǰֵ�����ȫ��ע��Ϊ0
basic_label(basic_label~=last_fillcalss)=0;
[fill_cpixelpoints,fill_rotate]=pick_suitarea_allrotate(basic_label,label_micro);
[curbasic_row,curbasic_col]=size(basic_label);
crop_row=min(curbasic_row,willbasic_windows(1));
crop_col=min(curbasic_col,willbasic_windows(2));
basic_label_c=imcrop(basic_label,[fill_cpixelpoints(2) fill_cpixelpoints(1)...
               crop_col-1 crop_row-1]);
basic_image_c=imcrop(basic_image,[fill_cpixelpoints(2) fill_cpixelpoints(1)...
               crop_col-1 crop_row-1]);
basic_image=basic_image_c;
basic_label=basic_label_c;