function [Cursameclass,Curclass_areavalue,choose_basic_index,choose_basicidx]=pick_basic_unit(Given_infor,...
                curclass_infor,choose_basic_index,choose_basicidx,Withcontain_classinfor_all)
%% 函数目的：
%  从当前类别curfill_class对应的basic unit集合中抽取area之和在[curfill_areanum,1.1curclass_area]之间的basic
%  unit,且抽取的basic unit的个数为curfill_areanum,并确定当前类别basic unit与otherclass_indexcol的位置关系
% Input：
%   Given_infor:包含一些固定信息(path information,basic unit,s)
%   curclass_infor:结构体,包含当前待抽取类别的基本信息
%       curfill_areanum:当前类别的basic unit个数
%       curfill_class:当前类别值
%       choose_basic_index:记录所有抽取的basic unit索引信息
%       curclass_area:当前类别的area总和
%       choose_classcol:当前训练数据的类别集合
%       em_toclass: 误分类别
% Output:
%   Cursameclass(大小为curfill_areanum*7的元组):抽取的basic unit的各项信息,其每一列分别表示basic
%                                    unit的label,image,curfill_class,sparity,georelation,size
%   Curclass_areavalue:抽取的basicunit的各个area信息
%   choose_basic_index:更新后的choose_basic_index
%   chooseclass_histogram：抽取误分类别的basic unit对应的分布信息

%% 基本信息加载
imagepath=Given_infor.imagepath;
labelpath=Given_infor.labelpath;
imageDir=dir([labelpath '*.png']);
upbasicunit_infor = Given_infor.upbasicunit_infor;

curfill_class=curclass_infor.curfill_class;
curclass_area=curclass_infor.curclass_area;
curfill_areanum=curclass_infor.curfill_areanum;
choose_classcol=curclass_infor.choose_classcol;
curclasswithotherclass_allgeorelation=curclass_infor.curclasswithotherclass_allgeorelation;     

% 确定抽取信息的basic unit,单独抽取
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
    

%获取当前类别的basic unit的信息：area,初始点(在原始影像上), 区域信息(在s*s的影像上), 该类易误分类的所有索引集合, 索引集合
curclassbasic_allarea = upbasicunit_infor(curfill_class).allcurclass_area; %获取满足当前类别的basic unit的area集合
curclass_alloriginalid = upbasicunit_infor(curfill_class).alloriginal_id;
curclass_allBoundingBox = upbasicunit_infor(curfill_class).allcurclass_BoundingBox;
statsneed_allclassindex=1:length(curclassbasic_allarea); %不放回抽样
%% 判断当前类别是否属于误分类别，若属于则根据其与对应类别的相似度>0.8以上的basic unit进行抽取
if curfill_areanum~=0
    statsneed_classindex_ori=statsneed_allclassindex;
    % 剔除已提取的basic unit信息
    delete_chooseindex=ismember(statsneed_classindex_ori, [choose_basic_index{curfill_class}]); 
    statsneed_classindex_ori(delete_chooseindex)=[];  %获取当前索引集合，不放回抽样
    % 分层抽取basic unit（层为各个影像信息）
    statsnees_classidx = curclass_alloriginalid(statsneed_classindex_ori);
    choose_id_already=unique([choose_basicidx{curfill_class}]);
    if length(choose_id_already)==length(unique(curclass_alloriginalid))
        choose_basicidx{curfill_class}=[];
    end
    statsneed_classindex=statsneed_classindex_ori(~ismember(statsnees_classidx,[choose_basicidx{curfill_class}]));
    %当basic unit全被抽取，则二次重新抽取
    if isempty(statsneed_classindex) || length(statsneed_classindex)<curfill_areanum 
        statsneed_classindex=statsneed_classindex_ori;
    end
    if isempty(statsneed_classindex)
        choose_basic_index{curfill_class}=[];
        statsneed_classindex=1:length(curclassbasic_allarea);
    end

    %% 根据上步获取的索引信息以及对应的像素总值进行抽取
    recurclassbasic_allarea = curclassbasic_allarea(statsneed_classindex); % 剔除已抽取basic unit的area信息(与上一个信息相对应)
    [selected_area, selected_indices] = selectbasicareaWithIndices(recurclassbasic_allarea, curclass_area, curfill_areanum); %抽取：根据抽取数确定抽取basic unit的信息
    selected_idcol=statsneed_classindex(selected_indices); %记录抽取basic unit的所有索引集合
    Curclass_areavalue=selected_area; %记录抽取basic unit的所有area信息集合
    choose_basic_index{curfill_class}=[choose_basic_index{curfill_class} selected_idcol];%记录所有抽取过的basic unit索引信息
else
    selected_idcol=[];
    Curclass_areavalue=Clearclass_area;
end

%% step2：获取指定区域大小的basic unit信息
if ~isempty(Clearclass)
    Statsclearclass_idx=zeros(1,length(Clearclass_area));
    for k2=1:length(Clearclass_area)
        curarea_classclear=Clearclass_area(k2);
        statsneed_classclearindex_ori=statsneed_allclassindex;
        % 剔除已提取的basic unit信息
        delete_chooseindex=ismember(statsneed_classclearindex_ori, [choose_basic_index{curfill_class} Statsclearclass_idx]); 
        statsneed_classclearindex_ori(delete_chooseindex)=[]; 
        curchoose_allarea=curclassbasic_allarea(statsneed_classclearindex_ori);
        [~,statschoose_mididx]=min(abs(curchoose_allarea-curarea_classclear));
        statschoose_idx=statsneed_classclearindex_ori(statschoose_mididx);
        Statsclearclass_idx(k2)=statschoose_idx;
    end
    selected_idcol=[selected_idcol Statsclearclass_idx];
    choose_basic_index{curfill_class}=[choose_basic_index{curfill_class} Statsclearclass_idx];%记录所有抽取过的basic unit索引信息
end
  
%% step3:存储basic unit信息
%初始化赋值
Cursameclass=cell(length(selected_idcol),9);%第一列为相同类别下各个区域的边界信息，第二列为各个区域在(s,s)上的影像信息
for k1=1:length(selected_idcol)%curfill_areanum
    curselected_id=selected_idcol(k1);  %该循环下的basic unit对应的索引值(对应索引集合为statsneed_allclassindex)
    chooseoriginalimage_index=curclass_alloriginalid(curselected_id);%将所有满足该类索引的basic unit展开后的索引信息 
    choose_basicidx{curfill_class}=[choose_basicidx{curfill_class} chooseoriginalimage_index];
    current_basicimage=imread([imagepath imageDir(chooseoriginalimage_index).name]); %获取原始数据信息
    current_basiclabel=imread([labelpath imageDir(chooseoriginalimage_index).name]);
    
    %获取当前选取的basic unit信息
    statscurclass_boundingbox=curclass_allBoundingBox(curselected_id,:)-[0 0 1 1];%[列 行 宽 高]
    current_basicimage=imcrop(current_basicimage,statscurclass_boundingbox);
    current_basiclabel=imcrop(current_basiclabel,statscurclass_boundingbox);
    %将不等于当前值的类别全部注释为0
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
    %填充
    %生成随机种子点,获取与上个类别区域的位置关系
    classwithothergeorelation_seed=rand;
    %keyboard
    %先获取当前类别与哪一类存在位置关系
    curclass_withotherclass_allgeorelation=sum(curclasswithotherclass_allgeorelation);
    selected_withotherclass_index = find(classwithothergeorelation_seed <= cumsum(curclass_withotherclass_allgeorelation), 1, 'first');
    selected_withotherclass_value = choose_classcol(selected_withotherclass_index); %获取有着位置关系的类别值
    selected_withotherclass_georelation = curclasswithotherclass_allgeorelation(:,selected_withotherclass_index); %获取其存在的可能行[相邻 包围]
    Cursameclass{k1,6} = selected_withotherclass_value;
    Cursameclass{k1,7} = selected_withotherclass_georelation; %记录与selected_withotherclass_value的关系矩阵[相邻 包围]
    binary_mask=current_basiclabel~=0;
    props = regionprops(binary_mask, 'Centroid');
    Cursameclass{k1,8}=props.Centroid;
    Cursameclass{k1,9}=0;
    %判断被包含basic信息 
    if length(selected_idcol)-k1<Clearclass_areanum
       % keyboard
        Cursameclass{k1,10}=withotherclassidx(k1-(length(selected_idcol)-length(withotherclassidx)));
    else
        Cursameclass{k1,10}=0;
    end
    Cursameclass{k1,11}=[];
end

  