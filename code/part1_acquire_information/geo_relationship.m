function [Intrainfor,Difclass_pixelsnum,AreaRelationship]=geo_relationship(gray_value,image,label,ignored_value,pro,N)
%geo_relationship���������ã�
%    geo_relationship(gray_value,image,label,ignored_value,pro,k);
%   (1) module B��ʵ�֣�For each raw observed image and its ground truth, it extracts basic unit information for each class present in the subregion.
%   (2) ��ȡ����ϵ��Ϣ: adjcant_relationship,contain_relationship,containwith_relationship
%
% Output��(1)gray_value�����ݼ������ (2)label:ground truth (3)ignored_value:���Ա�ǩ��Ӧ�ĻҶ�ֵ
%         (4)pro:basic unit��area��ֵ����(�б�basic unit�Ƿ���Ч) (5)k:ԭʼ���ݵ�˳������
%
% Intrainfor��12���ֶ���ɣ�  
%        (1)Image_index:��¼ԭʼ����˳������        
%        (2)Current_class:��¼��ǰbasic unit�����ֵ(��������������ֵ���б�������ȡbasic unit)
%        (3)Area_num:��¼��������Current_class�µ�basic unit������
%        (4)Area(size:Area_num*1)����¼��������Current_class�µĸ���basic unit����Ч��������
%        (5)Centroid(size:Area_num*2)����¼��������Current_class�µĸ���basic unit����������
%        (6)BoundingBox(size:Area_num*4)����¼��������Current_class�µĸ���basic unit��BoundingBox��Ϣ
%        (7)Sparity(size:Area_num*1):��¼��������Current_class�µĸ���basic unit��ϡ�����Ϣ
%        (8)Distance_infor(size:Area_num*Area_num):��¼��������Current_class�µĸ���basic unit֮��ľ�����Ϣ

%% initialization
% Intrainfor=struct('Image_index',[],'Current_class',[],'Area',[],'Centroid',[],'BoundingBox',[], 'Sparity',[],'Sameclass_Disvalue',[],...
%            'Boundary_judgment',[],'counts_histogram',[]);  
adjcant_relationship=zeros(length(gray_value));
contain_relationship=zeros(length(gray_value));
containwith_relationship=zeros(length(gray_value));
% Difclass_pixelsnum=struct('Image_index',[],'Current_class',[],'Pixelsnum',[]);
[m,n]=size(label); %��ǰӰ��ĳߴ�
classcol=unique(label); %��ǰӰ�����𼯺�
classcol(classcol==ignored_value)=[];%ɾ��0ֵ
s=512;

%% ����regionprops�������μ�¼Ӱ���и�������µ�������Ϣ(basic unit)
% ע����֮��Ĵ�����(�������49-163��)��Ի�ȡ��������Ϣ���ж����ж�: 
%     ����ȡ��basic unit�ĳߴ���s*s�ڣ�����Ϊ��Ϊbasic unit��
%     ��֮����ȥ���ռ�¼�������С�ߴ�ı�����Ϊ���ڳߴ磬��һ��������basic unit���л�ȡ
isempty_idx = [];
Intrainfor = []; % Ԥ�ȶ���ṹ������
Difclass_pixelsnum = [];
parfor k1=1:length(classcol)
    currentclass_region_logits=(label==classcol(k1)); %��ȡlabelӰ�������ֵΪclasscol(k1)��Ӧ���߼�����
    currentclass_region_imagelogits=repmat(currentclass_region_logits,[1 1 3]);
    currentclass_image=image.*uint8(currentclass_region_imagelogits);
    current_class_indices=find(gray_value==classcol(k1));%��ȡ�ڸ�ѭ���¶�Ӧ���������(Ҳ�����ֵ)
    current_classnum=sum(currentclass_region_logits(:));
    stats_c = regionprops(currentclass_region_logits,'Area', 'Centroid', 'BoundingBox');
    
    % �޳����С����ֵ������
    toosmall_logical = [stats_c.Area] < s*s*pro; % �߼���������ʾ���С����ֵ������
    stats_c(toosmall_logical) = []; % �޳�һЩ������Ϣ����С������Ϣ��
    if isempty(stats_c)
        isempty_idx=[isempty_idx k1];
        continue
    end
    %��ȡ�ڸ�ѭ���µ�basic unit����Ϣ��
    local_intrainfor = []; % ��ʱ���浱ǰ������ Intrainfor
    local_difclass_pixelsnum = []; % ��ʱ���浱ǰ������ Difclass_pixelsnum
    for k2 = 1:length(stats_c)
        % 1. ���㵱ǰ����¸��� basic unit ֮���ŷʽ����
        centroids = reshape([stats_c.Centroid], 2, []).';
        distances = pdist2(centroids, centroids);
        distances(eye(size(distances)) == 1) = inf; % ���Խ��߾�������Ϊ����󣬱����ԱȽ�
        Sameclass_Disvalue = min(distances(k2, :));
        
        % 2. ��ǰ����area��Ϣ
        area_currentclass=stats_c(k2).Area;
        % 3. area��������Ϣ����
        area_centercol=stats_c(k2).Centroid;
        % 4. area��BoundingBox��Ϣ����
        curbasic_box=stats_c(k2).BoundingBox;
        area_boxcol=curbasic_box;         
        % 5. ����area��ϡ���
        Spar=stats_c(k2).Area/(curbasic_box(3)*curbasic_box(4)); 
        % 6. basic unit�ı߽��ж�
        % 6.1 ��ʼ����ж�
        if sum(ismember(ceil([curbasic_box(1) curbasic_box(2)]),[1 m n]))>=1 || ... % ��ʼ���ж�
                sum(ismember(ceil([curbasic_box(1) curbasic_box(2)])+[curbasic_box(3) curbasic_box(4)],[1 m n]))>=1 % ĩβ���ж�
            Boundary_judg=1;%��ʾ��basic unit���ı���
        else
            Boundary_judg=2;
        end
        %7. counts_histogram����
        currenregion_image=imcrop(currentclass_image,curbasic_box);
        currenregion_label=imcrop(currentclass_region_logits,curbasic_box);
        current_onlyoneclassimage=currenregion_image(currenregion_label);
        [countsA,~] = histcounts(current_onlyoneclassimage, 256);
        % ���������µĵ�ǰ����basic unit������6����Ϣ���ܵ�region_info�ṹ����
        curclass_info = struct('Image_index',N,'Current_class',current_class_indices,'Area',area_currentclass,...
             'Centroid',area_centercol,'BoundingBox',area_boxcol,'Sparity',Spar,'Sameclass_Disvalue', ...
             Sameclass_Disvalue,'Boundary_judgment',Boundary_judg,'counts_histogram',countsA);
        difclass_pixelsnum=struct('Image_index',N,'Current_class',current_class_indices,'Pixelsnum',current_classnum);
        
        local_intrainfor = [local_intrainfor; curclass_info];
        local_difclass_pixelsnum = [local_difclass_pixelsnum; difclass_pixelsnum];
    end
    % �ϲ���ǰ�����Ľ�������ս����
    Intrainfor = [Intrainfor; local_intrainfor];
    Difclass_pixelsnum = [Difclass_pixelsnum; local_difclass_pixelsnum];
end
% ����������Ϣ��ȡ���
        
%% ��ȡ��ǰӰ��������Ϣ��(��)��Χ��ϵ�����ڹ�ϵ
% �����Ϣ���ж�:(1)����ǰ��������������ͣ���ȡ���Ӧ�ı߽����꣬
%               (2)���������ȡ���Ӱ���Ӧ������ֵ,������ֵ�����жϣ�
%                   * ���߽�����ֵ��Ϊͬһֵ�����ʾ�������򱻰�Χ����¼��
%                   * ���߽�����ֵΪ���ֵ�����ʾ��ǰ���������������ڣ���¼��
classcol(isempty_idx)=[]; %isempty_idx�Ǹ�����»�ȡ��basic unit�ߴ����С,���迼�ǣ���������ｫ�޳��������Ϣ
adjcant_relationship_pro=cell(1,length(classcol));
parfor k3=1:length(classcol)
    current_class_indices = double(classcol(k3)); %��ȡ�ڸ�ѭ���¶�Ӧ���������(Ҳ�����ֵ)
    labeldilate_logiuts = (label==current_class_indices); %��ȡ��ǰ����Ӧ������߼�����
    %����regionprops��ȡ���ͺ�Ӱ��������ı߽���Ϣ�������ݸñ߽���Ϣ��ȡ����label�϶�Ӧ������ֵ���Ӷ���ȡ���ܱ�������Ϣ
    SE = strel('disk',10); 
    label_dilate=imdilate(labeldilate_logiuts,SE);
    [dilate_labels, ~] = bwlabel(label_dilate);
    area_num=setdiff(unique(dilate_labels),0);
    
    % �����ֲ��������洢ÿ��ѭ���Ľ�������Ⲣ�г�ͻ
    local_adjcant_relationship = zeros(length(gray_value));
    local_adjcant_relationprop = cell(length(gray_value));
    local_contain_relationship = zeros(length(gray_value));
    local_containwith_relationship = zeros(length(gray_value));
    
    for k4=1:length(area_num)
        area_label = dilate_labels==area_num(k4);
        stats_dilate_boundries = bwboundaries(area_label,'noholes');%��ȡ�������׶���Ϣ������߽�����
        area_boundries=stats_dilate_boundries{1,1};% �߽�����
        % sub2ind�������±�ת��Ϊ��������
        linear_indices = sub2ind(size(label), area_boundries(:, 1), area_boundries(:, 2));
        % ��������������ȡ����ֵ
        pixelsvaluecol_label=label(linear_indices);
        geolation_withclass=setdiff(unique(pixelsvaluecol_label),[0 current_class_indices]);
        
        if ~isempty(geolation_withclass)
            if length(geolation_withclass) > 1
                local_adjcant_relationship(current_class_indices, unique(geolation_withclass)) = local_adjcant_relationship(current_class_indices, unique(geolation_withclass)) + 1;
                 % �����������һ��
                Withclasspro = histcounts(pixelsvaluecol_label, [geolation_withclass', max(geolation_withclass) + 1]) / length(linear_indices);
                Withclasspro_nor = Withclasspro / sum(Withclasspro);

                % �������� local_adjcant_relationprop
                local_adjcant_relationprop(current_class_indices, geolation_withclass) = ...
                    cellfun(@(existing, new) [existing, new], ...
                    local_adjcant_relationprop(current_class_indices, geolation_withclass), ...
                    num2cell(Withclasspro_nor), 'UniformOutput', false);

            elseif length(geolation_withclass) == 1
                pixelsvaluecol_boundpro=sum(pixelsvaluecol_label(:)==geolation_withclass)/length(linear_indices);
                if pixelsvaluecol_boundpro>=0.7
                    local_contain_relationship(unique(geolation_withclass),current_class_indices) = local_contain_relationship(unique(geolation_withclass),current_class_indices) + 1;
                    local_containwith_relationship(current_class_indices,unique(geolation_withclass)) = local_containwith_relationship(current_class_indices,unique(geolation_withclass)) + 1;
                else
                    local_adjcant_relationship(current_class_indices, unique(geolation_withclass)) = local_adjcant_relationship(current_class_indices, unique(geolation_withclass)) + 1;
                    local_adjcant_relationprop{current_class_indices, unique(geolation_withclass)} = ...
                        [local_adjcant_relationprop{current_class_indices, unique(geolation_withclass)},pixelsvaluecol_boundpro];
                end
            end
        end
        % ��ͨ��
%         if length(geolation_withclass)>1
%             adjcant_relationship(current_class_indices, unique(geolation_withclass))=adjcant_relationship(current_class_indices, unique(geolation_withclass))+1;
%         else
%             contain_relationship(unique(geolation_withclass),current_class_indices)=contain_relationship(unique(geolation_withclass),current_class_indices)+1;
%             containwith_relationship(current_class_indices,unique(geolation_withclass))=containwith_relationship(current_class_indices,unique(geolation_withclass))+1;
%         end
    end
    % �������㣺�ϲ��ֲ�������ȫ�ֽ����
    adjcant_relationship = adjcant_relationship + local_adjcant_relationship;
    adjcant_relationship_pro{k3} = local_adjcant_relationprop;
    contain_relationship = contain_relationship + local_contain_relationship;
    containwith_relationship = containwith_relationship + local_containwith_relationship;
end
% adjcant_relationship_pro������Ϣ
[rows, cols] = size(adjcant_relationship_pro{1}); % ȷ�� 15x15 �Ĵ�С
adjcant_relationshippro = cell(rows, cols); % ��ʼ���ϲ���� 15x15 Ԫ������
% ����ÿ��λ��
for i = 1:rows
    for j = 1:cols
        % ��ʼ������洢
        arraysToMerge = [];
        for idx = 1:length(adjcant_relationship_pro)
            currentArray = adjcant_relationship_pro{idx}{i, j};
            arraysToMerge = [arraysToMerge; currentArray(:)]; % ����ƴ��
        end
        % ���ϲ��������洢�������
        adjcant_relationshippro{i, j} = arraysToMerge; % ������һ����ֵ����
    end
end

AreaRelationship.adjcant_relationship=adjcant_relationship;
AreaRelationship.adjcant_relationshippro=adjcant_relationshippro;
AreaRelationship.contain_relationship=contain_relationship;
AreaRelationship.containwith_relationship=containwith_relationship;   