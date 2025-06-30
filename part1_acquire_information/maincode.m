clc;
clear;
%��ȡÿ��Ӱ���ϸ�������basic unit����Ϣ������ȡ�������֮��Ŀռ��ϵ(���ڣ���Χ������Χ)
%% configs
imagepath='D:\A_userfile\lly\data\lovada\Train\all_image\';
labelpath='D:\A_userfile\lly\data\lovada\Train\all_label\';
labmicrDir=dir([labelpath '*.png']);
savepath='D:\A_userfile\lly\data\lovada\code\uvaid_code\data_information\';
gray_value=1:7; %��ǰ���ݼ���Ӧ���ĻҶ�ֵ
ignored_value=0;
s=512; %ѵ�����ݵĴ�С
irr_class=[1,3,4];

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
adjcant_relationship_pro=cell(1,length(labmicrDir));
parfor k=1:length(labmicrDir)
    image_name_micro=labmicrDir(k).name;
    label=imread([labelpath image_name_micro]);
    image=imread([imagepath image_name_micro]);
    %��ȡӰ���basic unit��Ϣ�Լ��������֮���λ�ù�ϵ��Ϣ
    [Intrainfor,Difclass_pixelsnum,AreaRelationship]=...
        geo_relationship(gray_value,image,label,ignored_value,pro,k);
    %��ÿ��ԭʼ���ݻ�ȡ������ϢSubregion_intrainfor��Subregion_interinfor�ϲ���Allsubregion_intrainfor��Allsubregion_interinfor
     All_Intrainfor=[All_Intrainfor;Intrainfor];
     Difclass_pixelsnumall=[Difclass_pixelsnumall;Difclass_pixelsnum];
     adjcant_relationship=adjcant_relationship+AreaRelationship.adjcant_relationship;
     contain_relationship=contain_relationship+AreaRelationship.contain_relationship;
     containwith_relationship=containwith_relationship+AreaRelationship.containwith_relationship; 
     adjcant_relationship_pro{k}=AreaRelationship.adjcant_relationshippro;
end
% 
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
Relationship.adjcant_relationship=adjcant_relationship;
Relationship.adjcant_relationshippro=adjcant_relationshippro;
Relationship.contain_relationship=contain_relationship;
Relationship.containwith_relationship=containwith_relationship;

%% extracting_information������Ϣ�������������ݼ�����ֵ����Ľṹ����Ϣ����Ϊ�������ݼ������Ϣ����Ľṹ��
[basicunit_infor,Rawimage_pixelsnum]=extracting_information(All_Intrainfor,Difclass_pixelsnumall,gray_value);
 % output:
 %  basicunit_infor(����10���ֶΣ���СΪ1*length(gray_value)�Ľṹ�壬����Ϊ���ֵ)����¼ÿ����������������Ϣ
 %  Rawimage_pixelsnum(���������ֶ�raw_imageid��pixels_num,��СΪ1*length(gray_value)�Ľṹ�壬����Ϊ���ֵ)����¼ÿ������Ӧ����������
%upbasicunit_infor=basicunit_infor;
upbasicunit_infor = rmfield(basicunit_infor, {'allcurclass_centroid', 'allcurclass_SameclassDis', 'allcurclass_Boundaryjud', 'allcurclass_mindis', 'allcurclass_minbox'});
%% upbasicunit_infor����¼ÿ������Ӧ��basic unit���� 
% ���ã���basicunit_infor�������С����s*s��С�Ľ��ж��δ�����֤���е�basic unit��s*s��
%       ������ȡ�����µ�basic unit��Ϣ���ϵ�upbasicunit_infor�ṹ����
%minbox_diffclass=cell(length(gray_value),1);
% �����ֲ��������洢���
delete_irr_class=setdiff(gray_value,irr_class);
local_upbasicunit_infor = repmat(struct(), length(delete_irr_class), 1);
local_minbox_diffclass = cell(length(delete_irr_class), 1);
parfor k1_d = 1:length(delete_irr_class)
    k1=delete_irr_class(k1_d);
    % Step 1: Get minimum size information for the basic unit of the current class
    allboundbox = basicunit_infor(k1).allcurclass_BoundingBox;
    allboundbox_limit = find(allboundbox(:,3) >= 2^5 & allboundbox(:,4) >= 2^5);
    allboundboxlimit = allboundbox(allboundbox_limit, 3:4);
    allboundboxlimit_s = allboundboxlimit(:, 1) .* allboundboxlimit(:, 2);
    basic_windows = allboundboxlimit(allboundboxlimit_s == min(allboundboxlimit_s), :);
    local_minbox_diffclass{k1_d} = basic_windows;
    
    % Step 2: Get all basic unit sizes that exceed s*s
    original_allid = basicunit_infor(k1).alloriginal_id;
    original_allsparity = basicunit_infor(k1).allcurclass_sparity;
    allbasic_boxinfor = [basicunit_infor(k1).allcurclass_BoundingBox];
    statsbasic_id = find(allbasic_boxinfor(:, 3) > s | allbasic_boxinfor(:, 4) > s);
    % �����ǰ k1_d �� statsbasic_id ֵ
    fprintf('Current k1_d: %d, statsbasic_id: %s\n', k1_d, mat2str(length(statsbasic_id)));
    
    % Step 3: Generate optimized size combinations
    row_allpos = unique([(1:fix(s / basic_windows(1))) * basic_windows(1), s]);
    col_allpos = unique([(1:fix(s / basic_windows(2))) * basic_windows(2), s]);
    combinations = combvec(row_allpos, col_allpos)';
    products = prod(combinations, 2);
    differences = abs(combinations(:, 1) - combinations(:, 2));
    [~, sorting_index] = sortrows([products, -differences]);
    combinations_sorted = combinations(sorting_index, :);
    
    % Initialize variables to store the updated basic unit information
    updata_basicoriginalid = [];
    updata_basicarea = [];
    updata_basicboundingbox = zeros(0, 4);  % Preallocate with correct column size
    updata_basicsparity = [];
    updata_basiccountshistogram = zeros(0, 256);  % Preallocate with correct column size

    % Inner loop for statsbasic_id elements
    for k2 = 1:length(statsbasic_id)
        tic;
        originalid = original_allid(statsbasic_id(k2));
        inibasic_box = ceil(allbasic_boxinfor(statsbasic_id(k2), :) - [0 0 1 1]);
        curimage = imread([imagepath labmicrDir(originalid).name]);
        curlabel = imread([labelpath labmicrDir(originalid).name]);
        curbasicimage = imcrop(curimage, inibasic_box);
        curbasiclabel = imcrop(curlabel, inibasic_box);
        curbasiclabel(curbasiclabel ~= k1) = 0;
        curbasiclabel_copy=curbasiclabel;
        [row_p, col_p] = size(curbasiclabel);
        inibasic_sparity = sum((curbasiclabel(:) == k1)) / (row_p * col_p);

        % Filter combinations based on current dimensions
        filtered_combinations = combinations_sorted;
        if row_p <= s && col_p > s
            stats_comsort = find(combinations_sorted(:, 1) <= row_p);
            filtered_combinations = combinations_sorted(stats_comsort, :);
        end
        if col_p <= s && row_p > s
            stats_comsort = find(combinations_sorted(:, 2) <= col_p);
            filtered_combinations = combinations_sorted(stats_comsort, :);
        end

        % Evaluate each combination
        for k3 = 1:size(filtered_combinations, 1)
            k_com1 = size(filtered_combinations, 1) + 1 - k3;
            basic_row = filtered_combinations(k_com1, 1);
            basic_column = filtered_combinations(k_com1, 2);
            n1 = basic_row - 1;
            n2 = basic_column - 1;                        
            h_row = round(basic_row / 8 * 7);
            h_column = round(basic_column / 8 * 7);
            temp_k = k1 * ones(basic_row, basic_column);
            h_rowallpos = unique([1:h_row:row_p - n1, max(row_p - s + 1, 1)]);
            h_colallpos = unique([1:h_column:col_p - n2, max(col_p - s + 1, 1)]);
            
            % Nested loops for position
            for i_row = 1:length(h_rowallpos)
                for j_col = 1:length(h_colallpos)
                    i = h_rowallpos(i_row);
                    j = h_colallpos(j_col);
                    diff = abs(double(curbasiclabel(i:i + n1, j:j + n2)) - temp_k);

                    % Check if this region satisfies sparsity criteria
                    if sum(diff(:) == 0) >= basic_row * basic_column * inibasic_sparity
                        points_tooriginal = inibasic_box(1:2) + [j - 1, i - 1];
                        curbasic_boundingbox = [points_tooriginal, n2 , n1 ];
                        curbasic_sparity = sum(diff(:) == 0) / (basic_row * basic_column);
                        curbasic_orignalid = originalid;
                        curbasic_area = sum(diff(:) == 0);
                        currenregion_image = imcrop(curbasicimage, [j, i, n2 , n1 ]);
                        currenregion_label = imcrop(curbasiclabel_copy, [j, i, n2, n1]);
                        currenregion_label_logits = currenregion_label == k1;
                        current_onlyoneclassimage = currenregion_image(currenregion_label_logits);
                        [countshistogram, ~] = histcounts(current_onlyoneclassimage, 256);
                        curbasiclabel(i:i+n1/8*7,j:j+n2/8*7)=0;
%                         strr=strcat('image_',num2str(i),num2str(j),'.png');
%                         imwrite(uint8(currenregion_image),['D:\A_userfile\lly\wrong_infor\image\' strr])
                        % Store basic unit info
                        updata_basicoriginalid = [updata_basicoriginalid, curbasic_orignalid];
                        updata_basicarea = [updata_basicarea, curbasic_area];
                        updata_basicboundingbox = [updata_basicboundingbox; curbasic_boundingbox];
                        updata_basicsparity = [updata_basicsparity, curbasic_sparity];
                        updata_basiccountshistogram = [updata_basiccountshistogram; countshistogram];
                    end
                end
            end
        end
        elapsed_time = toc;
        % Print the current k1_d, k2, and elapsed time
        fprintf('Current k1_d: %d, k2: %d, Elapsed time: %.2f seconds\n', k1_d, k2, elapsed_time);
    end

    % Consolidate updated info back to upbasicunit_infor
    upid = upbasicunit_infor(k1).alloriginal_id;
    uparea = upbasicunit_infor(k1).allcurclass_area;
    upBoundingBox = upbasicunit_infor(k1).allcurclass_BoundingBox;
    upsparity = upbasicunit_infor(k1).allcurclass_sparity;
    upcountshistogram = upbasicunit_infor(k1).allcurclass_countshistogram;
    
    % Remove outdated information
    upid(statsbasic_id) = [];
    uparea(statsbasic_id) = [];
    upBoundingBox(statsbasic_id, :) = [];
    upsparity(statsbasic_id) = [];
    upcountshistogram(statsbasic_id, :) = [];
    
    % �����º����Ϣ�洢���ֲ��ṹ�������
    local_upbasicunit_infor(k1_d).alloriginal_id =[upid updata_basicoriginalid];
    local_upbasicunit_infor(k1_d).allcurclass_area = [uparea updata_basicarea];
    local_upbasicunit_infor(k1_d).allcurclass_BoundingBox = [upBoundingBox;updata_basicboundingbox];
    local_upbasicunit_infor(k1_d).allcurclass_sparity = [upsparity updata_basicsparity];
    local_upbasicunit_infor(k1_d).allcurclass_countshistogram = [upcountshistogram;updata_basiccountshistogram];
end

minbox_diffclass = cell(length(gray_value), 1);
for k1_d = 1:length(delete_irr_class)
    k1 = delete_irr_class(k1_d);
    upbasicunit_infor(k1).alloriginal_id = local_upbasicunit_infor(k1_d).alloriginal_id;
    upbasicunit_infor(k1).allcurclass_area = local_upbasicunit_infor(k1_d).allcurclass_area;
    upbasicunit_infor(k1).allcurclass_BoundingBox = local_upbasicunit_infor(k1_d).allcurclass_BoundingBox;
    upbasicunit_infor(k1).allcurclass_sparity = local_upbasicunit_infor(k1_d).allcurclass_sparity;
    upbasicunit_infor(k1).allcurclass_countshistogram = local_upbasicunit_infor(k1_d).allcurclass_countshistogram;
    minbox_diffclass{k1} = local_minbox_diffclass{k1_d};
end

for k1_irr=1:length(irr_class)
    irr_classinfor=irr_class(k1_irr);
    irr_classcol_area=upbasicunit_infor(irr_classinfor).allcurclass_area;
    [~,irr_classcol_areaminidx]=min(irr_classcol_area);
    minbox=upbasicunit_infor(irr_classinfor).allcurclass_BoundingBox(irr_classcol_areaminidx,3:4);
    minbox_diffclass{irr_classinfor}=minbox;
end




%% Obtain the set of possible category combinations within an s��s region of the original training dataset
Curclass_infor = statsinfor_cropimage_bystep(labelpath,[s s]);
%% ���upbasic_unit�ϳߴ�С��10��������Ϣ
upbasicunit_infor_new=upbasicunit_infor;
basicunit_infor_new=basicunit_infor;
for kk=1:length(gray_value)
    curbasicunit=upbasicunit_infor(kk);
    curclassbounding=curbasicunit.allcurclass_BoundingBox;
    re_idx=curclassbounding(:,3)>10 & curclassbounding(:,4)>10;
    alloriginal_id=upbasicunit_infor(kk).alloriginal_id;
    upbasicunit_infor_new(kk).alloriginal_id = alloriginal_id(re_idx);
    upbasicunit_infor_new(kk).allcurclass_area = upbasicunit_infor(kk).allcurclass_area(re_idx);
    upbasicunit_infor_new(kk).allcurclass_BoundingBox = upbasicunit_infor(kk).allcurclass_BoundingBox(re_idx,:);
    upbasicunit_infor_new(kk).allcurclass_sparity = upbasicunit_infor(kk).allcurclass_sparity(re_idx);
    upbasicunit_infor_new(kk).allcurclass_countshistogram = upbasicunit_infor(kk).allcurclass_countshistogram(re_idx,:);
    if ismember(kk,irr_class)
        basicunit_infor_new(kk).alloriginal_id = alloriginal_id(re_idx);
        basicunit_infor_new(kk).allcurclass_area = upbasicunit_infor(kk).allcurclass_area(re_idx);
        basicunit_infor_new(kk).allcurclass_centroid = basicunit_infor_new(kk).allcurclass_centroid(re_idx);
        basicunit_infor_new(kk).allcurclass_BoundingBox = upbasicunit_infor(kk).allcurclass_BoundingBox(re_idx,:);
        basicunit_infor_new(kk).allcurclass_sparity = upbasicunit_infor(kk).allcurclass_sparity(re_idx);
        basicunit_infor_new(kk).allcurclass_SameclassDis =  basicunit_infor_new(kk).allcurclass_SameclassDis(re_idx);
        basicunit_infor_new(kk).allcurclass_Boundaryjud = basicunit_infor_new(kk).allcurclass_Boundaryjud(re_idx);
        basicunit_infor_new(kk).allcurclass_countshistogram = upbasicunit_infor(kk).allcurclass_countshistogram(re_idx,:);
    end
end
basicunit_infor=basicunit_infor_new;
upbasicunit_infor=upbasicunit_infor_new;

%% simi_matrix��Ϣ��ȡ
% upbasicunit_infor(13).alloriginal_id=basicunit_infor(13).alloriginal_id;
% upbasicunit_infor(13).allcurclass_area=basicunit_infor(13).allcurclass_area;
% upbasicunit_infor(13).allcurclass_BoundingBox=basicunit_infor(13).allcurclass_BoundingBox;
% upbasicunit_infor(13).allcurclass_sparity=basicunit_infor(13).allcurclass_sparity;
% upbasicunit_infor(13).allcurclass_countshistogram=basicunit_infor(13).allcurclass_countshistogram;
for k2=1:length(gray_value)
    currentclass_his=upbasicunit_infor(k2).allcurclass_countshistogram;
    currentclass_his_nor=currentclass_his./sum(currentclass_his,2);
    otherclass_value=setdiff(gray_value,k2);
    for k3=1:length(otherclass_value)
        other_class=otherclass_value(k3);
        otherclass_his=upbasicunit_infor(other_class).allcurclass_countshistogram;
        otherclass_his_nor=otherclass_his./sum(otherclass_his,2);
        similarity_matrix = sqrt(currentclass_his_nor * otherclass_his_nor');
        % Ϊ�ļ�������Ψһ�ı�ʶ����ȷ�����й����в����ͻ
        save_similar_infor=strcat(savepath,'similar_infor\');
        strr = strcat('similarity_matrix_', num2str(k2), '_', num2str(other_class), '.mat');
        if ~exist(save_similar_infor, 'dir')
            % ���Ŀ¼�����ڣ��򴴽���
            mkdir(save_similar_infor);
        end
        file_path = fullfile(save_similar_infor, strr);
        % ʹ�� '-v7.3' �汾�������
        save(file_path, 'similarity_matrix', '-v7.3');
    end
end

save([savepath 'Difclass_pixelsnumall.mat'],'Difclass_pixelsnumall')
save([savepath 'All_Intrainfor.mat'],'All_Intrainfor')
save([savepath 'basicunit_infor.mat'],'basicunit_infor')
save([savepath 'upbasicunit_infor.mat'],'upbasicunit_infor')
save([savepath 'Rawimage_pixelsnum.mat'],'Rawimage_pixelsnum')
save([savepath 'Relationship.mat'],'Relationship')
save([savepath 'minbox_diffclass.mat'],'minbox_diffclass')
save([savepath 'Curclass_infor.mat'],'Curclass_infor')


