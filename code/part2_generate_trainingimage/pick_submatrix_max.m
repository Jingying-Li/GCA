function [fill_pixelpoints]=pick_submatrix_max(label_micro,current_basiclabel)
%mask_label_micro��ʾ�ϸ���������λ��,�ж��������������ϵ��
%pick_submatrix(rotated_region_info,image_rotated_region_info,mask_label_micro_c,gray_currentfill,areawfill_infor,willfill_area);
window_size=size(current_basiclabel);%mask_label_micro�ϵ�ǰ����������ߴ���Ϣ
% ��ÿһ�н��д���columns ��������Ϊ���ڵ�����
row_submatrix = size(label_micro, 1) / window_size(1);
col_submatrix = size(label_micro, 2) / window_size(2);
row_subma=ceil(row_submatrix);
col_subma=ceil(col_submatrix);
%��¼��ǰ�п����ʼ�����Լ���Ч������ֵ
Cal_adjacentnum=[]; 
StartRow=[]; StartCol=[];
Rowcol=[];Colcol=[];
for row=1: row_subma
    for col = 1: col_subma
        currentl_window=zeros(window_size);
        if row<row_subma && col<col_subma
            startRow = (row - 1) *  window_size(1)  + 1;
            startCol = (col - 1) * window_size(2)+ 1;
            currentl_fillblock = label_micro(startRow:(startRow + window_size(1) - 1), startCol:(startCol + window_size(2) - 1));
            cal_valid_features= (currentl_fillblock==0);
%             cal_valid_features= (currentl_fillblock==curfill_class) & (mask_label_micro~=0);
            Cal_adjacentnum=[Cal_adjacentnum sum(cal_valid_features(:)==1)];
            StartRow=[StartRow startRow];
            StartCol=[StartCol startCol];
        elseif row>=row_subma && col<col_subma 
            startRow = size(label_micro,1)-window_size(1)+1;
            startCol = (col - 1) * window_size(2)+ 1;
            currentl_fillblock = label_micro(startRow:(startRow + window_size(1) - 1), startCol:(startCol + window_size(2) - 1));
            cal_valid_features= (currentl_fillblock==0);
            %cal_valid_features= (currentl_fillblock==curfill_class) & (mask_label_micro~=0);
            Cal_adjacentnum=[Cal_adjacentnum sum(cal_valid_features(:)==1)];
            StartRow=[StartRow startRow];
            StartCol=[StartCol startCol];
        elseif row<row_subma && col>=col_subma 
            startRow = (row - 1) *  window_size(1)  + 1;
            startCol = size(label_micro,2)-window_size(2)+1;
            currentl_fillblock = label_micro(startRow:(startRow + window_size(1) - 1), startCol:(startCol + window_size(2) - 1));
            cal_valid_features= (currentl_fillblock~=0);
            %cal_valid_features= (currentl_fillblock==curfill_class) & (mask_label_micro~=0);
            Cal_adjacentnum=[Cal_adjacentnum sum(cal_valid_features(:)==1)];
            StartRow=[StartRow startRow];
            StartCol=[StartCol startCol];
        else row>=row_subma && col>=col_subma
            startRow = size(label_micro,1)-window_size(1)+1;
            startCol = size(label_micro,2)-window_size(2)+1;
            currentl_fillblock = label_micro(startRow:(size(label_micro,1)-1), startCol:(size(label_micro,2)-1));
            currentl_window(1:size(currentl_fillblock,1),1:size(currentl_fillblock,2))=currentl_fillblock;
            currentl_fillblock=currentl_window;
            cal_valid_features= (currentl_fillblock==0);
            %cal_valid_features= (currentl_fillblock==curfill_class) & (mask_label_micro==1);
            Cal_adjacentnum=[Cal_adjacentnum sum(cal_valid_features(:)==1)];
            StartRow=[StartRow startRow];
            StartCol=[StartCol startCol];
        end
    end
end
%����ǰ��Ч������ֵ��ӽ�willfill_area��ֵ��Ϊ��ȡ�����basic unit
%[calfeature_maxvalue,calfeature_maxidx]=min(abs(Cal_adjacentnum-willfill_area));

fillchoose_idx=find(Cal_adjacentnum==max(Cal_adjacentnum));
%���ȡֵ
rand_idx=randperm(length(fillchoose_idx),1);
cropRow=StartRow(fillchoose_idx(rand_idx));
cropCol=StartCol(fillchoose_idx(rand_idx));
fill_pixelpoints=[cropRow cropCol];
% crop_rowlength=Rowcol(calfeature_maxidx(1));
% crop_collength=Colcol(calfeature_maxidx(1));
