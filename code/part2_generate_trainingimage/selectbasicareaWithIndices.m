function [selected_values, selected_indices] = selectbasicareaWithIndices(array_area, area_sum, selectednum)
 % array_area=recurclassbasic_allarea; area_sum=curclass_area;selectednum=curfill_areanum
% Initialize arrays to store selected values and their indices:
selected_values = []; 
selected_indices_sort =  [];
[array_area_sort,array_areasort_indices]=sort(array_area);
if length(array_area)<selectednum
    selectednum=length(array_area);
end

if selectednum==1 || area_sum<array_area_sort(1)
    [~,mindif_id]=min(abs(array_area-repmat(area_sum,1,length(array_area))));
    selected_indices=mindif_id(1);
    selected_values=array_area(selected_indices);
else
    % Step 1: ��ʼ��
    remaining_sum = area_sum;   % ʣ��δ��ȡ���ܺ�

    % Step 2: ��ȡ������������ֵ
    for i =1:selectednum-1
        % ���㵱ǰ����ѡ������ֵ����Сֵ

        min_possible_value = min(setdiff(array_area,selected_values));
        max_possible_value = max(remaining_sum ,0);
        
        
        %�޳���ȡ����ֵ
        indices_mremain=setdiff(1:length(array_area_sort),selected_indices_sort);
        array_area_mremain= array_area_sort(indices_mremain);
        nearstsize_eait=min(abs(remaining_sum-array_area_mremain));
        if (min_possible_value>=max_possible_value | nearstsize_eait<remaining_sum*0.2) ...
                & i >=2
            break;
        end

        %ɸѡ�����ֵ����Ϣ
        stats_indices = array_area_mremain>=min_possible_value & array_area_mremain<=max_possible_value;
        mredmain_arrayindice = indices_mremain(stats_indices);
        mredmain_arrayarea = array_area_mremain(stats_indices);

        indices = randperm(length(mredmain_arrayarea),1);
        %selected_indices = mredmain_arrayindice(indices);
        selected_nums = mredmain_arrayarea(indices);

        selected_values = [selected_values selected_nums];
        selected_indices_sort = [selected_indices_sort mredmain_arrayindice(indices)];
        remaining_sum=area_sum-sum(selected_values);
    end

    flag=1;
    while true
        %��ȡ������ֵ
        last_areanum = 1;
        last_areasum_upper = area_sum-sum(selected_values);
        last_areasum_lower = max(0.8 * area_sum-sum(selected_values),0);
        %�޳���ȡ����ֵ
        indices_remain=setdiff(1:length(array_area_sort),selected_indices_sort);
        array_area_remain= array_area_sort(indices_remain);
        %�޳�������С��ֵ
        last_statsareaarray = array_area_remain<=last_areasum_upper & array_area_remain>=last_areasum_lower;
        last_arrayindice = indices_remain(last_statsareaarray);
        last_arrayarea = array_area_remain(last_statsareaarray);
        if isempty(last_arrayarea)
            [~,arraysum_nearsetid]=min(abs(array_area_remain+sum(selected_values)-area_sum));
            selected_indices_sort=[selected_indices_sort indices_remain(arraysum_nearsetid)];
            selected_values = [selected_values array_area_remain(arraysum_nearsetid)];
            flag=2;
        else
           clast_indices = randperm(length(last_arrayarea), last_areanum);
           clast_selectednum = last_arrayarea(clast_indices);
           %selected_nums = array_area(indices);
            if clast_selectednum+sum(selected_values) > 0.8*area_sum && clast_selectednum+sum(selected_values) <= area_sum
                % If within range, store selected values and their indices
                selected_values  = [selected_values clast_selectednum];
                selected_indices_sort = [selected_indices_sort last_arrayindice(clast_indices)];
                flag=2;
            end  
        end 
        if flag==2
            break
        end
    end
    selected_indices=array_areasort_indices(selected_indices_sort);
end
        
%         % Randomly select three values until their sum is within [a, 1.1*a]
%         flag=1;
%         while true
%             % Randomly choose three indices without replacement
%             if sum(array_area_sort(1:6))> area_sum
%                 array_area=array_area-1;
%             end
%             indices = randperm(length(array_area), selectednum);
%             selected_nums = array_area(indices);
% 
%             % Check if the sum of selected numbers is within [a, 1.1*a]
%             sum_selected = sum(selected_nums);
%             if sum_selected >= area_sum && sum_selected <= 1.1*area_sum
%                 % If within range, store selected values and their indices
%                 selected_values = selected_nums;
%                 selected_indices = indices;
%                 break;
%             end      
%         end
%     end
% end