function Curbasic_unit=update_Curbasicunit(Curbasic_unit_all) 
%% 填充时仅考虑单独一个区域信息
 Curbasic_unit= Curbasic_unit_all;
for k2=1:size(Curbasic_unit_all,1)
    willfill_image=Curbasic_unit_all{k2,2}; % 填充的basic unit image
    willfill_label=Curbasic_unit_all{k2,1}; % 填充的basic unit label
    class_label=Curbasic_unit_all{k2,3};
    class_region = willfill_label == class_label;
    % 使用regionprops获取当前类别的区域信息
    stats_only = regionprops(class_region, 'BoundingBox', 'Area','PixelIdxList');
    idx_only=find(max([stats_only.Area])==[stats_only.Area]);
    cropped_mask = imcrop(class_region, stats_only(idx_only).BoundingBox);
    % 2. 将这个子图中所有区域分析出来，只保留面积最大的区域或只保留中心区域（更稳妥是用 PixelIdxList 来筛）
    %     用 PixelIdxList 创建一个干净的 binary mask，只含当前区域
    isolated_mask = false(size(class_region));
    isolated_mask(stats_only(idx_only).PixelIdxList) = true;
    %     然后在原图中也裁剪这个干净的区域
    isolated_crop = imcrop(isolated_mask, stats_only(idx_only).BoundingBox);
    % 3 获取对应的影像以及标签数据
    cropped_image = imcrop(willfill_image, stats_only(idx_only).BoundingBox);
    cropped_label = imcrop(willfill_label, stats_only(idx_only).BoundingBox);
    save_image=cropped_image.*uint8(repmat(isolated_crop,[1 1 3]));
    save_label=cropped_label.*uint8((isolated_crop));
    %figure,imshow(uint8(save_label*50))
    Curbasic_unit{k2,2}=save_image;
    Curbasic_unit{k2,1}=save_label;
end