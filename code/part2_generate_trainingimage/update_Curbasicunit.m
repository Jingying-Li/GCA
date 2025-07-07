function Curbasic_unit=update_Curbasicunit(Curbasic_unit_all) 
%% ���ʱ�����ǵ���һ��������Ϣ
 Curbasic_unit= Curbasic_unit_all;
for k2=1:size(Curbasic_unit_all,1)
    willfill_image=Curbasic_unit_all{k2,2}; % ����basic unit image
    willfill_label=Curbasic_unit_all{k2,1}; % ����basic unit label
    class_label=Curbasic_unit_all{k2,3};
    class_region = willfill_label == class_label;
    % ʹ��regionprops��ȡ��ǰ����������Ϣ
    stats_only = regionprops(class_region, 'BoundingBox', 'Area','PixelIdxList');
    idx_only=find(max([stats_only.Area])==[stats_only.Area]);
    cropped_mask = imcrop(class_region, stats_only(idx_only).BoundingBox);
    % 2. �������ͼ�������������������ֻ����������������ֻ�����������򣨸��������� PixelIdxList ��ɸ��
    %     �� PixelIdxList ����һ���ɾ��� binary mask��ֻ����ǰ����
    isolated_mask = false(size(class_region));
    isolated_mask(stats_only(idx_only).PixelIdxList) = true;
    %     Ȼ����ԭͼ��Ҳ�ü�����ɾ�������
    isolated_crop = imcrop(isolated_mask, stats_only(idx_only).BoundingBox);
    % 3 ��ȡ��Ӧ��Ӱ���Լ���ǩ����
    cropped_image = imcrop(willfill_image, stats_only(idx_only).BoundingBox);
    cropped_label = imcrop(willfill_label, stats_only(idx_only).BoundingBox);
    save_image=cropped_image.*uint8(repmat(isolated_crop,[1 1 3]));
    save_label=cropped_label.*uint8((isolated_crop));
    %figure,imshow(uint8(save_label*50))
    Curbasic_unit{k2,2}=save_image;
    Curbasic_unit{k2,1}=save_label;
end