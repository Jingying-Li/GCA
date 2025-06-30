function [fill_pixelpoints,fill_rotate]=pick_suitarea_allrotate(basic_label,label_micro)
willbasic_windows=size(label_micro);
[curbasic_row,curbasic_col]=size(basic_label);
crop_row=min(curbasic_row,willbasic_windows(1));
crop_col=min(curbasic_col,willbasic_windows(2));
% ��һ�������rotate 0
fill_cpixelpoints=pick_suitarea(basic_label,label_micro);
basic_label_c=imcrop(basic_label,[fill_cpixelpoints(2) fill_cpixelpoints(1)...
               crop_col-1 crop_row-1]);
validpixels_logits= basic_label_c~=0 & label_micro==0;
validpixels_num= sum(validpixels_logits(:));
           
% �ڶ��������rotate 90
label_micro_1=imrotate(label_micro,90);
fill_cpixelpoints_1=pick_suitarea(basic_label,label_micro_1);
basic_label_c1=imcrop(basic_label,[fill_cpixelpoints_1(2) fill_cpixelpoints_1(1)...
               crop_col-1 crop_row-1]);
validpixels_logits_1= basic_label_c1~=0 & label_micro_1==0;
validpixels_num_1= sum(validpixels_logits_1(:));
           
% �����������rotate 180
label_micro_2=imrotate(label_micro,180);
fill_cpixelpoints_2=pick_suitarea(basic_label,label_micro_2);
basic_label_c2=imcrop(basic_label,[fill_cpixelpoints_2(2) fill_cpixelpoints_2(1)...
               crop_col-1 crop_row-1]);
validpixels_logits_2= basic_label_c2~=0 & label_micro_2==0;
validpixels_num_2= sum(validpixels_logits_2(:));

% �����������rotate 270
label_micro_3=imrotate(label_micro,270);
fill_cpixelpoints_3=pick_suitarea(basic_label,label_micro_3);
basic_label_c3=imcrop(basic_label,[fill_cpixelpoints_3(2) fill_cpixelpoints_3(1)...
               crop_col-1 crop_row-1]);
validpixels_logits_3= basic_label_c3~=0 & label_micro_3==0;
validpixels_num_3= sum(validpixels_logits_3(:));

%% ȷ������ʵĵ��Լ���ת�Ƕ�
all_fillpixels=[fill_cpixelpoints;fill_cpixelpoints_1;fill_cpixelpoints_2;fill_cpixelpoints_3];
all_rotate=[0 90 180 270];
[~,maxvalid_idx]=max([validpixels_num validpixels_num_1 validpixels_num_2 validpixels_num_3]);
fill_pixelpoints=all_fillpixels(maxvalid_idx,:);
fill_rotate=all_rotate(maxvalid_idx);

