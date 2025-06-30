function  [basicunit_infor,Rawimage_pixelsnum]=extracting_information(All_Intrainfor,Difclass_pixelsnum,gray_value)
% All_Intrainfor�����ϣ����ϸ�������basic unit����Ϣ������ȡ��Ӧ���basic unit��ϡ�����Ϣ

% Preallocate struct array
basicunit_infor(length(gray_value)).class = [];
basicunit_infor(length(gray_value)).alloriginal_id = [];
basicunit_infor(length(gray_value)).allcurclass_area=[];
basicunit_infor(length(gray_value)).allcurclass_centroid=[];
basicunit_infor(length(gray_value)).allcurclass_BoundingBox=[];
basicunit_infor(length(gray_value)).allcurclass_sparity=[];
basicunit_infor(length(gray_value)).allcurclass_SameclassDis=[];
basicunit_infor(length(gray_value)).allcurclass_Boundaryjud=[];
basicunit_infor(length(gray_value)).allcurclass_countshistogram=[];
basicunit_infor(length(gray_value)).allcurclass_mindis=[];
basicunit_infor(length(gray_value)).allcurclass_minbox=[];
Rawimage_pixelsnum(length(gray_value)).raw_imageid=[]; 
Rawimage_pixelsnum(length(gray_value)).pixels_num=[];  
for k1=1:length(gray_value)
    current_class=gray_value(k1); 
    % 1.Extract information of the current class's basic unit
    allcurclass_id= find([All_Intrainfor.Current_class]==current_class); %��ȡ��ǰ�����All_Intrainfor�ϵ�����
    alloriginal_id=[All_Intrainfor(allcurclass_id).Image_index]; %��ʾbasic unit��Ӧraw observed image������ֵ����
    allcurclass_area=[All_Intrainfor(allcurclass_id).Area];%��ʾbasic unit��Ӧ��areaֵ
    allcurclass_centroid=[All_Intrainfor(allcurclass_id).Centroid];%��ʾbasic unit��Ӧ�����ĵ�
    allcurclass_centroid=reshape(allcurclass_centroid,2,[])';
    allcurclass_BoundingBox=[All_Intrainfor(allcurclass_id).BoundingBox];%��ʾbasic unit��Ӧ��boundingbox
    allcurclass_BoundingBox=reshape(allcurclass_BoundingBox,4,[])';
    allcurclass_sparity=[All_Intrainfor(allcurclass_id).Sparity];%��ʾbasic unit��Ӧ��Sparity
    allcurclass_SameclassDis=[All_Intrainfor(allcurclass_id).Sameclass_Disvalue]; %��ʾͬ��basic unit����̾���
    allcurclass_Boundaryjud=[All_Intrainfor(allcurclass_id).Boundary_judgment]; %��ʾbasic unit�Ƕ�λ��Ӱ���ı߽߱�
    allcurclass_countshistogram=[All_Intrainfor(allcurclass_id).counts_histogram];%��ʾbasic unit��Ӧ�����ĵ�
    allcurclass_countshistogram=reshape(allcurclass_countshistogram,256,[])';

    sameclass_mindis=min(setdiff(allcurclass_SameclassDis,0));
    % ��ȡλ�ڷ��ıߵ���Сbasic unit area��Ϣ
    sameclass_Boundaryjud_id = find(allcurclass_Boundaryjud==2);
    sameclass_areacol=allcurclass_area(sameclass_Boundaryjud_id);   
    [~,sameclass_minarea_id] = min(sameclass_areacol);
    sameclass_minarea_id_toallarea=sameclass_Boundaryjud_id(sameclass_minarea_id);
    minarea_boxinfor=[allcurclass_BoundingBox(sameclass_minarea_id_toallarea,3) allcurclass_BoundingBox(sameclass_minarea_id_toallarea,4)];
    
    basicunit_infor(k1).class = current_class;
    basicunit_infor(k1).alloriginal_id = alloriginal_id;
    basicunit_infor(k1).allcurclass_area=allcurclass_area;
    basicunit_infor(k1).allcurclass_centroid=allcurclass_centroid;
    basicunit_infor(k1).allcurclass_BoundingBox=allcurclass_BoundingBox;
    basicunit_infor(k1).allcurclass_sparity=allcurclass_sparity;
    basicunit_infor(k1).allcurclass_SameclassDis=allcurclass_SameclassDis;
    basicunit_infor(k1).allcurclass_Boundaryjud=allcurclass_Boundaryjud;
    basicunit_infor(k1).allcurclass_countshistogram=allcurclass_countshistogram;
    basicunit_infor(k1).allcurclass_mindis=sameclass_mindis;
    basicunit_infor(k1).allcurclass_minbox=minarea_boxinfor;
    
    %Difclass_pixelsnum=struct('Image_index',[],'Current_class',[],'Pixelsnum',[]);
    % 2.Difclass_pixelsnum
    difclasspixels_id= find([Difclass_pixelsnum.Current_class]==current_class); %��ȡ��ǰ�����All_Intrainfor�ϵ�����
    raw_imageid=[Difclass_pixelsnum(difclasspixels_id).Image_index]; %��ʾbasic unit��Ӧraw observed image������ֵ����
    rawimage_difclassnum=[Difclass_pixelsnum(difclasspixels_id).Pixelsnum];%��ʾbasic unit��Ӧ��areaֵ
    Rawimage_pixelsnum(k1).raw_imageid=raw_imageid; 
    Rawimage_pixelsnum(k1).pixels_num=rawimage_difclassnum; 
end