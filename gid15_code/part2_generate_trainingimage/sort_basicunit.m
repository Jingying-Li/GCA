function basicunit_fillsort_end=sort_basicunit(Curbasic_unit,lastclass,s)
% ����Ŀ�ģ�������ϡ��ȡ���С���Լ�������ϵ����Ϣ��������. ����Ҫ���� spar ԽСԽ����,size ��ϢҲҪԽСԽ����,�� spar �����ȼ����� size
% ͬʱ,���ĳ������ size �ϴ󵫲����ڰ�����ϵ���������� size ��Ӱ��;��ϡ��Ȳ����ر�С������£���������������.


%1.��ȡ����basic unit��ϡ��ȣ���С��Ϣ�Լ�������ϵ��Ϣ
basicunit_class=[Curbasic_unit{:,3}];
basicunit_spar=[Curbasic_unit{:,4}];
basicunit_size=[Curbasic_unit{:,5}];
%��ȡlatclass��λ����Ϣ
lastclass_infor=find(basicunit_class==lastclass);
%�ж϶���������˳��
[~,basicunit_fillsort_last]=sort(basicunit_size(lastclass_infor),'ascend');
basicunit_fillsort_lastclass=lastclass_infor(basicunit_fillsort_last);
% ��һ������
basicunit_spar_normalized = (basicunit_spar - min(basicunit_spar)) / (max(basicunit_spar) - min(basicunit_spar));
basicunit_size_normalized = (basicunit_size - min(basicunit_size)) / (max(basicunit_size) - min(basicunit_size));

% ����Ȩ������
weight_spar = 0.6;  % spar ��Ȩ������
weight_size = 0.4;  % size ��Ȩ������
% �����Ȩ���ֵ
weighted_value = weight_spar * basicunit_spar_normalized - weight_size * basicunit_size_normalized;
%����weighted_value����
[~,basicunit_fillsort]=sort(weighted_value,'ascend');
if ~isempty(lastclass_infor)
    basicunit_fillsort_end=[basicunit_fillsort(~ismember(basicunit_fillsort,basicunit_fillsort_lastclass)) basicunit_fillsort_lastclass];
else
    basicunit_fillsort_end=basicunit_fillsort;    
end

