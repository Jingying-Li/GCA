function  State=updatasort_accordingconrelationship(choose_classcol_initial,chooseempairs,Relationship,basicunit_infor)
% Input��
%     choose_classcol_initial����ǰѡȡ����𼯺�
%     chooseempairs:�������
%     Relationship���������Ĺ�ϵ��Ϣ
%     basicunit_infor�����ж������Ϣ
%Output:
%     State���������ֶ�:
%       (1)�ֶ�classcol:��ʾ���ݰ�����ϵ������������𼯺�
%       (2)�ֶ�em_toclass����ʾ��choose_class����ֵ����ֵ
%       (3)�ֶ�choose_class����ʾѡȡ�����������֮һ
%% ��ȡ���ݰ�����ϵ�����������𼯺�
% ��ȡ��ǰ��𼯺϶�Ӧ�İ�����ϵ����
contain_relationship=Relationship.contain_relationship;
curclass_conrelation=contain_relationship(choose_classcol_initial,choose_classcol_initial);
% ��ȡ��ǰ���Ķ�������
objectnum_toclasscol=cell2mat(cellfun(@(x) length(x),{basicunit_infor(choose_classcol_initial).alloriginal_id},...
    'UniformOutput', false));
% ����ռ��
containpro_toclasscol=sum(curclass_conrelation,2)./objectnum_toclasscol';
[~,containidx]=sort(containpro_toclasscol,'descend');
choose_classcol=choose_classcol_initial(containidx);

% 2: ���¶���choose_class��em_toclass��ֵ(��Ϊ�������Ⱥ�˳��)
choose_class=chooseempairs(2);
em_toclass=chooseempairs(1);
if find(choose_classcol == em_toclass)< find(choose_classcol == choose_class)
    em_toclass=chooseempairs(2);
    choose_class=chooseempairs(1);
end
State.classcol=choose_classcol;
State.em_toclass=em_toclass;
State.choose_class=choose_class;



    









