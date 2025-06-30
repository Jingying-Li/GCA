function croppedData = cropimage_bystep(inilabelpath,cropSize)

    % 获取所有 PNG 文件
    labelDir = dir(fullfile(inilabelpath, '*.png'));
    flag1=1;

    for k = 1:length(labelDir)
        % 读取每张影像
        lab1 = imread(fullfile(inilabelpath, labelDir(k).name));
        originalSize = size(lab1);
        [strideH, strideW] = computeStride(size(lab1), cropSize);

        % 计算行数和列数
        numRows = ceil((originalSize(1)-cropSize(1))/strideH);
        numCols = ceil((originalSize(2)-cropSize(2))/strideW);

        %initialization
        position_infor=zeros(numRows*numCols,4);
        class_infor=cell(numRows*numCols,1);
        flag=1;

        % 循环裁剪影像
        for i = 1:numRows
            for j = 1:numCols
                % 计算裁剪起始坐标
                startRow = min((i - 1) * strideH + 1,originalSize(1) - cropSize(1) + 1);
                startCol = min((j - 1) * strideW + 1,originalSize(2) - cropSize(2) + 1);

                % 设置裁剪终止坐标
                endRow = min(startRow + cropSize(1) - 1, originalSize(1));
                endCol = min(startCol + cropSize(2) - 1, originalSize(2));

                % 读取裁剪数据
                croppedDataImage = lab1(startRow:endRow, startCol:endCol);

                
                % 获取裁剪图像的类别信息（去除背景类0）
                croppedImageClass = setdiff(unique(croppedDataImage), 0);
                
                % 创建一个新结构体并存储信息
                position_infor(flag,:) = [startCol  startRow cropSize(1) - 1 cropSize(2) - 1];
                class_infor{flag} = croppedImageClass;
                flag=flag+1;
%                 newEntry.index=k;
%                 newEntry.position = [startRow, startCol];
%                 newEntry.classes = croppedImageClass;
%                 croppedData(end + 1) = newEntry; % 添加新结构体
            end
        end
        classinfor_uniqeuidx = cellfun(@(x) num2str(x'), class_infor, 'UniformOutput', false);

        % 找出唯一的组合
        [uniqueCombinations, ~, idx] = unique(classinfor_uniqeuidx, 'stable');
        for k1=1:size(uniqueCombinations,1)
            classinfor_rawimage=str2num(uniqueCombinations{k1});
            classidxinfor_rawimage=idx==k1;
            newEntry.index=k;
            newEntry.classes = classinfor_rawimage;
            newEntry.position = position_infor(classidxinfor_rawimage,:);
            croppedData(flag1) = newEntry; % 添加新结构体
            flag1=flag1+1;
        end    
    end
end
