

%clear; close all;
path = 'data/';
cate_id = {'plane', 'bike', 'chair', 'helicopter'};
cate_name = {'Plane', 'Bike', 'Chair', 'Helicopter'};

for ic = 4:length(cate_id)

    %%
    % AtlasNet2
    an_align = load([path 'atlasnet2_' cate_id{ic} '_aligned.mat'], 'distance');
    % Kim12
    kim12_rot = load([path 'kim12_' cate_id{ic} '_rot.mat'], 'distance');
    % Kim13
    kim13_rot = load([path 'kim13_' cate_id{ic} '_rot.mat'], 'distance');
    % lmvcnn
    lmvcnn_rot = load([path 'lmvcnn_' cate_id{ic} '_rot.mat'], 'distance');
    % shapeunicode
    su_align = load([path 'shapeunicode_' cate_id{ic} '_aligned.mat'], 'distance');
    % Chen
    chen_align = load([path 'chen_' cate_id{ic} '_aligned.mat'], 'distance');
    chen_rot = load([path 'chen_' cate_id{ic} '_rot.mat'], 'distance');
    % Ours
    our_align = load([path 'ours_' cate_id{ic} '_aligned.mat'], 'distance');
    our_rot = load([path 'ours_' cate_id{ic} '_rot.mat'], 'distance');
    % theshold
    load([path 'chen_' cate_id{ic} '_aligned.mat'], 'threshold');
    
    %%
    set(0,'defaultaxesfontsize',24);
    set(0,'defaultaxesfontname','Times New Roman');
    set(0,'defaultaxesfontweight','bold');
    fig = figure('Position', [0 0 640*1.2 480*1.2]);
    plot(threshold, lmvcnn_rot.distance*100, '--m', 'LineWidth', 3), hold all;
    plot(threshold, su_align.distance*100, 'g', 'LineWidth', 3), hold all;
    plot(threshold, chen_align.distance*100, 'b', 'LineWidth', 3), hold all;
    plot(threshold, chen_rot.distance*100, '--b', 'LineWidth', 3), hold all;
    plot(threshold, kim12_rot.distance*100, '--k', 'LineWidth', 3), hold all;
    plot(threshold, kim13_rot.distance*100, '--c', 'LineWidth', 3), hold all;
    plot(threshold, an_align.distance*100, 'Color', [0.8 0.8 0], 'LineWidth', 3), hold all;
    plot(threshold, our_align.distance*100, 'r', 'LineWidth', 3), hold all;
    plot(threshold, our_rot.distance*100, '--r', 'LineWidth', 3), hold all;
    %
    xlabel('Euclidean Distance');
    ylabel('Correspondences (%)');
    axis([0, 0.25, 0, 100]);
    set(gca,'XTick',0:.05:.25);
    set(gca,'YTick',0:10:100);
    grid on;
    title(cate_name{ic});
    
    legend( 'LMVCNN(rotate)', 'ShapeUnicode', 'Chen et al.', 'Chen et al.(rotate)', 'Kim12(rotate)','Kim13(rotate)', 'AtlasNet', 'Ours', 'Ours(rotate)');
    pause;
end





