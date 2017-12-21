%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUAL TRACKING
% ----------------------
% Background Subtraction
% ----------------
% Date: 17 October 2017
% Authors: Mohit Ahuja !!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;
%%%%% LOAD THE IMAGES
%=======================

% Give image directory and extension
Image_Path = 'highway/input'; imExt = 'jpg';
Ground_Truth_Path = 'highway/groundtruth'; gtExt = 'png';
% check if directory and files exist
if isdir(Image_Path) == 0
    error('USER ERROR : The image directory does not exist');
end

filearray = dir([Image_Path filesep '*.' imExt]); % get all files in the directory
Ground_Truth_Array = dir([Ground_Truth_Path filesep '*.' gtExt]);

NumImages = size(filearray,1); % get the number of images
if NumImages < 0
    error('No image in the directory');
end

disp('Loading image files from the video sequence, please be patient...');

% Get image parameters
Image_Name = [Image_Path filesep filearray(1).name]; % get image name
Ground_Truth_Name = [Ground_Truth_Path filesep Ground_Truth_Array(1).name];

I = imread(Image_Name); % read the 1st image and pick its size
GT = imread(Ground_Truth_Name);
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);

Image_Seq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
Ground_Truth_Seq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
for i=1:NumImages
    Image_Name = [Image_Path filesep filearray(i).name]; % get image name
    Image_Seq(:,:,i) = rgb2gray(imread(Image_Name)); % load image
    Ground_Truth_Name = [Ground_Truth_Path filesep Ground_Truth_Array(i).name]; % get image name
    Ground_Truth_Seq(:,:,i) = im2bw(imread(Ground_Truth_Name)); % load image
end
disp(' ... OK!');



% BACKGROUND SUBTRACTION
%% ============= Part 1.1: Frame Differences ==============================

for i = 1:VIDEO_HEIGHT
    for j = 1:VIDEO_WIDTH
        % Taking median of frame by frame
        I = median(Image_Seq(i,j,1:470));
        Background(i,j) = I;
    end
end

% Background Image
imshow(Background,[]);
title('Background Image');
figure;

True_Positive = 0;True_Negative = 0;False_Positive = 0;False_Negative = 0;
% Moving object in every frame with applying threshold.
for m = 470:size(Image_Seq,3)

    M = Image_Seq(:,:,m);
    Foreground = M- Background;
    Foreground = Foreground>20;   % Thresholding
    imshow(Foreground,[]);
    title('Moving object Detection with Frame Difference');

    % Morphological Operations
    Foreground = imfill(Foreground,'holes'); % Filling the holes in the image
    Foreground = imopen(Foreground, strel('rectangle', [2 2])); % Opening
    
    True_Positive = True_Positive + nnz(Foreground & Ground_Truth_Seq(:,:,m));
    True_Negative = True_Negative + nnz(~Foreground & ~Ground_Truth_Seq(:,:,m));
    False_Positive = False_Positive + nnz(Foreground & ~Ground_Truth_Seq(:,:,m));
    False_Negative = False_Negative + nnz(~Foreground & Ground_Truth_Seq(:,:,m));
end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('<------ For Frame Difference ------>');
precision = True_Positive/(True_Positive+False_Positive)
recall = True_Positive/(True_Positive+False_Negative)
F_Score = 2*precision*recall/(precision + recall)
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');



%% =================== Part 2.1: Running average gaussian =================

% Initializing mean, variance and alpha to start the process
mean = Image_Seq(:,:,1);                         % Initialising  Mean
sigma = 1000*ones(VIDEO_HEIGHT,VIDEO_WIDTH); % Initialising  Variance
alpha = 0.01;
final = zeros(VIDEO_HEIGHT,VIDEO_WIDTH);
figure;

True_Positive = 0; True_Negative = 0; False_Positive = 0; False_Negative = 0;
for m = 470: size(Image_Seq,3)
    %Mean
    mean = alpha*Image_Seq(:,:,m)+(1-alpha)*mean;
    %Variance
    d = abs(Image_Seq(:,:,m)-mean);
    sigma = (d.^2)*alpha+(1-alpha)*(sigma);

    final = d>2*sqrt(sigma);
    imshow(final,[]);
    title('Running average gaussian');

    final = imfill(final,'holes');
    final = imopen(final, strel('rectangle', [2 2]));
    True_Positive = True_Positive + nnz(final & Ground_Truth_Seq(:,:,m));
    True_Negative = True_Negative + nnz(~final & ~Ground_Truth_Seq(:,:,m));
    False_Positive = False_Positive + nnz(final & ~Ground_Truth_Seq(:,:,m));
    False_Negative = False_Negative + nnz(~final & Ground_Truth_Seq(:,:,m));
end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('<------ For Running Average Gaussian ------>');
precision = True_Positive/(True_Positive+False_Positive)
recall = True_Positive/(True_Positive+False_Negative)
F_Score = 2*precision*recall/(precision + recall)
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');



%% ========= Part 2.2: Running average gaussian With Bounding Box =========
% DEFINE A BOUNDING BOX AROUND THE OBTAINED REGION
% you can draw the bounding box and show it on the image

% Initializing mean, variance and alpha to start the process
mean = Image_Seq(:,:,1);                    % Initialising Mean
sigma = 100*ones(VIDEO_HEIGHT,VIDEO_WIDTH); % Initialising Variance
alpha = 0.01;
T = 2.5;
figure;

for m = 470: size(Image_Seq,3)
    % Mean
    mean = alpha*Image_Seq(:,:,m) + (1-alpha)*mean;
    % Variance
    sigma = d.^2.*alpha + (1-alpha).*sigma;
    d = abs(Image_Seq(:,:,m) - mean);
    
    % Detect Foreground
    mask = abs(Image_Seq(:,:,m)-mean) > T*sqrt(sigma);
    Final = Image_Seq(:,:,m).*mask;
    
    % Post processing on image
    Image = im2bw(Final);  % Converting into Black and White
    Image = imerode(Image, strel('rectangle', [2 2]));  % Eroding
    Image = imdilate(Image, strel('rectangle', [5 5])); % Dilating
    
    % Applying Bounding Box on image
    s = regionprops(Image, 'BoundingBox', 'Area');
    area = cat(1, s.Area);
    if(area)
        [~,ind] = max(area);
        bbox = s(ind).BoundingBox;
    end
    
    % Image show and then bounding box will detect the moving area
    imshow(Image_Seq(:,:,m),[]);
    title('Moving object with bounding box in Running Average Gaussian');
    
    % Bounding Box will follow the car
    if(area)
        hold on;
        rectangle('Position', bbox,'EdgeColor','r');
        hold off;
    end
    drawnow;
end




%% =================== Part 3: Eigen Background ===========================

% Initialising Variables
N = 30; k = 15; T = 25;
% Reshaping Images
for i = 1:size(Image_Seq,3)
   x(:,i) = reshape(Image_Seq(:,:,i), [], 1);
end

% The Mean Image
mean = 1/N*sum(x(:,1:N),2);
% Compute mean-normalized image vectors
Normalised_Mean = x - repmat(mean,1,size(x,2));
% SVD of the matrix X
[U, S, V] = svd( Normalised_Mean, 'econ');

% Keep the k principal components of U
EIgen_Background = U(:,1:k);
figure;

True_Positive = 0; True_Negative = 0; False_Positive = 0; False_Negative = 0;
for i = 470:size(Image_Seq,3)
    
    % Project an image
    Project = x(:,i);
    p = EIgen_Background' * (Project - mean);
    Sub_Image = EIgen_Background * p + mean;

    % Detect moving object
    Mask = abs(Sub_Image - Project) > T;
    Image_Reshaped = reshape(Project .* Mask, VIDEO_HEIGHT, VIDEO_WIDTH);
    
    Final_Image = im2bw(Image_Reshaped); % Binarizing the Image
     % Morphological operation
    Final_Image = imerode(Final_Image, strel('rectangle', [2 2])); % Erosion
    Final_Image = imdilate(Final_Image, strel('rectangle', [5 5])); % Dilation
    
    True_Positive = True_Positive + nnz(Final_Image & Ground_Truth_Seq(:,:,m));
    True_Negative = True_Negative + nnz(~Final_Image & ~Ground_Truth_Seq(:,:,m));
    False_Positive = False_Positive + nnz(Final_Image & ~Ground_Truth_Seq(:,:,m));
    False_Negative = False_Negative + nnz(~Final_Image & Ground_Truth_Seq(:,:,m));
    
    % Get the bounding box
    Bounding_Box = regionprops(Final_Image, 'BoundingBox', 'Area');
    Area_Box = cat(1, Bounding_Box.Area);
    if(Area_Box)
        [~,ind] = max(Area_Box);
        B_box = Bounding_Box(ind).BoundingBox;
    end
    
    % Show result
    imshow(Image_Seq(:,:,i),[]), title('Moving object with bounding box in Eigen Background)');
    if(Area_Box)
        hold on;
        rectangle('Position', B_box,'EdgeColor','r');
        hold off;
    end
    drawnow;
end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('<------ For Eigen Background ------>');
precision = True_Positive/(True_Positive+False_Positive)
recall = True_Positive/(True_Positive+False_Negative)
F_Score = 2*precision*recall/(precision + recall)
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');


% ++===================================================================++ %
% ++============================= THE END =============================++ %
% ++===================================================================++ %
