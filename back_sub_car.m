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
imPath = 'car'; imExt = 'jpg';

% check if directory and files exist
if isdir(imPath) == 0
    error('USER ERROR : The image directory does not exist');
end

filearray = dir([imPath filesep '*.' imExt]); % get all files in the directory
NumImages = size(filearray,1); % get the number of images
if NumImages < 0
    error('No image in the directory');
end

disp('Loading image files from the video sequence, please be patient...');
% Get image parameters
imgname = [imPath filesep filearray(1).name]; % get image name
I = imread(imgname); % read the 1st image and pick its size
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);
ImSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);

for mean=1:NumImages
    imgname = [imPath filesep filearray(mean).name]; % get image name
    ImSeq(:,:,mean) = imread(imgname); % load image
end
disp(' ... OK!');


% BACKGROUND SUBTRACTION
%% ============= Part 1.1: Frame Differences ==============================

for i = 1:VIDEO_HEIGHT
    for j = 1:VIDEO_WIDTH
        % Taking median of frame be frame
        I = median(ImSeq(i,j,:));
        Background(i,j) = I;    % Background Image
    end
end

% Moving object in every frame with applying threshold.
for m = 1:size(ImSeq,3)
    
    M = ImSeq(:,:,m);               % Current Frame
    Foreground = M- Background;        % Foreground Image Without Threshold
    Foreground_Threshold = Foreground>25; % Foreground Image With Threshold
    subplot(221),imshow(Background,[]), title('Background Image');
    subplot(222),imshow(M,[]), title('Current Image');
    subplot(223),imshow(Foreground_Threshold,[]), title('Frame Difference');
    subplot(224),imshow(Foreground,[]), title('Object Detection W/O Threshold');
    drawnow;
end


%% =================== Part 2.1: Running average gaussian =================

% Initializing mean, variance and alpha to start the process
mean = ImSeq(:,:,1);                         % Initialising  Mean
sigma = 1000*ones(VIDEO_HEIGHT,VIDEO_WIDTH); % Initialising  Variance
alpha = 0.01;
final = zeros(VIDEO_HEIGHT,VIDEO_WIDTH);
figure;

for m = 1: size(ImSeq,3)
    M = ImSeq(:,:,m);   % Current Frame
    Foreground = M - Background;      % Foreground Image Without Threshold
    
    %Mean
    mean(:,:,m+1) = alpha*ImSeq(:,:,m)+(1-alpha)*mean(:,:,m);
    d = abs(ImSeq(:,:,m)-mean(:,:,m+1));
    %Variance
    sigma(:,:,m+1) = (d.^2)*alpha+(1-alpha)*(sigma(:,:,m));

    final = d>2*sqrt(sigma(:,:,m+1));   % Computed Foreground
    subplot(221),imshow(Background,[]), title('Background Image');
    subplot(222),imshow(M,[]), title('Current Image');
    subplot(223),imshow(final,[]), title('Running average gaussian');
    subplot(224),imshow(Foreground,[]), title('Object Detection W/O Threshold');
    drawnow;
end




%% ========= Part 2.2: Running average gaussian With Bounding Box =========
% DEFINE A BOUNDING BOX AROUND THE OBTAINED REGION
% you can draw the bounding box and show it on the image

% Initializing mean, variance and alpha to start the process
mean = ImSeq(:,:,1);                        % Initialising Mean
sigma = 100*ones(VIDEO_HEIGHT,VIDEO_WIDTH); % Initialising Variance
alpha = 0.01;
T = 2.5;
figure;

for m = 1: size(ImSeq,3)
    % Mean
    mean = alpha*ImSeq(:,:,m) + (1-alpha)*mean;
    % Variance
    sigma = d.^2.*alpha + (1-alpha).*sigma;
    d = abs(ImSeq(:,:,m) - mean);
    
    % Detect Foreground
    mask = abs(ImSeq(:,:,m)-mean) > T*sqrt(sigma);
    Final = ImSeq(:,:,m).*mask;
    
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
    imshow(ImSeq(:,:,m),[]);
    title('Moving object with bounding box');
    
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
for i = 1:size(ImSeq,3)
   x(:,i) = reshape(ImSeq(:,:,i), [], 1);
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

for i = 1:size(ImSeq,3)
    %Project an image
    Project = x(:,i);
    p = EIgen_Background' * (Project - mean);
    Sub_Image = EIgen_Background * p + mean;

    %Detect moving object
    Mask = abs(Sub_Image - Project) > T;
    Image_Reshaped = reshape(Project .* Mask, VIDEO_HEIGHT, VIDEO_WIDTH);
    
    %Get the bounding box
    Final_Image = im2bw(Image_Reshaped);
    % Morphological Operations
    Final_Image = imerode(Final_Image, strel('rectangle', [2 2]));
    Final_Image = imdilate(Final_Image, strel('rectangle', [5 5]));
    Bounding_Box = regionprops(Final_Image, 'BoundingBox', 'Area');
    Area_Box = cat(1, Bounding_Box.Area);
    if(Area_Box)
        [~,ind] = max(Area_Box);
        B_box = Bounding_Box(ind).BoundingBox;
    end
    
    %Show result
    imshow(ImSeq(:,:,i),[]), title('Moving object with bounding box in Eigen Background');
    if(Area_Box)
        hold on;
        rectangle('Position', B_box,'EdgeColor','r');
        hold off;
    end
    drawnow;
end




%% =================== Part 4: Highway sequence ===========================

% For the Highway Sequence, you can run file: bac_sub_Highway with the 
% same code but with different path. THere is no difference in the code
% except the Path. Rest all is the same.


% ++===================================================================++ %
% ++============================= THE END =============================++ %
% ++===================================================================++ %