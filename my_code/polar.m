path(path,'/Users/artsyinc/Documents/fundus/scripts/logsample')
cd('/Users/artsyinc/Documents/fundus/data/sample')

imy = imread('16_left.jpeg');
imshow(logsample(imy, 100,2000, floor(size(imy,2)/2), floor(size(imy,1)/2),1000,1000))
% problem with this is that every image is distorted differently depending on
% what is in the center (macula is not always centered)

% we can also set the center as something standard, like the macula (to blow up
% defects that are close to it)
om = logsample(imy, 100,2000, 1463,1287,1000,1000);
% and then re-arrange until the optic nerve is in the ceneter
divy = 450;
imshow([om((divy+1):1000,:,:);om(1:divy,:,:)]);

% overall doesn't seem worth it since it introduces a form of distortion
% and will have a different black border for every image
