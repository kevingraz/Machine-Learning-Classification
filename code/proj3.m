%  CSE474: Introduction to Machine Learning
%  
%  Project 3: Classification
%  
%  This file contains implementations of the logistic regression and neural
%  network algorithms. Each is used to classify hand written digits from
%  the MNIST data set. The MNIST database is a large database of
%  handwritten digits that is commonly used fortraining various image
%  processing systems. The database contains 60,00 training images and
%  10,000 testing images. 

%% Initialization
clear ; close all; clc

%  Load training and test data into MATLAB
train_images = loadMNISTImages('image_train');  
train_labels = loadMNISTLabels('label_train');  
test_images = loadMNISTImages('image_test');  
test_labels = loadMNISTLabels('label_test');

train_size = size(train_images);  
test_size = size(test_images);  

n = train_size(2);  
j = 300;  
d = train_size(1);  
k = 10;  

num_iterations = 10;  
scaling_factor = 0.05;  
learning_rate = 0.1; 
epsilon = 1e-6;

new_train_images = [train_images' ones(size(train_images',1),1)];
new_train_labels = zeros(size(train_labels,1),10);

% Logistic Regression Implimentation

for numbers = 0:9
    
    for i = 1:size(train_labels,1)
     
        if(train_labels(i) == numbers)
            new_train_labels(i, numbers + 1) = 1;
        
        end
    end
end

all_theta = zeros(size(new_train_images,2),10); 
for i = 1 : num_iterations
    
    fprintf('Iteration %d\n', i);
    beta = ones(size(train_labels));
    bias = ones(size(train_labels));
    set_cost = 1e6;
    distance = 10;
    while distance >= epsilon
        
        new_cost = set_cost;
        beta = new_train_images * all_theta(:,i);
        denumerator = 1 + exp(-beta);
        bias = 1 ./ denumerator;
        param1 = new_train_labels(:, i) .* log(bias);
        param2 = (1 - new_train_labels(:, i)) .* log(1 - bias);
        set_cost = (-1 / n) * sum(param1 + param2);      
        min = (1 / n) .* (bias - new_train_labels(:, i))' * new_train_images;
        all_theta(:, i) = all_theta(:, i) - learning_rate .* min';
        distance = new_cost - set_cost;
        disp(distance);
    end
end

Wlr = all_theta(1:784,:);
blr = all_theta(785,:);

fprintf('Logistic regression complete. Press enter to run neural network.\n');
pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Neural Network Implimentation
fprintf('Running Neural Network...');

weight1 = rand(d + 1, j);  
weight2 = rand(j + 1, k); 

forward_temp1 = zeros(d + 1, 1);  
forward_temp2 = zeros(j + 1, 1);  
forward_temp3 = zeros(j, 1);  
forward_temp4 = zeros(k, 1);  

backward_temp1 = zeros(d + 1, 1);  
backward_temp2 = zeros(j + 1, 1);  
backward_temp3 = zeros(k, 1);  

eye_k = eye(k); 

for t = 1 : num_iterations
    fprintf('Iteration %d\n', t);
    perm = randperm(n);   
    for p = 1 : n
               
        perm_index = perm(p);  
        forward_temp1 = [train_images(:, perm_index); 1];  
        forward_temp3 = nn_sigmoid(weight1' * forward_temp1, [scaling_factor, 0]);  
        forward_temp2 = [forward_temp3; 1];  
        forward_temp4 = nn_sigmoid(weight2' * forward_temp2, [scaling_factor, 0]);  
 
        backward_temp3 = forward_temp4 - eye_k(:, train_labels(perm_index) + 1);  
        backward_temp2 = weight2 * (backward_temp3 .* forward_temp4 .* (1 - forward_temp4));  
        backward_temp2 = backward_temp2(1 : j);  
        backward_temp1 = weight1 * backward_temp2;  
     
        weight2 = weight2 - learning_rate * forward_temp2 * (backward_temp3 .* forward_temp4 .* (1 - forward_temp4))';  
        weight1 = weight1 - learning_rate * forward_temp1 * (backward_temp2 .* forward_temp3 .* (1 - forward_temp3))';  
        
        if (mod(p, 5000)== 0)
            fprintf('\t%d\n', p);
        end
    end
   
    train_validation = zeros(num_iterations, 1); 
    train_validation(t) = 0;  
   
    for i = 1 : n  
        
        forward_temp1 = [train_images(:, i); 1];  
        forward_temp2 = [nn_sigmoid(weight1' * forward_temp1, [scaling_factor, 0]); 1];  
        forward_temp4 = nn_sigmoid(weight2' * forward_temp2, [scaling_factor, 0]);  
        [d, m] = max(forward_temp4);  
        if (m == train_labels(i) + 1)  
            train_validation(t) = train_validation(t) + 1;  
        end  
    end  
end  
 
 Wnn1 = weight1(1:784,:);
 Wnn2 = weight2(1:300,:);
 bnn1 = weight1(785,:);
 bnn2 = weight2(301,:);
 h = 'sigmoid';
 
 test_validation = 0;  
 
 for i = 1 : test_size(2)  
    
    forward_temp1 = [test_images(:, i); 1];  
    forward_temp2 = [nn_sigmoid(weight1' * forward_temp1, [scaling_factor, 0]); 1];  
    forward_temp4 = nn_sigmoid(weight2' * forward_temp2, [scaling_factor, 0]);  
    [d, m] = max(forward_temp4);  
    if (m == test_labels(i) + 1)  
        test_validation = test_validation + 1;  
    end  
 end  


