#!/bin/shd

python3 ./feature_ranking.py --model 'vgg16' --fitness_function 'miou'--rank_features 'True' --create_training_set 'False' --train_test_classifier 'False'

python3 ./feature_ranking.py --model 'vgg16' --fitness_function 'importance'--rank_features True --create_training_set False --train_test_classifier False

#python3 ./feature_ranking.py --model 'vgg16' --fitness_function 'miou'--rank_features False --create_training_set True --train_test_classifier 'False

#python3 ./feature_ranking.py --model 'vgg16' --fitness_function 'importance'--rank_features False --create_training_set True --train_test_classifier False

#python3 ./feature_ranking.py --model 'vgg16' --fitness_function 'miou'--rank_features 'False --create_training_set False --train_test_classifier True

#python3 ./feature_ranking.py --model 'vgg16' --fitness_function 'importance'--rank_features False --create_training_set False --train_test_classifier True
