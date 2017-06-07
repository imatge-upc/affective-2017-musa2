# This scripts calculates the contributions of the anps i



import pickle
import numpy as np

import sys

# data_anp_net dictionary:
# {b"filename": (logits, label, contributions)}

parser = argparse.ArgumentParser('Calculate the contributions of the adjectives and nouns for each ANP')
# Inputs
parser.add_argument('--nouns_logit', help='Pickle with logits of the nouns net', required=True)
parser.add_argument('--adjective_logit', help='Pickle with logits of the adjectives net', required=True)
parser.add_argument('--anp_logits', help='Pickle with logits of the ANPNet', required=True)
parser.add_argument('--anp_labels', help='File with the ANP labels', required=True)
parser.add_argument('--noun_labels', help='File with the noun labels', required=True)
parser.add_argument('--adj_labels', help='File with the adjectives labels', required=True)

# Outputs
parser.add_argument('--acc_file', help='File to save the accuracy per class', required=True)
parser.add_argument('--ANR_file', help='File to save ANR per class', required=True)
parser.add_argument('--noun_contrib_file', help='File to save top 5 contributions of nouns', required=True)
parser.add_argument('--adj_contrib_file', help='File to save top 5 contributions of adjectives', required=True)


args = parser.parse_args()

data_noun_net = pickle.load( open( args.nouns_logit , "rb" ) )
data_adjective_net = pickle.load( open( args.adjective_logit, "rb" ) )
data_anp_net = pickle.load( open( args.anp_logits, "rb" ) )

with open(args.anp_labels) as f:
    anps = [anp.rstrip() for anp in f.readlines()]

# 167 nouns
with open(args.noun_labels) as f:
    nouns = [noun.rstrip() for noun in f.readlines()]

# 117 adjectives
with open(args.adj_labels) as f:
    adjectives = [adj.rstrip() for adj in f.readlines()]

def correct_top_k(array, k, label):
	return array.argsort()[-k:][::-1].__contains__(label)

ntop = 5

top_k = [1, 5, 10]

n_classes = 553
n_nouns = len(nouns)
n_adj = len(adjectives)

n_features = n_nouns + n_adj

num_image_per_class = np.zeros([n_classes, 3])
total_num_image_per_class = np.zeros(n_classes)

average_feature_contributions = np.zeros((n_features, n_classes))

accuracy_count = np.zeros(3)
accuracy_count_noun = np.zeros(3)
accuracy_count_ANP = np.zeros(3)

for key, value in data_anp_net.items():
	logits = value[0]
	label = value[1]

	total_num_image_per_class[label] += 1

	for i, k in enumerate(top_k):
		if correct_top_k(logits, k, label):
			accuracy_count[i] += 1
			num_image_per_class[label, i] += 1

			if i == 1:
				contributions = value[2]

                # Add the contribution of the highest logit
				average_feature_contributions[:,logits.argsort()[-1]] += contributions[:,0]

print("Accuracy top 1:" + str(accuracy_count[0]/len(data_anp_net)))
print("Accuracy top 5:" + str(accuracy_count[1]/len(data_anp_net)))
print("Accuracy top 10:" + str(accuracy_count[2]/len(data_anp_net)))

print(sum(num_image_per_class[:,0]==0))
print(sum(num_image_per_class[:,1]==0))
print(sum(num_image_per_class[:,2]==0))

with open(args.acc_file, 'w') as f:
	for i, item in enumerate(total_num_image_per_class):
		f.write( anps[i] + ";" + str(num_image_per_class[i, 0]/item) + ";" + str(num_image_per_class[i, 1]/item) + ";" + 
			str(num_image_per_class[i, 2]/item) + "\n")


anr_file = open(args.ANR_file, 'w')
top_5_noun_contrib_file = open(args.noun_contrib_file, 'w')
top_5_adj_contrib_file = open(args.adj_contrib_file, 'w')

for i in range(n_classes):
    # Sort feature contributions:
    if num_image_per_class[i, 1] == 0:
    	anr_file.write( anps[i] + ";" + str(-1) + "\n")
    	continue

    average_feature_contributions[:,i] /= num_image_per_class[i, 1] # normalize each class contribution by the number of images
    noun_max_feature = np.sort(average_feature_contributions[0:n_nouns,i])[::-1] # sort adjectives by contribution
    adj_max_feature = np.sort(average_feature_contributions[n_nouns:n_features, i])[::-1] # sort nouns by contribution
    noun_argmax_feature = np.argsort(average_feature_contributions[0:n_nouns, i])[::-1]
    adj_argmax_feature = np.argsort(average_feature_contributions[n_nouns:n_features, i])[::-1]

    # Compute adj vs noun contribution
    adj_contribution = sum(adj_max_feature) / n_adj
    noun_contribution = sum(noun_max_feature) / n_nouns

    norm = adj_contribution + noun_contribution
    adj_contribution = adj_contribution / norm
    noun_contribution = noun_contribution / norm

    ANR = adj_contribution/noun_contribution

    anr_file.write( anps[i] + ";" + str(ANR) + "\n")
    
    top_5_noun_contrib_file.write(anps[i] + ";")
    top_5_adj_contrib_file.write(anps[i] + ";")

    for noun in noun_argmax_feature[0:ntop]:
    	top_5_noun_contrib_file.write(";" + nouns[noun])

    for adj in adj_argmax_feature[0:ntop]:
    	top_5_adj_contrib_file.write(";" + adjectives[adj])

    top_5_noun_contrib_file.write("\n")
    top_5_adj_contrib_file.write("\n")

