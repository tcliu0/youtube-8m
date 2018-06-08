import os
import random
import cPickle as pickle
import matplotlib.pyplot as plt

if os.path.exists('vocabulary.pkl'):
    f = open('vocabulary.pkl')
    vocab = pickle.load(f)
    f.close()
else:
    f = open('vocabulary.csv')
    lines = f.readlines()[1:]
    f.close()
    vocab = {}
    for line in lines:
        line = line.split(',')
        vocab[line[0]] = line[3]
    f = open('vocabulary.pkl', 'wb')
    pickle.dump(vocab, f)
    f.close()

if os.path.exists('labels.pkl'):
    f = open('labels.pkl')
    labels = pickle.load(f)
    f.close()
else:
    f = open('labels.csv')
    lines = f.readlines()
    f.close()
    labels = {}
    for line in lines:
        line = line.split(',')
        labels[line[0]] = line[1].strip().split(' ')
    f = open('labels.pkl', 'wb')
    pickle.dump(labels, f)
    f.close()

if os.path.exists('predictions.pkl'):
    f = open('predictions.pkl')
    predictions = pickle.load(f)
    f.close()
else:
    f = open('predictions.csv')
    lines = f.readlines()
    f.close()
    predictions = {}
    for line in lines:
        line = line.split(',')
        pairs = line[1].strip().split(' ')
        pairs = zip(pairs[::2], pairs[1::2])
        predictions[line[0]] = pairs
    f = open('predictions.pkl', 'wb')
    pickle.dump(predictions, f)
    f.close()

k = 20
count = 0
videos = labels.keys()
# random.shuffle(videos)
# for video in videos:
#     top_k = [i for i, _ in predictions[video][:k]]
#     if all([i not in labels[video] for i in top_k]):
#         print video
#         vid = video
#         break
# while True:
#     vid, videos = videos[0], videos[1:]
#     if len(labels[vid]) == 1:
#         break
# print vid

#vid = '8V2EZ3U1nt4'
# x = [unicode(vocab[i], 'utf-8') for i, _ in predictions[vid]][::-1]
# c = ['green' if i in labels[vid] else 'red' for i, _ in predictions[vid]][::-1]
# y = [float(s) for _, s in predictions[vid]][::-1]

# print x[::-1]
# print [vocab[i] for i in labels[vid]]

# fig, ax = plt.subplots()
# plt.plot(y, x)
# plt.yticks(x, x)
# [t.set_color(i) for (i,t) in zip(c, ax.yaxis.get_ticklabels())]
# plt.subplots_adjust(left=0.4)
# plt.grid()
# plt.xlabel('Confidence')
# #plt.ylim([0.0, 1.0])
# plt.show()

counts = {}
for video in videos:
    pred = [i for i, _ in predictions[video]][:len(labels[video])]
    for label in labels[video]:
        if label not in counts:
            counts[label] = [0, 0]
        counts[label][0] += (1 if label in pred else 0)
        counts[label][1] += 1

recall = {}
for key in counts.keys():
    if counts[key][1]:
        recall[key] = 1.0 * counts[key][0] / counts[key][1]

f = open('recall.pkl', 'wb')
pickle.dump(recall, f)
f.close()        
    
