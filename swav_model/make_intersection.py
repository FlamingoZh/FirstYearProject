import os
import cv2
import json
import csv
import random
import pickle
from sklearn.manifold import TSNE
import pathlib
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)

from main_swav import get_image_embedding

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return_list = list()
        for i, row in enumerate(reader):
            # print(row)
            if row[1] == '1':
                return_list.append((row[0], row[2], i - 1))
    return return_list

def loadGlove(filename,dumpname='embeddings_dict_840B.pkl'):
    if not pathlib.Path(dumpname).is_file():
        embeddings_dict = {}
        with open(filename, 'r', encoding="utf-8") as f:
            ii=0
            for line in f:
                ii+=1
                values = line.split()
                #print(values)
                word = values[0]
                try:
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector
                except:
                    print(values[0:5],ii)
        pickle.dump(embeddings_dict, open(dumpname, 'wb'))
    else:
        embeddings_dict=pickle.load(open(dumpname,'rb'))

    return embeddings_dict

def visualizingGlove(embeddings_dict,name,slice=False):
    plt.figure(figsize=(30, 30))
    tsne = TSNE(n_components=2, random_state=0)
    words = list(embeddings_dict.keys())
    vectors = [embeddings_dict[word] for word in words]
    if slice:
        Y = tsne.fit_transform(vectors[:1000])
    else:
        Y = tsne.fit_transform(vectors)
    plt.scatter(Y[:, 0], Y[:, 1])

    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    plt.show()

def get_cropped_image(dataset,randomseed=None,section=None):

    if randomseed:
        random.seed(randomseed)
    else:
        random.seed(0)

    if dataset=="openimage":
        #os.chdir('OIDv4_ToolKit-master/OID/Dataset/')
        path = os.walk('Dataset/train')
        i = 0
        for path, d, filelist in path:
            for filename in filelist:
                i += 1
                # sanity check
                # if i%2==0:
                #     if(filename[:-4]!=temp): print(filename,temp)
                # else:
                #     temp=filename[:-4]

                # print(filename)
                whole_path = os.path.join(path, filename)
                if i % 2 == 1:
                    img = cv2.imread(whole_path)
                else:
                    with open(whole_path, "r") as f:
                        data = f.read().split()[:5]
                    name = data[0] + '.jpg'
                    l, t, r, b = data[1:]
                    l = int(float(l))
                    t = int(float(t))
                    r = int(float(r))
                    b = int(float(b))
                    # print(l,t,r,b)
                    #print(name)
                    crop_img = img[t:b, l:r]
                    # cv2.imshow("cropped", crop_img)
                    # cv2.waitKey(0)
                    savepath = 'cropped_image/openimage/train/' + name
                    #print(os.getcwd())
                    #print(savepath)
                    cv2.imwrite(savepath, crop_img)
    elif dataset=="vrd_noun":
        annotations = json.load(open('json_dataset/annotations_train.json', 'rb'))
        # print(len(annotations.keys()))

        all_objects = json.load(open('json_dataset/objects.json', 'rb'))
        #print(sorted(all_objects))

        X_str_to_idx = dict()
        X_idx_to_str = dict()
        for i, string in enumerate(all_objects):
            X_str_to_idx[string] = i
            X_idx_to_str[i] = string

        jpg_names = set()
        if section:
            all_objects=all_objects[section[0]:section[1]]

            print("all_objects:",all_objects)
        for target_obj in all_objects:
            flag=0
            while flag == 0:
                key = random.sample(annotations.keys(), 1)[0]
                #print(key)
                #jpg_names.add(key)
                triplets = annotations[key]
                for triplet in triplets:
                    if flag: break
                    #predicate = triplet["predicate"]
                    object = triplet["object"]["category"]
                    subject = triplet["subject"]["category"]
                    if object==X_str_to_idx[target_obj]:
                        print("target_obj:", target_obj, X_str_to_idx[target_obj])
                        print("object:", object)
                        object_bbox = triplet["object"]["bbox"]
                        whole_path='sg_dataset/sg_train_images/'+key
                        img = cv2.imread(whole_path)
                        t,b,l,r=object_bbox
                        t=int(t)
                        b=int(b)
                        l=int(l)
                        r=int(r)
                        crop_img = img[t:b, l:r]
                        #cv2.imshow("cropped", crop_img)
                        #cv2.waitKey(0)
                        flag=1
                        savepath = 'cropped_image/vrd_noun/train/' + target_obj + '.jpg'
                        cv2.imwrite(savepath, crop_img)
                    elif subject==X_str_to_idx[target_obj]:
                        print("target_obj:", target_obj, X_str_to_idx[target_obj])
                        print("subject:", subject)
                        subject_bbox = triplet["subject"]["bbox"]
                        whole_path = 'sg_dataset/sg_train_images/' + key
                        img = cv2.imread(whole_path)
                        t, b, l, r = subject_bbox
                        t = int(t)
                        b = int(b)
                        l = int(l)
                        r = int(r)
                        crop_img = img[t:b, l:r]
                        #cv2.imshow("cropped", crop_img)
                        #cv2.waitKey(0)
                        flag=1
                        savepath = 'cropped_image/vrd_noun/train/' + target_obj + '.jpg'
                        cv2.imwrite(savepath, crop_img)
    elif dataset=="vrd_verb":

        all_predicates = json.load(open('json_dataset/predicates.json', 'rb'))
        all_predicates_sorted = sorted(all_predicates)
        # print(all_predicates)
        all_objects = json.load(open('json_dataset/objects.json', 'rb'))
        # print(sorted(all_objects))

        annotations = json.load(open('json_dataset/annotations_train.json', 'rb'))
        # print(annotations)

        obj_str_to_idx = dict()
        obj_idx_to_str = dict()
        for i, string in enumerate(all_objects):
            obj_str_to_idx[string] = i
            obj_idx_to_str[i] = string

        pred_str_to_idx = dict()
        pred_idx_to_str = dict()
        for i, string in enumerate(all_predicates):
            pred_str_to_idx[string] = i
            pred_idx_to_str[i] = string

        pred_str_to_idx_sorted = dict()
        pred_idx_to_str_sorted = dict()
        for i, string in enumerate(all_predicates_sorted):
            pred_str_to_idx_sorted[string] = i
            pred_idx_to_str_sorted[i] = string

        selected_predicates = read_csv('VRD_predicates.csv')
        # print(selected_predicates)

        #random.seed(2)
        # random.seed(1)
        jpg_names = set()
        for target_predicate_tuple in selected_predicates:
            (target_predicate, target_predicate_shorten, target_predicate_idx_sorted) = target_predicate_tuple
            # target_predicate_idx=pred_str_to_idx[pred_idx_to_str_sorted[target_predicate_idx_sorted]]
            flag = 0
            while flag == 0:
                key = random.sample(annotations.keys(), 1)[0]
                #print(key)
                # jpg_names.add(key)
                triplets = annotations[key]
                # random.shuffle(triplets)
                # print(triplets)
                for triplet in triplets:
                    # print("triplet:",triplet)
                    if flag: break
                    predicate_idx = triplet["predicate"]
                    object_idx = triplet["object"]["category"]
                    subject_idx = triplet["subject"]["category"]
                    object_bbox = triplet["object"]["bbox"]
                    subject_bbox = triplet["subject"]["bbox"]
                    # print(predicate,target_predicate_idx)
                    if predicate_idx == pred_str_to_idx[target_predicate]:
                        print(obj_idx_to_str[subject_idx], target_predicate, "(", target_predicate_shorten, ")",
                              obj_idx_to_str[object_idx])
                        # print("target_predicate:", target_predicate, X_str_to_idx[target_predicate])
                        whole_path = 'sg_dataset/sg_train_images/' + key
                        img = cv2.imread(whole_path)
                        t1, b1, l1, r1 = object_bbox
                        t2, b2, l2, r2 = subject_bbox
                        t = int(min(t1, t2))
                        b = int(max(b1, b2))
                        l = int(min(l1, l2))
                        r = int(max(r1, r2))
                        crop_img = img[t:b, l:r]
                        # cv2.imshow("cropped", crop_img)
                        # cv2.waitKey(0)
                        flag = 1
                        savepath = 'cropped_image/vrd_verb/train/' + target_predicate_shorten + '.jpg'
                        cv2.imwrite(savepath, crop_img)

def with_key(dataset,image_embedding):
    if dataset=="vrd_verb":
        selected_predicates = read_csv('VRD_predicates.csv')
        # print(len(ls),ls)

        ls2=image_embedding
        #ls2 = pickle.load(open("embedding_crop_predicate_vrd.pkl", "rb"))
        print(len(ls2))
        # print(ls2[0])
        embedding_crop_dict = dict()
        for i, j in zip(selected_predicates, ls2):
            (_, word, _) = i
            # print(word)
            embedding_crop_dict[word] = j
        # print(embedding_crop_dict)
        #pickle.dump(embedding_crop_dict, open("embedding_crop_predicate_with_key_vrd.pkl", "wb"))
        #visualizingGlove(embedding_crop_dict, name=None)
        return embedding_crop_dict
    elif dataset=="vrd_noun":
        all_objects=json.load(open('json_dataset/objects.json','rb'))
        ls=sorted(all_objects)
        print(len(ls),ls)

        ls2 = image_embedding
        #ls2=pickle.load(open("embedding_crop_vrd.pkl","rb"))
        print(len(ls2))
        #print(ls2[0])
        embedding_crop_dict=dict()
        for i,j in zip(ls,ls2):
            embedding_crop_dict[i]=j
        print(embedding_crop_dict)
        #pickle.dump(embedding_crop_dict,open("embedding_crop_with_key_vrd.pkl","wb"))
        #visualizingGlove(embedding_crop_dict,name=None)
        return embedding_crop_dict

def compose_intersection(image_data,embeddings_dict):
    keys = [i.lower() for i in image_data.keys()]
    # print(keys)



    sub_embeddings_list = list()
    for key in keys:
        if key not in embeddings_dict:
            print("error!", key)
            image_data.pop(key)
        else:
            sub_embeddings_list.append(embeddings_dict[key])

    image_data_2 = [i for i in image_data.values()]
    image_array = np.array(list(image_data_2))
    print(image_array.shape)

    word_array = np.array(sub_embeddings_list)
    print(word_array.shape)

    my_intersection_dict = dict()
    my_intersection_dict['z_0'] = image_array
    my_intersection_dict['z_1'] = word_array
    # TODO fix keys (pop error key)
    my_intersection_dict['vocab_intersect'] = keys
    # pickle.dump(my_intersection_dict,open("my_intersection_image_word_vrd.pkl",'wb'))
    #pickle.dump(my_intersection_dict, open("my_intersection_image_predicate_vrd_2.pkl", 'wb'))
    return my_intersection_dict





print("loading glove dict...")
glove_dict=loadGlove(filename='glove.840B.300d.txt',dumpname='embeddings_dict_840B.pkl')
print("finish loading glove dict!")

#dataset_name="vrd_verb"
dataset_name="vrd_noun"

# visual_emebedding=get_image_embedding('cropped_image/' + dataset_name)
#
# visual_embedding_with_key=with_key(dataset_name, visual_emebedding)
# #print(len(verb_embedding_with_key))
#
# my_intersection=compose_intersection(visual_embedding_with_key, glove_dict)

intersection_list=list()
for i in range(1):
    get_cropped_image(dataset_name,randomseed=i,section=[0,30])
    visual_emebedding = get_image_embedding('cropped_image/' + dataset_name)
    visual_embedding_with_key = with_key(dataset_name, visual_emebedding)
    my_intersection = compose_intersection(visual_embedding_with_key, glove_dict)
    intersection_list.append(my_intersection)
pickle.dump(intersection_list,open("pkl_file/vrd_noun_1_30items_2.pkl","wb"))