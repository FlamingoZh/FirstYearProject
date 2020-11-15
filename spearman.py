from scipy.stats import spearmanr
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

def alignment_score(a, b, method='spearman'):
    """Return the alignment score between two similarity matrices.

    Assumes that matrix a is the smaller matrix and crops matrix b to
    be the same shape.
    """
    n_row = a.shape[0]
    b_cropped = b[0:n_row, :]
    b_cropped = b_cropped[:, 0:n_row]
    idx_upper = np.triu_indices(n_row, 1)

    if method == 'spearman':
        # Alignment score is the Spearman correlation coefficient.
        alignment_score, _ = spearmanr(a[idx_upper], b_cropped[idx_upper])
        # plt.hist(a[idx_upper])
        # plt.show()
    else:
        raise ValueError(
            "The requested method '{0}'' is not implemented.".format(method)
        )
    return alignment_score

def run_spearman(intersect_data,shuffle=False):
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    if shuffle:
        z_1=np.random.permutation(z_1)
    #vocab_intersect = intersect_data['vocab_intersect']

    #print("z_0 shape:", z_0.shape)
    #print("z_1 shape:", z_1.shape)
    sim_0 = cosine_similarity(z_0)
    sim_1 = cosine_similarity(z_1)
    #print("similarity matrix shape:", sim_0.shape, sim_1.shape)
    score = alignment_score(sim_0, sim_1)
    whole_similarity_matrix, _ = spearmanr(sim_0, sim_1,axis=None)
    #print("a_score:",whole_similarity_matrix)
    #print("spearman correlation:",score)
    return score


# data_origin = pickle.load(open("intersect\intersect_glove.840b-openimage.box.p", 'rb'))
# data_openimage = pickle.load(open("my_intersection_image_word.pkl", 'rb'))
# data_vrd = pickle.load(open("my_intersection_image_word_vrd.pkl", 'rb'))
# data_vrd_predicate=pickle.load(open("my_intersection_image_predicate_vrd.pkl", 'rb'))
# data_vrd_predicate_2=pickle.load(open("my_intersection_image_predicate_vrd_2.pkl", 'rb'))



#print(data_origin['vocab_intersect'])

# print("data from the original dataset:")
# print(run_spearman(data_origin))
# print("data from openimage")
# print(run_spearman(data_openimage))
# print("data from vrd (nouns):")
# print(run_spearman(data_vrd))
#
def multi_spearman(data,verbose=False):
    score_list = list()
    for i in range(len(data)):
        score=run_spearman(data[i])
        score_list.append(score)
        if verbose:
            print("sampling", i + 1)
            print("spearman correlation:", score)
    print("mean correlation:", np.mean(score_list))
    print("correlation variance:", np.var(score_list))

def mean_data(list_data):
    new_data=dict()
    print(list_data[0]["z_0"].shape)
    temp1=np.zeros((list_data[0]["z_0"].shape[0],list_data[0]["z_0"].shape[1]))
    temp2=np.zeros((list_data[0]["z_1"].shape[0],list_data[0]["z_1"].shape[1]))
    for i in list_data:
        temp1+=i["z_0"]
        temp2+=i["z_1"]
    temp1=temp1/len(list_data)
    temp2=temp2/len(list_data)
    new_data["z_0"]=temp1
    new_data["z_1"]=temp2
    #print(new_data)
    return new_data


vrd_verb=pickle.load(open("swav_model/pkl_file/vrd_verb_25.pkl","rb"))
print("data from vrd (verbs):")
multi_spearman(vrd_verb,verbose=False)

print("new_vrd_verb")
new_vrd_verb=mean_data(vrd_verb)
multi_spearman([new_vrd_verb],verbose=False)



vrd_noun=pickle.load(open("swav_model/pkl_file/vrd_noun_25.pkl","rb"))
vrd_noun1=pickle.load(open("swav_model/pkl_file/vrd_noun_25_30items.pkl","rb"))
vrd_noun2=pickle.load(open("swav_model/pkl_file/vrd_noun_25_30items_2.pkl","rb"))
print("\ndata from vrd (nouns):")
multi_spearman(vrd_noun,verbose=False)
multi_spearman(vrd_noun1,verbose=False)
multi_spearman(vrd_noun2,verbose=False)

print("new_vrd_noun")
new_vrd_noun=mean_data(vrd_noun1)
multi_spearman([new_vrd_noun],verbose=False)


def vstack(data1,data2):
    data=dict()
    data['z_0']=np.vstack((data1['z_0'],data2['z_0']))
    data['z_1'] = np.vstack((data1['z_1'], data2['z_1']))
    data['vocab_intersect']=data1['vocab_intersect']+data2['vocab_intersect']
    return data

merged=[vstack(vrd_noun1[i],vrd_verb[i]) for i in range(25)]
print("\nnouns + verbs:")
multi_spearman(merged,verbose=False)


merged=[vstack(vrd_noun1[i],vrd_noun2[i]) for i in range(25)]
print("\nnouns + nouns:")
multi_spearman(merged,verbose=False)


















# print("shuffled data from the original dataset:")
# run_spearman(data_origin,shuffle=True)
# print("shuffled data from openimage")
# run_spearman(data_openimage,shuffle=True)
# print("shuffled data from vrd (nouns):")
# run_spearman(data_vrd,shuffle=True)


# origin_z_0=data_origin['z_0']
# openimage_z_1=data_openimage['z_1']
#
# print(origin_z_0.shape,openimage_z_1.shape)