import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import numpy
import cv2
from itertools import combinations

# get medians from file
my_file = open("medians.txt", "r")
content_list = my_file.readlines()
median_values = list(map(lambda x: float(x.replace('\n', '')), content_list))


class ImageEmbedder():

    def __init__(self):
        self.layer_output_size = 1024
        self.model_name = 'densenet'
        self.model = models.densenet121(pretrained=True)
        self.extraction_layer = self.model.features[-1]

        self.layer_output_size = self.model.classifier.in_features
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, imgs: [numpy.ndarray]):
        images_vectors = list(map(lambda x: cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC), imgs))
        a = [self.normalize(self.to_tensor(im)) for im in images_vectors]
        images = torch.stack(a)
        my_embedding = torch.zeros(len(imgs), self.layer_output_size, 7, 7)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        with torch.no_grad():
            h_x = self.model(images)
        h.remove()
        return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]


class TFVectorizer:
    def __init__(self):
        self.medians = median_values
        self.vec_len = 11
        self.random_positions = [303, 146, 606, 3, 289, 446, 512, 274, 744, 844, 240]

    def get_tf_string(self, embedding):
        vec = []
        for pos in self.random_positions:
            if embedding[pos] > self.medians[pos]:
                vec.append(1)
            else:
                vec.append(0)

        return "".join([str(i) for i in vec])


def one_comb(comb, vec):
    r = []
    for i in range(len(vec)):
        if i in comb:
            r.append(vec[i])
        else:
            r.append((vec[i] + 1) % 2)
    return r


def find_neighbours_for_combs(vec, combinations):
    res = list(map(lambda x: one_comb(x, vec), combinations))
    return res


def neighbours(tf_string, MUTATION_RATE=0.2, VEC_LEN=11):
    vec = [int(s) for s in tf_string]
    results = []
    mutations = int(VEC_LEN * MUTATION_RATE)
    for i in range(0, mutations + 1):
        combs = list(combinations(np.arange(0, VEC_LEN), VEC_LEN - i))
        results.extend(find_neighbours_for_combs(vec, combs))

    results = list(map(lambda x: "".join([str(i) for i in x]), results))
    return results
