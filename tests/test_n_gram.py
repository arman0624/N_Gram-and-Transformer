import pytest

from n_gram import NGramLM
from perplexity import evaluate_perplexity

DATA = [
    [0, 8, 143, 144, 145, 8, 146, 147, 75, 148, 149, 58, 11, 150, 151, 29, 141, 139, 58, 13],
    [0, 59, 74, 75, 76, 9, 77, 78, 79, 80, 81, 9, 82, 83, 9, 84, 85, 86, 87, 13],
    [0, 6, 63, 9, 64, 65, 66, 67, 1, 68, 69, 70, 71, 25, 72, 73, 13],
    [0, 202, 240, 202, 241, 242, 20, 243, 39, 244, 245, 59, 246, 61, 247, 248, 182, 249, 13],
    [0, 9, 133, 134, 135, 136, 39, 137, 138, 139, 11, 139, 140, 97, 21, 141, 20, 137, 97, 142, 13],
    [0, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 11, 57, 58, 59, 60, 61, 62, 58, 13],
    [0, 175, 2, 176, 177, 43, 178, 9, 179, 94, 180, 181, 182, 183, 184, 185, 38, 186, 187, 184, 188, 189, 13],
    [0, 14, 15, 16, 17, 18, 19, 9, 20, 21, 22, 23, 13],
    [0, 152, 94, 153, 154, 155, 38, 156, 157, 2, 158, 159, 160, 161, 13],
    [0, 99, 100, 6, 101, 102, 103, 6, 104, 105, 29, 106, 59, 107, 108, 109, 110, 94, 111, 19, 9, 112, 38, 9, 113, 1,
     100, 30, 114, 115, 116, 117, 118, 13],
    [0, 202, 203, 9, 204, 205, 206, 94, 207, 29, 81, 208, 59, 209, 210, 211, 212, 11, 213, 188, 214, 100, 215, 13],
    [0, 162, 25, 190, 191, 192, 9, 193, 1, 194, 195, 196, 197, 94, 178, 198, 199, 38, 200, 201, 13],
    [0, 9, 216, 118, 6, 30, 217, 9, 218, 94, 9, 219, 8, 220, 221, 29, 222, 223, 224, 225, 226, 13],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [0, 116, 119, 120, 1, 35, 121, 122, 123, 9, 124, 94, 125, 126, 9, 127, 128, 129, 130, 30, 114, 83, 131, 132, 29, 39,
     58, 13],
    [0, 35, 9, 36, 37, 38, 39, 40, 41, 42, 43, 44, 13],
    [0, 162, 63, 59, 163, 164, 94, 84, 165, 75, 166, 9, 167, 168, 169, 170, 9, 171, 172, 173, 174, 13],
    [0, 142, 227, 59, 228, 229, 38, 9, 230, 231, 94, 200, 232, 233, 29, 234, 235, 236, 9, 237, 223, 238, 239, 13],
    [0, 9, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 13],
    [0, 88, 89, 69, 90, 91, 92, 93, 56, 94, 52, 95, 96, 97, 98, 13],
]

SUBSETS = [
    [4, 19, 6, 5, 2],
    [8, 5, 10, 18, 12, 1, 14, 9],
    [15, 17, 4, 10, 9, 6, 12, 13, 5, 7, 18, 19, 14],
    [11, 6, 5, 19, 10, 4, 14, 1, 12, 8, 16, 15, 17, 0, 9, 3, 13, 18, 2, 7]
]

EXPECTED = {
    (1, 5): 177.899204,
    (1, 8): 161.944637,
    (1, 13): 160.453624,
    (1, 20): 163.029605,
    (2, 5): 1.899786,
    (2, 8): 2.077985,
    (2, 13): 2.084503,
    (2, 20): 2.054103,
    (3, 5): 1.163345,
    (3, 8): 1.16111,
    (3, 13): 1.17449,
    (3, 20): 1.172792,
}


@pytest.mark.ngram_1
def test_learning():
    for n in range(1, 4):
        model = NGramLM(n)
        model.learn(DATA)


@pytest.mark.ngram_2
def test_perplexity():
    for n in range(1, 4):
        model = NGramLM(n)
        model.learn(DATA)

        for subset in SUBSETS:
            subset_len = len(subset)
            eval_data = [DATA[idx] for idx in subset]
            perplexity = round(evaluate_perplexity(model, eval_data), 6)
            assert (EXPECTED[(n, subset_len)] == perplexity)
