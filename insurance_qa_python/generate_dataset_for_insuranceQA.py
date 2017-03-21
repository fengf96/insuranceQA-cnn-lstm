import os
import pdb
import pickle


def load(file_name):
    return pickle.load(open(os.path.join(path, file_name), 'rb'))


path = "."

train = load("train")
vocab = load("vocabulary")
answers = load("answers")

with open("../insuranceQA/train", "w") as f:
    for q_and_a in train:
        question_words = [vocab[x] for x in q_and_a["question"]]
        if len(question_words) < 200:
            question_words += ["<a>"] * (200 - len(question_words))
        else:
            question_words = question_words[:200]

        for ans_id in q_and_a["answers"]:
            answer_words = [vocab[x] for x in answers[ans_id]]
            if len(answer_words) < 200:
                answer_words += ["<a>"] * (200 - len(answer_words))
            else:
                answer_words = answer_words[:200]
            f.write("1 qid:0 %s %s\n" % ("_".join(question_words), "_".join(answer_words)))

test1 = load("test1")
with open("../insuranceQA/test1", "w") as f:
    for i, q_and_a in enumerate(test1):
        question_words = [vocab[x] for x in q_and_a["question"]]
        if len(question_words) < 200:
            question_words += ["<a>"] * (200 - len(question_words))
        else:
            question_words = question_words[:200]


        for ans_id in q_and_a["good"]:
            good_answer_words = [vocab[x] for x in answers[ans_id]]
            if len(good_answer_words) < 200:
                good_answer_words += ["<a>"] * (200 - len(good_answer_words))
            else:
                good_answer_words = good_answer_words[:200]
            f.write("1 qid:%s %s %s\n" % (i, "_".join(question_words), "_".join(good_answer_words)))

        for ans_id in q_and_a["bad"]:
            bad_answer_words = [vocab[x] for x in answers[ans_id]]
            if len(bad_answer_words) < 200:
                bad_answer_words += ["<a>"] * (200 - len(bad_answer_words))
            else:
                bad_answer_words = bad_answer_words[:200]
                f.write("0 qid:%s %s %s\n" % (i, "_".join(question_words), "_".join(bad_answer_words)))

pdb.set_trace()
