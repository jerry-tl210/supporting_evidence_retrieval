def get_analysis(data):
    out_string = ""
    for d in data:
        for q in d['QUESTIONS']:
            if not q['SHINT_']:
                continue
            out_string += q['QID'] + '\n'
            out_string += q['QTEXT_CN'] + '\n'
            out_string += "sentences: \n"
            documentSentences = [s['text'] for s in d['SENTS']]
            if 'weight' in q.keys():
                for s_i, s in enumerate(documentSentences):
                    out_string += '\t' + '{}: {}, score:{} sentence_weight:{}\n'.\
                        format(s_i, s, q['scores'][s_i], q["weights"][s_i])
            else:
                for s_i, s in enumerate(documentSentences):
                    out_string += '\t' + '{}: {}, score:{}'.format(s_i, s, q['scores'][s_i])
            out_string += "answer:{}".format([ans['ATEXT_CN'] for ans in q['ANSWER']]) + '\n'
            out_string += "gold_SE:{}".format(q['SHINT_']) + '\n'
            out_string += "predict_SE:{}".format(q['sp']) + '\n'
            out_string += '\n'
        out_string += '=================================================\n'
    return out_string
