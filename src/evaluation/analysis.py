def get_analysis(data):
    out_string = ""
    for d in data:
        for q in d['QUESTIONS']:
            if not q['SHINT_']:
                continue
            out_string += q['QID'] + '\n'
            out_string += q['QTEXT_CN'] + '\n'
            out_string += "answer:{}".format([ans['ATEXT_CN'] for ans in q['ANSWER']]) + '\n'
            out_string += "gold_SE:{}".format(q['SHINT_']) + '\n'
            out_string += "predict_SE:{}".format(q['sp']) + '\n'
            out_string += '\n'
            all_set = set(q['SHINT_']) | set(q['sp'])
        
        out_string += '=================================================\n'
    return out_string
