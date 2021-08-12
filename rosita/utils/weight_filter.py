import torch, copy, logging


'''
ans_vocab: [{'abc': 0, }, {'0': 'abc', }]
'''
def qa_cls_weight_filter(weight, bias, orin_ans_table, target_ans_table):
    assert weight.size(0) == len(orin_ans_table[0]) and bias.size(0) == len(orin_ans_table[0])
    target_weight = copy.deepcopy(weight)
    target_bias = copy.deepcopy(bias)
    target_weight = torch.cat((target_weight, torch.zeros((1, target_weight.size(1)), dtype=target_weight.dtype, device=target_weight.device)), dim=0)
    target_bias = torch.cat((target_bias, torch.zeros(1, dtype=target_bias.dtype, device=target_bias.device)), dim=0)
    assert target_weight.size(0) == (len(orin_ans_table[0]) + 1) and target_bias.size(0) == (len(orin_ans_table[0]) + 1)


    if orin_ans_table[0] == target_ans_table[0]:
        logging.info('qa answer vocab are same from loaded weight, skip filter qa head cls weight')
    else:
        logging.info('qa answer vocab are not same from loaded weight, start to filter qa head cls weight')
        loaded = 0
        unload = 0
        ans_map_list = []
        empty_pos = len(target_ans_table[0]) * [1]
        for i in range(len(target_ans_table[0])):
            ans = target_ans_table[1][str(i)]
            if ans in orin_ans_table[0]:
                orin_ix = orin_ans_table[0][ans]
                ans_map_list.append(orin_ix)
                loaded += 1
            else:
                ans_map_list.append(-1)
                unload += 1
                empty_pos[i] = 0

        target_weight = target_weight[ans_map_list]
        target_bias = target_bias[ans_map_list]
        assert target_weight.size(0) == len(target_ans_table[0]) and target_bias.size(0) == len(target_ans_table[0])

        for i in range(len(target_ans_table[0])):
            if empty_pos[i]:
                assert target_weight[i].sum() == weight[orin_ans_table[0][target_ans_table[1][str(i)]]].sum()
                assert target_bias[i].sum() == bias[orin_ans_table[0][target_ans_table[1][str(i)]]].sum()
            else:
                assert target_weight[i].sum() == 0
                assert target_bias[i].sum() == 0

        logging.info('[orin all: {}][target all: {}][loaded: {}][unload: {}]'.format(len(orin_ans_table[0]), len(target_ans_table[0]), loaded, unload))

    return target_weight, target_bias