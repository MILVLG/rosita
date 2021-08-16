# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui
 # -------------------------------------------------------- 
 
import logging, json, os
import numpy as np
import torch.distributed as dist

from utils.vqa.vqa import VQA
from utils.vqa.vqaEval import VQAEval


def vqa_eval(__C, loader, net, valid=False):
    ans_ix_list = []
    text_id_list = []

    for step, data in enumerate(loader):
        proc_rank = __C.GRANK if __C.MP_STORAGE_SHR['eval'] else __C.LRANK
        if step % 100 == 0 and proc_rank == 0:
            logging.info('Evaluated [{:.2f} %]'.format(step / len(loader) * 100.))

        text_input_ids, text_mask, \
        imgfeat_input, imgfeat_mask, imgfeat_bbox, \
        qa_label, qa_loss_valid, text_id = data

        net_input = (text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)
        net_output = net(net_input)

        pooled_output, text_output, imgfeat_output, pred_mm_qa = net_output

        pred_np = pred_mm_qa.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)
        if pred_argmax.shape[0] != __C.EVAL_BATCH_SIZE:
            pred_argmax = np.pad(pred_argmax, (0, __C.EVAL_BATCH_SIZE - pred_argmax.shape[0]), mode='constant', constant_values=-1)
        ans_ix_list.append(pred_argmax)
        text_id_list.extend(text_id)

    # Files must in multi-node shared storage
    tmp_result_eval_file = os.path.join(__C.TMP_RESULT_PATH, ('result_' + __C.VERSION + '.json'))
    if valid:
        result_eval_file = os.path.join(__C.VAL_RESULT_PATH, ('result_' + __C.VERSION + '.json'))
    else:
        result_eval_file = os.path.join(__C.TEST_RESULT_PATH, ('result_' + __C.VERSION + '.json'))

    for rank_ in range(__C.WORLD_SIZE * __C.NODE_SIZE):
        dist.barrier()
        if __C.GRANK == rank_:
            qid_set = set()
            result = []
            if rank_ != 0:
                result = json.load(open(tmp_result_eval_file, 'r'))
                for result_ in result:
                    qid_set.add(result_['question_id'])
            ans_ix_list = np.array(ans_ix_list).reshape(-1)

            for step, text_id in enumerate(text_id_list):
                if int(text_id) not in qid_set and ans_ix_list[step] != -1:
                    result.append(
                        {'answer': loader.dataset.ix_to_ans[str(ans_ix_list[step])], 'question_id': int(text_id)})
            json.dump(result, open(tmp_result_eval_file, 'w+'))
        dist.barrier()

    proc_rank = __C.GRANK if __C.MP_STORAGE_SHR['eval'] else __C.LRANK
    if proc_rank == 0:
        assert len(json.load(open(tmp_result_eval_file, 'r'))) == len(loader.dataset)
        os.system('echo y | cp -r ' + tmp_result_eval_file + ' ' + result_eval_file)
    dist.barrier()

    proc_rank = __C.GRANK if __C.MP_STORAGE_SHR['eval'] else __C.LRANK
    if valid and proc_rank == 0:
        # create vqa object and vqaRes object
        eval_set = __C.DATASET_LIST['val'][0].split(':')[1]
        if eval_set in ['val']:
            ques_file_path = os.path.join(__C.DATASET_ROOTPATH, __C.DATASET_PATHMAP[
                'vqa'], 'v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json')
            ans_file_path = os.path.join(__C.DATASET_ROOTPATH, __C.DATASET_PATHMAP[
                'vqa'], 'v2_Annotations_Val_mscoco/v2_mscoco_val2014_annotations.json')
        elif eval_set in ['minival']:
            ques_file_path = os.path.join(__C.DATASET_ROOTPATH, __C.DATASET_PATHMAP['vqa'], 'ques_minival.json')
            ans_file_path = os.path.join(__C.DATASET_ROOTPATH, __C.DATASET_PATHMAP['vqa'], 'anno_minival.json')

        vqa = VQA(ans_file_path, ques_file_path)
        vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

        # create vqaEval object by taking vqa and vqaRes
        vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

        # evaluate results
        """
        If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
        By default it uses all the question ids in annotation file
        """
        vqaEval.evaluate()

        # loggin accuracies
        logging.info("\n")
        logging.info("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        logging.info("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            logging.info("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        logging.info("\n")

        logfile = open(os.path.join(__C.LOG_PATH, (__C.VERSION + '.txt')), 'a+')
        logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        for ansType in vqaEval.accuracy['perAnswerType']:
            logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        logfile.write("\n\n")
        logfile.close()
