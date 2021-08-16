# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui
 # -------------------------------------------------------- 
 
import os, json, logging, re
import torch.distributed as dist


class TextSegment:
    def __init__(self, __C, RUN_MODE):
        self.__C = __C
        self.RUN_MODE = RUN_MODE
        self.total_len = 0
        self.segment_path = self.get_segment_path()
        proc_rank = self.__C.GRANK if self.__C.MP_STORAGE_SHR['tmp'] else self.__C.LRANK
        self.synced_segment_text(proc_rank)

    def get_segment_path(self):
        identity = ''
        for dataset_name in self.__C.DATASET_LIST[self.RUN_MODE]:
            tset = dataset_name.split(':')[0]
            ttype = dataset_name.split(':')[1]
            identity += '-{}:{}'.format(tset, ttype)
        segment_path = self.__C.SEGMENT_PATH['files'] + identity + '/'
        os.system('mkdir -p ' + segment_path)
        logging.info('using text segment at: [{}]'.format(segment_path))

        return segment_path

    def segment_text(self):
        # Multi-task datasets pre-loading
        data_aggr = []
        for dataset_name in self.__C.DATASET_LIST[self.RUN_MODE]:
            tset = dataset_name.split(':')[0]
            ttype = dataset_name.split(':')[1]
            formatted_data = json.load(open(self.__C.DATASET_ROOTPATH_MAP[tset], 'r'))[ttype]
            data_aggr += formatted_data
            logging.info('[segment: {}] loading [{}] data: {}'.format(self.RUN_MODE, dataset_name, len(formatted_data)))

        self.total_len = len(data_aggr)

        if self.__C.RE_SEGMENT:
            for step in range(self.total_len):
                if step % 100000 == 0:
                    logging.info('[segment: {}] processing [{} | {}]'.format(self.RUN_MODE, step, self.total_len))

                json_filename = self.segment_path + self.RUN_MODE + '-' + str(step) + '.json'
                json_file = open(json_filename, 'w+')
                json.dump(data_aggr[step], json_file)
                json_file.close()

        del data_aggr

    def dict_to_plain(self, line_dict, split_key='%^&*', split_value='@!#$'):
        line_plain = ''
        for key in line_dict:
            value = line_dict[key]
            if key == 'text':
                value = re.sub(r"([.,'!?\"()*#:;])", '', value.strip()).replace('-', ' ').replace('/', ' ').replace(
                    '\n', ' ').replace('@', '').replace('%', ' percent')

            line_plain += '{}{}{}{}'.format(key, split_key, value, split_value)
        line_plain += '\n'

        return line_plain

    def plain_to_dict(self, line_plain, split_key='%^&*', split_value='@!#$'):
        line_dict = {}
        key_values = line_plain.strip().split(split_value)[:-1]
        for key_value in key_values:
            assert len(key_value.split(split_key)) == 2
            key, value = key_value.split(split_key)
            if key == 'multi_label' and value.startswith('['):
                value_list = self.str_to_list(value, type='str')
                line_dict[key] = value_list
            else:
                line_dict[key] = value

        return line_dict

    def str_to_list(self, string, type='str'):
        string_list = []
        if len(string) > 2:
            for string_split in string.strip().strip('[] ').split(','):
                if type in ['str']:
                    token = string_split.strip().strip("'")
                elif type in ['int']:
                    token = int(string_split.strip())

                string_list.append(token)

        return string_list

    def write_to_sync(self):
        sync_list = [str(self.total_len) + '\n']
        sync_file = open(self.__C.SEGMENT_PATH['sync'], 'w+')
        sync_file.writelines(sync_list)
        sync_file.close()

    def load_to_sync(self):
        sync_file = open(self.__C.SEGMENT_PATH['sync'], 'r')
        sync_list = sync_file.readlines()
        # print(sync_list)
        self.total_len = int(sync_list[0].strip())
        sync_file.close()

    def load(self, idx):
        json_filename = self.segment_path + self.RUN_MODE + '-' + str(idx) + '.json'
        json_file = open(json_filename, 'r')
        formatted_data = json.load(json_file)
        json_file.close()

        return formatted_data

    def synced_segment_text(self, rank):
        if rank == 0:
            self.segment_text()
            self.write_to_sync()
        dist.barrier()

        if rank != 0:
            self.load_to_sync()
        dist.barrier()
        logging.info('Segment Files Synced')
