# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import errno
import logging
import json
import os
import os.path as op
from tqdm import tqdm


def create_lineidx(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)


class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=False):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file. 
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            create_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            logging.info('{}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx):
        return self.seek_first_column(idx)

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.info('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


class CompositeTSVFile():
    def __init__(self, file_list, seq_file, root='.'):
        if isinstance(file_list, str):
            self.file_list = load_list_file(file_list)
        else:
            assert isinstance(file_list, list)
            self.file_list = file_list

        self.seq_file = seq_file
        self.root = root
        self.initialized = False
        self.initialize()

    def get_key(self, index):
        idx_source, idx_row = self.seq[index]
        k = self.tsvs[idx_source].get_key(idx_row)
        return '_'.join([self.file_list[idx_source], k])

    def num_rows(self):
        return len(self.seq)

    def __getitem__(self, index):
        idx_source, idx_row = self.seq[index]
        return self.tsvs[idx_source].seek(idx_row)

    def __len__(self):
        return len(self.seq)

    def initialize(self):
        '''
        this function has to be called in init function if cache_policy is
        enabled. Thus, let's always call it in init funciton to make it simple.
        '''
        if self.initialized:
            return
        self.seq = []
        with open(self.seq_file, 'r') as fp:
            for line in fp:
                parts = line.strip().split('\t')
                self.seq.append([int(parts[0]), int(parts[1])])
        self.tsvs = [TSVFile(op.join(self.root, f)) for f in self.file_list]
        self.initialized = True


def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_linelist_file(linelist_file):
    if linelist_file is not None:
        line_list = []
        with open(linelist_file, 'r') as fp:
            for i in fp:
                line_list.append(int(i.strip()))
        return line_list


def tsv_writer(values, tsv_file, sep='\t'):
    mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            # this step makes sure python2 and python3 encoded img string are the same.
            # for python2 encoded image string, it is a str class starts with "/".
            # for python3 encoded image string, it is a bytes class starts with "b'/".
            # v.decode('utf-8') converts bytes to str so the content is the same.
            # v.decode('utf-8') should only be applied to bytes class type. 
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format(sep.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_tmp, tsv_file)
    os.rename(lineidx_file_tmp, lineidx_file)


def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]


def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str


def get_line_list(linelist_file=None, num_rows=None):
    if linelist_file is not None:
        return load_linelist_file(linelist_file)

    if num_rows is not None:
        return [i for i in range(num_rows)]


def generate_labelmap_file(label_file, save_file=None):
    rows = tsv_reader(label_file)
    labelmap = []
    for i, row in enumerate(rows):
        labelmap.extend(set([rect['class'] for rect in json.loads(row[1])]))
    labelmap = sorted(list(set(labelmap)))

    save_file = config_save_file(label_file, save_file, '.labelmap.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(labelmap))


def extract_column(tsv_file, col=1, save_file=None):
    rows = tsv_reader(tsv_file)
    def gen_rows():
        for i, row in enumerate(rows):
            row1 = [row[0], row[col]]
            yield row1

    save_file = config_save_file(tsv_file, save_file, '.col.{}.tsv'.format(col))
    tsv_writer(gen_rows(), save_file)


def remove_column(tsv_file, col=1, save_file=None):
    rows = tsv_reader(tsv_file)
    def gen_rows():
        for i, row in enumerate(rows):
            del row[col]
            yield row

    save_file = config_save_file(tsv_file, save_file, '.remove.{}.tsv'.format(col))
    tsv_writer(gen_rows(), save_file)


def generate_linelist_file(label_file, save_file=None, ignore_attrs=()):
    # generate a list of image that has labels
    # images with only ignore labels are not selected. 
    line_list = []
    rows = tsv_reader(label_file)
    for i, row in tqdm(enumerate(rows)):
        labels = json.loads(row[1])
        if labels:
            if ignore_attrs and all([any([lab[attr] for attr in ignore_attrs if attr in lab]) \
                                for lab in labels]):
                continue
            line_list.append([i])

    save_file = config_save_file(label_file, save_file, '.linelist.tsv')
    tsv_writer(line_list, save_file)

