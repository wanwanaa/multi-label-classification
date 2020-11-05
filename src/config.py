class Config():
    def __init__(self):
        self.work_dir = work_dir if work_dir else os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.data_dir = data_dir if data_dir else os.path.join(self.work_dir, 'data')
        self.output_dir = output_dir if output_dir else os.path.join(self.work_dir, "output")

        self.model_filename = os.path.join(self.output_dir, 'model.pkl')

        self.filename_train_txt = os.path.join(self.data_dir, 'test.txt')
        self.filename_valid_txt = os.path.join(self.data_dir, 'test.txt')
        self.filename_test_txt = os.path.join(self.data_dir, 'test.txt')

        self.filename_train_label = os.path.join(self.data_dir, 'test.label')
        self.filename_valid_label = os.path.join(self.data_dir, 'test.label')
        self.filename_test_label = os.path.join(self.data_dir, 'test.label')


        self.hidden_size = 512
        self.bert_size = 768
        self.seq_max_len = 78
        self.mulit_num = 2
        
        self.epoch = 39
        self.batch_size = 64
        self.lr = 0.001