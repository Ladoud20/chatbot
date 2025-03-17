import faiss



class BaseFaissIndex:
        def __init__(self, index_name, dimension, use_gpu=False, index_output_path='./indices/'):
            self.index_name = index_name
            self.dimension = dimension
            self.use_gpu = use_gpu
            self.index = None
            self.index_output_path = index_output_path


        def move_to_gpu(self):
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


        def move_to_cpu(self):
            if self.use_gpu:
                self.index = faiss.index_gpu_to_cpu(self.index)

        def save_index(self):
            faiss.write_index(self.index, self.index_output_path + self.index_name)
            print(f"Index saved as {self.index_name}")


