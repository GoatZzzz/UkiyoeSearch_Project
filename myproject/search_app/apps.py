# search_app/apps.py

from django.apps import AppConfig

class SearchAppConfig(AppConfig):
    name = 'search_app'

    def ready(self):
        # 当Django启动时执行以下代码
        import torch
        import faiss
        import pandas as pd
        from transformers import pipeline

        # 导入我们自己定义的全局变量模块
        from . import global_vars

        #######################
        # 1) 加载 mBART 翻译模型
        #######################
        model_path = "/home/zhu01/UkiyoeSearch_Project/models/mbart-large-50-many-to-many-mmt"
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        translator = pipeline(
            task="translation",
            model=model_path,
            tokenizer=model_path,
            src_lang="zh_CN",
            tgt_lang="en_XX",
            device=device
        )
        global_vars.translator = translator
        global_vars.device = device

        #######################
        # 2) 加载 CLIP (ViT-B/32)
        #######################
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        global_vars.clip_model = clip_model
        global_vars.clip_preprocess = clip_preprocess

        #######################
        # 3) 加载 FAISS 索引
        #######################
        faiss_index_path = "/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/index/faiss_index_ivf.index"
        faiss_index = faiss.read_index(faiss_index_path)
        global_vars.faiss_index = faiss_index

        #######################
        # 4) 加载元数据 (CSV)
        #######################
        metadata_path = "/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/merged_metadata.csv"
        metadata_df = pd.read_csv(metadata_path)
        global_vars.metadata_df = metadata_df
