import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import logging
import clip

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultilingualProcessor:
    def __init__(self, device=None):
        """
        初始化多语言处理器
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing MultilingualProcessor on device: {self.device}")
        
        # 初始化XLM-R模型和分词器
        logger.info("Loading XLM-RoBERTa model and tokenizer...")
        self.xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.xlmr_model = XLMRobertaModel.from_pretrained('xlm-roberta-base').to(self.device)
        self.xlmr_model.eval()  # 设置为评估模式
        
        # 初始化CLIP的文本编码器 - 使用 ViT-L/14 (768维特征)
        logger.info("Loading CLIP model for text encoding...")
        self.clip_model, _ = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()  # 设置为评估模式
        
        # 初始化特征映射层 - 更新维度适配 ViT-L/14
        self.feature_mapping = torch.nn.Linear(768, 768).to(self.device)  # XLM-R: 768, CLIP: 768
        
        logger.info("MultilingualProcessor initialized successfully")

    def __call__(self, text: str) -> torch.Tensor:
        """处理查询文本并返回特征向量"""
        with torch.no_grad():
            # CLIP 特征 (现在是768维)
            text_tokens = clip.tokenize([text]).to(self.device)
            clip_features = self.clip_model.encode_text(text_tokens)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            
            # XLM-R 特征
            inputs = self.xlmr_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            xlmr_outputs = self.xlmr_model(**inputs)
            xlmr_features = xlmr_outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # 映射 XLM-R 特征到 CLIP 空间 (现在都是768维)
            mapped_features = self.feature_mapping(xlmr_features)
            mapped_features = mapped_features / mapped_features.norm(dim=-1, keepdim=True)
            
            # 组合特征 (0.7 CLIP + 0.3 XLM-R)
            combined_features = 0.7 * clip_features + 0.3 * mapped_features
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            return combined_features.cpu().detach()